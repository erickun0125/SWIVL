"""
SE(2) Screw-Decomposed Twist-Driven Impedance Controller

Implements the SWIVL framework's Layer 2 (Reference Twist Field Generator) and 
Layer 4 (Screw Axes-Decomposed Impedance Controller) for SE(2) planar manipulation
with 1-DOF articulated objects.

Key Features:
- Reference twist field generation with pose-error correction
- G-orthogonal projection operators for bulk/internal motion decomposition
- Twist-driven impedance control avoiding explicit SE(2) Jacobian
- Support for learned impedance variables from RL policy (Layer 3)

Following Modern Robotics (Lynch & Park) convention:
- Twist: V = [ω, vx, vy]^T (angular velocity first!)
- Wrench: F = [τ, fx, fy]^T (torque first!)
- Screw axis: B = [sω, sx, sy]^T (angular component first!)

Target dynamics (decomposed):
- Internal motion: d_∥ * (V_∥^ref - V_∥)
- Bulk motion: d_⊥ * (V_⊥^ref - V_⊥)

Reference: SWIVL Paper, Sections 3.2-3.4
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field

from src.se2_math import (
    SE2Pose,
    se2_log,
    se2_inverse,
    se2_adjoint,
    normalize_angle
)
from src.se2_dynamics import SE2Dynamics, SE2RobotParams


@dataclass
class ScrewDecomposedImpedanceParams:
    """
    Impedance parameters for screw-decomposed control.
    
    These correspond to the RL policy action space (Layer 3):
    a_t = (d_l_parallel, d_r_parallel, d_l_perp, d_r_perp, k_p_l, k_p_r, alpha)
    
    For single gripper, we use:
    - d_parallel: Damping for internal motion (along screw axis)
    - d_perp: Damping for bulk motion (orthogonal to screw axis)
    - k_p: Stiffness for pose-error correction in reference twist
    - alpha: Characteristic length for metric tensor G
    """
    d_parallel: float = 10.0      # Damping for internal motion [N·s/m or N·m·s/rad]
    d_perp: float = 50.0          # Damping for bulk motion [N·s/m or N·m·s/rad]
    k_p: float = 5.0              # Stiffness for pose-error correction [1/s]
    alpha: float = 10.0           # Characteristic length [pixels] for metric tensor


class SE2ScrewDecomposedImpedanceController:
    """
    SE(2) Screw-Decomposed Twist-Driven Impedance Controller.
    
    Implements SWIVL's Layer 2 + Layer 4 for a single end-effector:
    
    Layer 2 (Reference Twist Field Generator):
        V_ref = Ad_{T_bd} * V_des + k_p * E
        
    Layer 4 (Screw-Decomposed Impedance Controller):
        K_d = G * (P_∥ * d_∥ + P_⊥ * d_⊥)
        F_cmd = K_d * (V_ref - V) + μ_b
    
    Key features:
    - G-orthogonal projection onto internal/bulk motion subspaces
    - Twist-driven formulation bypassing SE(2) pose-error Jacobian
    - Support for dynamic impedance variable updates from RL policy
    """

    def __init__(
        self,
        screw_axis: np.ndarray,
        params: Optional[ScrewDecomposedImpedanceParams] = None,
        robot_dynamics: Optional[SE2Dynamics] = None,
        max_force: float = 100.0,
        max_torque: float = 500.0
    ):
        """
        Initialize screw-decomposed impedance controller.
        
        Args:
            screw_axis: Body-frame screw axis B = [sω, sx, sy] ∈ R³ (MR convention)
                       For revolute joint: [1, 0, 0]
                       For prismatic along x: [0, 1, 0]
            params: Impedance parameters (uses defaults if None)
            robot_dynamics: SE2Dynamics for Coriolis compensation (optional)
            max_force: Maximum force magnitude [N]
            max_torque: Maximum torque magnitude [N·m]
        """
        assert screw_axis.shape == (3,), "Screw axis must be 3D vector [sω, sx, sy]"
        
        self.screw_axis = screw_axis.copy()
        self.params = params if params is not None else ScrewDecomposedImpedanceParams()
        self.robot_dynamics = robot_dynamics
        self.max_force = max_force
        self.max_torque = max_torque
        
        # Pre-compute projection operators with current alpha
        self._update_projection_operators()
        
        # Debug storage
        self.last_reference_twist = None
        self.last_pose_error = None
        self.last_velocity_error = None
        self.last_control_wrench = None

    def _update_projection_operators(self):
        """
        Compute G-orthogonal projection operators.
        
        G = diag(α², 1, 1)  (metric tensor)
        P_∥ = B (B^T G B)^{-1} B^T G  (project onto internal motion)
        P_⊥ = I - P_∥  (project onto bulk motion)
        
        These projectors satisfy:
        - G-self-adjointness: P^T G = G P
        - Orthogonality: P_∥ P_⊥ = 0
        - Partition of identity: P_∥ + P_⊥ = I
        """
        alpha = self.params.alpha
        B = self.screw_axis
        
        # Metric tensor G = diag(α², 1, 1)
        self.G = np.diag([alpha**2, 1.0, 1.0])
        
        # B^T G B (scalar for 1-DOF)
        BtGB = B @ self.G @ B
        
        # Avoid division by zero
        epsilon = 1e-10
        BtGB_inv = 1.0 / (BtGB + epsilon)
        
        # P_∥ = B (B^T G B)^{-1} B^T G
        # For 1-DOF: P_∥ = (B ⊗ B^T G) / (B^T G B)
        self.P_parallel = np.outer(B, B @ self.G) * BtGB_inv
        
        # P_⊥ = I - P_∥
        self.P_perp = np.eye(3) - self.P_parallel

    def set_screw_axis(self, screw_axis: np.ndarray):
        """
        Update screw axis (for configuration-dependent screws).
        
        Args:
            screw_axis: New screw axis [sω, sx, sy] (MR convention)
        """
        self.screw_axis = screw_axis.copy()
        self._update_projection_operators()

    def set_impedance_params(self, params: ScrewDecomposedImpedanceParams):
        """
        Update impedance parameters (called by RL policy at each step).
        
        Args:
            params: New impedance parameters
        """
        # Check if alpha changed (requires projection update)
        alpha_changed = (self.params.alpha != params.alpha)
        
        self.params = params
        
        if alpha_changed:
            self._update_projection_operators()

    def set_impedance_variables(
        self,
        d_parallel: float,
        d_perp: float,
        k_p: float,
        alpha: float
    ):
        """
        Update individual impedance variables (convenience method for RL policy).
        
        Args:
            d_parallel: Damping for internal motion
            d_perp: Damping for bulk motion
            k_p: Stiffness for pose-error correction
            alpha: Characteristic length for metric tensor
        """
        self.set_impedance_params(ScrewDecomposedImpedanceParams(
            d_parallel=d_parallel,
            d_perp=d_perp,
            k_p=k_p,
            alpha=alpha
        ))

    def compute_pose_error(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray
    ) -> np.ndarray:
        """
        Compute pose error via SE(2) logarithm map.
        
        E = log(T_bd)^∨ where T_bd = T_sb^{-1} T_sd
        
        This gives pose error expressed in the body frame.
        
        Args:
            current_pose: Current pose T_sb as [x, y, theta]
            desired_pose: Desired pose T_sd as [x, y, theta]
            
        Returns:
            E: Pose error [ω_e, vx_e, vy_e] in body frame (MR convention)
        """
        T_sb = SE2Pose.from_array(current_pose).to_matrix()
        T_sd = SE2Pose.from_array(desired_pose).to_matrix()
        
        # T_bd = T_sb^{-1} @ T_sd
        T_bd = se2_inverse(T_sb) @ T_sd
        
        # E = log(T_bd)^∨
        E = se2_log(T_bd)  # Returns [ω, vx, vy] in MR convention
        
        self.last_pose_error = E
        return E

    def compute_reference_twist(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        desired_twist: np.ndarray
    ) -> np.ndarray:
        """
        Compute reference twist using Stable Imitation Vector Field (Layer 2).
        
        V_ref = Ad_{T_bd} * V_des + k_p * E
        
        where:
        - T_bd = T_sb^{-1} T_sd (current to desired transformation)
        - V_des: desired body twist in frame {d}
        - E: pose error from logarithm map
        - k_p: stiffness coefficient
        
        Args:
            current_pose: Current pose T_sb as [x, y, theta]
            desired_pose: Desired pose T_sd as [x, y, theta]
            desired_twist: Desired body twist V_des [ω, vx, vy] in desired frame
            
        Returns:
            V_ref: Reference twist [ω, vx, vy] in current body frame
        """
        # Compute pose error E = log(T_bd)^∨
        E = self.compute_pose_error(current_pose, desired_pose)
        
        # Compute transformation T_bd = T_sb^{-1} @ T_sd
        T_sb = SE2Pose.from_array(current_pose).to_matrix()
        T_sd = SE2Pose.from_array(desired_pose).to_matrix()
        T_bd = se2_inverse(T_sb) @ T_sd
        
        # Adjoint transformation Ad_{T_bd}
        Ad_bd = se2_adjoint(T_bd)
        
        # Transform desired twist to current body frame
        # V_des is in desired frame {d}, we need it in body frame {b}
        V_des_in_body = Ad_bd @ desired_twist
        
        # Reference twist with pose-error correction
        # V_ref = Ad_{T_bd} * V_des + k_p * E
        V_ref = V_des_in_body + self.params.k_p * E
        
        self.last_reference_twist = V_ref
        return V_ref

    def compute_damping_matrix(self) -> np.ndarray:
        """
        Compute damping matrix with screw decomposition.
        
        K_d = G * (P_∥ * d_∥ + P_⊥ * d_⊥)
        
        Returns:
            K_d: Damping matrix (3x3)
        """
        K_d = self.G @ (
            self.P_parallel * self.params.d_parallel +
            self.P_perp * self.params.d_perp
        )
        return K_d

    def decompose_twist(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose twist into parallel and perpendicular components.
        
        V_∥ = P_∥ V  (internal motion, along screw axis)
        V_⊥ = P_⊥ V  (bulk motion, orthogonal to screw axis)
        
        Args:
            V: Twist [ω, vx, vy]
            
        Returns:
            (V_parallel, V_perp): Decomposed twists
        """
        V_parallel = self.P_parallel @ V
        V_perp = self.P_perp @ V
        return V_parallel, V_perp

    def decompose_wrench(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose wrench into parallel and perpendicular components.
        
        By twist-wrench duality:
        F_∥ = P_∥^T F  (productive wrench, performs work along internal motion)
        F_⊥ = P_⊥^T F  (fighting wrench, orthogonal to internal motion)
        
        Args:
            F: Wrench [τ, fx, fy]
            
        Returns:
            (F_parallel, F_perp): Decomposed wrenches
        """
        F_parallel = self.P_parallel.T @ F
        F_perp = self.P_perp.T @ F
        return F_parallel, F_perp

    def compute_control(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        current_twist: np.ndarray,
        desired_twist: np.ndarray,
        external_wrench: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete impedance control computation.
        
        This is the main interface combining Layer 2 + Layer 4.
        
        Control law:
            V_ref = Ad_{T_bd} * V_des + k_p * E        (Layer 2)
            K_d = G * (P_∥ * d_∥ + P_⊥ * d_⊥)         (Layer 4)
            F_cmd = K_d * (V_ref - V) + μ_b           (Layer 4)
        
        Following Modern Robotics convention:
        - Twist: [ω, vx, vy]
        - Wrench: [τ, fx, fy]
        
        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]
            current_twist: Current body twist [ω, vx, vy]
            desired_twist: Desired body twist [ω, vx, vy] in desired frame
            external_wrench: External wrench [τ, fx, fy] (for info only)
            
        Returns:
            F_cmd: Control wrench [τ, fx, fy]
            info: Dictionary with debugging information
        """
        # ========================================
        # Layer 2: Reference Twist Field Generator
        # ========================================
        V_ref = self.compute_reference_twist(current_pose, desired_pose, desired_twist)
        
        # ========================================
        # Layer 4: Screw-Decomposed Impedance Control
        # ========================================
        
        # Compute velocity error
        V_error = V_ref - current_twist
        self.last_velocity_error = V_error
        
        # Compute damping matrix
        K_d = self.compute_damping_matrix()
        
        # Main impedance feedback term
        F_feedback = K_d @ V_error
        
        # Coriolis/centrifugal compensation (if dynamics model available)
        mu_b = np.zeros(3)
        if self.robot_dynamics is not None:
            C_b = self.robot_dynamics.compute_coriolis_matrix(current_twist)
            mu_b = C_b @ current_twist
        
        # Control wrench
        F_cmd = F_feedback + mu_b
        
        # Saturate wrench
        F_cmd = self._saturate_wrench(F_cmd)
        
        self.last_control_wrench = F_cmd
        
        # ========================================
        # Collect debug/analysis info
        # ========================================
        
        # Decompose for analysis
        V_ref_parallel, V_ref_perp = self.decompose_twist(V_ref)
        V_parallel, V_perp = self.decompose_twist(current_twist)
        
        F_cmd_parallel, F_cmd_perp = self.decompose_wrench(F_cmd)
        
        # External wrench decomposition (if provided)
        F_ext_parallel = np.zeros(3)
        F_ext_perp = np.zeros(3)
        if external_wrench is not None:
            F_ext_parallel, F_ext_perp = self.decompose_wrench(external_wrench)
        
        info = {
            # Errors
            'pose_error': self.last_pose_error,
            'velocity_error': V_error,
            'pose_error_norm': np.linalg.norm(self.last_pose_error),
            'velocity_error_norm': np.linalg.norm(V_error),
            
            # Reference twist
            'reference_twist': V_ref,
            'reference_twist_parallel': V_ref_parallel,
            'reference_twist_perp': V_ref_perp,
            
            # Current twist decomposition
            'current_twist_parallel': V_parallel,
            'current_twist_perp': V_perp,
            
            # Control wrench
            'control_wrench': F_cmd,
            'control_wrench_parallel': F_cmd_parallel,
            'control_wrench_perp': F_cmd_perp,
            
            # External wrench decomposition
            'external_wrench_parallel': F_ext_parallel,
            'external_wrench_perp': F_ext_perp,
            'fighting_force': np.linalg.norm(F_ext_perp),
            
            # Impedance parameters
            'screw_axis': self.screw_axis,
            'd_parallel': self.params.d_parallel,
            'd_perp': self.params.d_perp,
            'k_p': self.params.k_p,
            'alpha': self.params.alpha,
        }
        
        return F_cmd, info

    def _saturate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """
        Saturate wrench to maximum limits.
        
        Following MR convention: [τ, fx, fy]
        
        Args:
            wrench: Input wrench [τ, fx, fy]
            
        Returns:
            Saturated wrench [τ, fx, fy]
        """
        wrench_out = wrench.copy()
        
        # Saturate torque (first element in MR convention)
        wrench_out[0] = np.clip(wrench[0], -self.max_torque, self.max_torque)
        
        # Saturate force magnitude (elements 1, 2 in MR convention)
        force_mag = np.linalg.norm(wrench[1:3])
        if force_mag > self.max_force:
            wrench_out[1:3] = wrench[1:3] / force_mag * self.max_force
        
        return wrench_out

    def reset(self):
        """Reset controller state."""
        self.last_reference_twist = None
        self.last_pose_error = None
        self.last_velocity_error = None
        self.last_control_wrench = None


class MultiGripperSE2ScrewDecomposedImpedanceController:
    """
    Multi-gripper SE(2) Screw-Decomposed Impedance Controller.
    
    Manages controllers for bimanual manipulation, with per-arm
    impedance variables as specified in SWIVL's action space:
    
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7
    
    Note: α is shared between both arms (single learned metric tensor).
    """

    def __init__(
        self,
        num_grippers: int = 2,
        screw_axes: Optional[np.ndarray] = None,
        robot_params: Optional[SE2RobotParams] = None,
        default_params: Optional[ScrewDecomposedImpedanceParams] = None,
        max_force: float = 100.0,
        max_torque: float = 500.0
    ):
        """
        Initialize multi-gripper controller.
        
        Args:
            num_grippers: Number of grippers (default: 2 for bimanual)
            screw_axes: Initial screw axes (N, 3) or None
            robot_params: Robot physical parameters
            default_params: Default impedance parameters
            max_force: Maximum force per gripper
            max_torque: Maximum torque per gripper
        """
        self.num_grippers = num_grippers
        self.robot_params = robot_params
        self.max_force = max_force
        self.max_torque = max_torque
        
        # Default screw axes (revolute joint)
        if screw_axes is None:
            # Default: pure rotation [1, 0, 0] for both
            screw_axes = np.array([[1.0, 0.0, 0.0]] * num_grippers)
        
        self.screw_axes = screw_axes
        
        # Default impedance parameters
        if default_params is None:
            default_params = ScrewDecomposedImpedanceParams()
        
        # Create individual controllers
        self.controllers = []
        for i in range(num_grippers):
            # Create dynamics if robot_params provided
            dynamics = SE2Dynamics(robot_params) if robot_params else None
            
            controller = SE2ScrewDecomposedImpedanceController(
                screw_axis=screw_axes[i],
                params=default_params,
                robot_dynamics=dynamics,
                max_force=max_force,
                max_torque=max_torque
            )
            self.controllers.append(controller)

    def set_screw_axes(self, screw_axes: np.ndarray):
        """
        Update screw axes for all grippers.
        
        Args:
            screw_axes: (N, 3) array of screw axes [sω, sx, sy]
        """
        self.screw_axes = screw_axes.copy()
        for i, controller in enumerate(self.controllers):
            controller.set_screw_axis(screw_axes[i])

    def set_impedance_variables(
        self,
        d_l_parallel: float,
        d_r_parallel: float,
        d_l_perp: float,
        d_r_perp: float,
        k_p_l: float,
        k_p_r: float,
        alpha: float
    ):
        """
        Set impedance variables from RL policy action.
        
        Action space: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)
        
        Args:
            d_l_parallel: Left arm damping for internal motion
            d_r_parallel: Right arm damping for internal motion
            d_l_perp: Left arm damping for bulk motion
            d_r_perp: Right arm damping for bulk motion
            k_p_l: Left arm stiffness for pose-error correction
            k_p_r: Right arm stiffness for pose-error correction
            alpha: Shared characteristic length (metric tensor)
        """
        # Left arm (index 0)
        self.controllers[0].set_impedance_variables(
            d_parallel=d_l_parallel,
            d_perp=d_l_perp,
            k_p=k_p_l,
            alpha=alpha
        )
        
        # Right arm (index 1)
        self.controllers[1].set_impedance_variables(
            d_parallel=d_r_parallel,
            d_perp=d_r_perp,
            k_p=k_p_r,
            alpha=alpha
        )

    def set_impedance_variables_array(self, action: np.ndarray):
        """
        Set impedance variables from action array.
        
        Args:
            action: (7,) array [d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α]
        """
        self.set_impedance_variables(
            d_l_parallel=action[0],
            d_r_parallel=action[1],
            d_l_perp=action[2],
            d_r_perp=action[3],
            k_p_l=action[4],
            k_p_r=action[5],
            alpha=action[6]
        )

    def compute_wrenches(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray,
        current_twists: np.ndarray,
        desired_twists: np.ndarray,
        external_wrenches: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute wrenches for all grippers.
        
        Args:
            current_poses: (N, 3) current poses [x, y, theta]
            desired_poses: (N, 3) desired poses [x, y, theta]
            current_twists: (N, 3) current body twists [ω, vx, vy]
            desired_twists: (N, 3) desired body twists [ω, vx, vy]
            external_wrenches: (N, 3) external wrenches [τ, fx, fy] (optional)
            
        Returns:
            wrenches: (N, 3) control wrenches [τ, fx, fy]
            info: Combined debug information
        """
        if current_poses.shape[0] != self.num_grippers:
            raise ValueError(f"Expected {self.num_grippers} poses, got {current_poses.shape[0]}")
        
        wrenches = []
        infos = []
        
        for i in range(self.num_grippers):
            ext_wrench = external_wrenches[i] if external_wrenches is not None else None
            
            wrench, info = self.controllers[i].compute_control(
                current_poses[i],
                desired_poses[i],
                current_twists[i],
                desired_twists[i],
                ext_wrench
            )
            wrenches.append(wrench)
            infos.append(info)
        
        # Aggregate info
        combined_info = {
            'left': infos[0],
            'right': infos[1] if self.num_grippers > 1 else None,
            'total_fighting_force': sum(info['fighting_force'] for info in infos),
        }
        
        return np.array(wrenches), combined_info

    def decompose_external_wrenches(
        self,
        external_wrenches: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose external wrenches into parallel and perpendicular components.
        
        Useful for reward computation and analysis.
        
        Args:
            external_wrenches: (N, 3) external wrenches [τ, fx, fy]
            
        Returns:
            F_parallel: (N, 3) productive wrenches
            F_perp: (N, 3) fighting wrenches
        """
        F_parallel = []
        F_perp = []
        
        for i, controller in enumerate(self.controllers):
            F_par, F_per = controller.decompose_wrench(external_wrenches[i])
            F_parallel.append(F_par)
            F_perp.append(F_per)
        
        return np.array(F_parallel), np.array(F_perp)

    def reset(self):
        """Reset all controllers."""
        for controller in self.controllers:
            controller.reset()

    def get_controller(self, idx: int) -> SE2ScrewDecomposedImpedanceController:
        """Get individual controller by index."""
        return self.controllers[idx]


# ============================================================================
# Utility functions for projection operator verification
# ============================================================================

def verify_projection_properties(
    P_parallel: np.ndarray,
    P_perp: np.ndarray,
    G: np.ndarray
) -> Dict[str, bool]:
    """
    Verify mathematical properties of G-orthogonal projection operators.
    
    Properties:
    1. Completeness: P_∥ + P_⊥ = I
    2. Idempotency: P² = P
    3. Orthogonality: P_∥ P_⊥ = 0
    4. G-self-adjointness: P^T G = G P
    
    Args:
        P_parallel: Parallel projection operator
        P_perp: Perpendicular projection operator
        G: Metric tensor
        
    Returns:
        Dictionary with property verification results
    """
    tol = 1e-10
    I = np.eye(3)
    
    results = {
        'completeness': np.allclose(P_parallel + P_perp, I, atol=tol),
        'idempotent_parallel': np.allclose(P_parallel @ P_parallel, P_parallel, atol=tol),
        'idempotent_perp': np.allclose(P_perp @ P_perp, P_perp, atol=tol),
        'orthogonality': np.allclose(P_parallel @ P_perp, np.zeros((3, 3)), atol=tol),
        'G_self_adjoint_parallel': np.allclose(P_parallel.T @ G, G @ P_parallel, atol=tol),
        'G_self_adjoint_perp': np.allclose(P_perp.T @ G, G @ P_perp, atol=tol),
    }
    
    results['all_valid'] = all(results.values())
    
    return results


# ============================================================================
# Test and demonstration code
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SE(2) Screw-Decomposed Twist-Driven Impedance Controller")
    print("SWIVL Layer 2 + Layer 4 Implementation Test")
    print("=" * 70)
    
    # Test 1: Projection operators with metric tensor
    print("\nTest 1: G-Orthogonal Projection Operators")
    print("-" * 70)
    
    # Revolute joint screw axis
    B_revolute = np.array([1.0, 0.0, 0.0])  # Pure rotation
    alpha = 10.0  # Characteristic length (pixels)
    
    params = ScrewDecomposedImpedanceParams(
        d_parallel=10.0,
        d_perp=50.0,
        k_p=5.0,
        alpha=alpha
    )
    
    controller = SE2ScrewDecomposedImpedanceController(
        screw_axis=B_revolute,
        params=params
    )
    
    print(f"Screw axis (revolute): {B_revolute}")
    print(f"Alpha (characteristic length): {alpha}")
    print(f"\nMetric tensor G:\n{controller.G}")
    print(f"\nP_parallel:\n{controller.P_parallel}")
    print(f"\nP_perp:\n{controller.P_perp}")
    
    # Verify properties
    props = verify_projection_properties(
        controller.P_parallel,
        controller.P_perp,
        controller.G
    )
    print(f"\nProjection properties: {props}")
    
    # Test 2: Twist decomposition
    print("\n\nTest 2: Twist Decomposition")
    print("-" * 70)
    
    test_twist = np.array([0.5, 2.0, 1.0])  # [ω, vx, vy]
    V_par, V_perp = controller.decompose_twist(test_twist)
    
    print(f"Test twist: {test_twist}")
    print(f"V_parallel (internal): {V_par}")
    print(f"V_perp (bulk): {V_perp}")
    print(f"Reconstruction: {V_par + V_perp}")
    print(f"Match: {np.allclose(test_twist, V_par + V_perp)}")
    
    # G-orthogonality check
    g_inner = V_par @ controller.G @ V_perp
    print(f"G-inner product <V_∥, V_⊥>_G: {g_inner:.10f}")
    
    # Test 3: Reference twist computation
    print("\n\nTest 3: Reference Twist Field Generation (Layer 2)")
    print("-" * 70)
    
    current_pose = np.array([100.0, 100.0, 0.0])
    desired_pose = np.array([120.0, 110.0, 0.2])
    desired_twist = np.array([0.1, 5.0, 2.0])  # [ω, vx, vy] in desired frame
    
    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Desired twist (in desired frame): {desired_twist}")
    
    V_ref = controller.compute_reference_twist(current_pose, desired_pose, desired_twist)
    print(f"\nReference twist V_ref: {V_ref}")
    print(f"Pose error E: {controller.last_pose_error}")
    
    # Test 4: Full control computation
    print("\n\nTest 4: Complete Control Computation")
    print("-" * 70)
    
    current_twist = np.array([0.0, 1.0, 0.5])  # [ω, vx, vy]
    external_wrench = np.array([5.0, 10.0, 8.0])  # [τ, fx, fy]
    
    F_cmd, info = controller.compute_control(
        current_pose,
        desired_pose,
        current_twist,
        desired_twist,
        external_wrench
    )
    
    print(f"Current twist: {current_twist}")
    print(f"External wrench: {external_wrench}")
    print(f"\nControl wrench F_cmd: {F_cmd}")
    print(f"  Torque:  {F_cmd[0]:.4f} N·m")
    print(f"  Force X: {F_cmd[1]:.4f} N")
    print(f"  Force Y: {F_cmd[2]:.4f} N")
    
    print(f"\nVelocity error: {info['velocity_error']}")
    print(f"Control wrench (parallel): {info['control_wrench_parallel']}")
    print(f"Control wrench (perp): {info['control_wrench_perp']}")
    print(f"Fighting force (||F_ext_⊥||): {info['fighting_force']:.4f} N")
    
    # Test 5: Multi-gripper controller
    print("\n\nTest 5: Multi-Gripper Controller")
    print("-" * 70)
    
    robot_params = SE2RobotParams(mass=1.2, inertia=97.6)
    
    multi_controller = MultiGripperSE2ScrewDecomposedImpedanceController(
        num_grippers=2,
        screw_axes=np.array([
            [1.0, 0.0, 0.0],  # Left: revolute
            [1.0, 0.0, 0.0],  # Right: revolute
        ]),
        robot_params=robot_params
    )
    
    # Set impedance variables (simulating RL policy output)
    multi_controller.set_impedance_variables(
        d_l_parallel=10.0,
        d_r_parallel=10.0,
        d_l_perp=50.0,
        d_r_perp=50.0,
        k_p_l=5.0,
        k_p_r=5.0,
        alpha=10.0
    )
    
    # Test inputs
    current_poses = np.array([
        [100.0, 100.0, 0.0],
        [200.0, 100.0, np.pi]
    ])
    desired_poses = np.array([
        [120.0, 110.0, 0.2],
        [180.0, 110.0, np.pi - 0.2]
    ])
    current_twists = np.array([
        [0.0, 1.0, 0.5],
        [0.0, -1.0, 0.5]
    ])
    desired_twists = np.array([
        [0.1, 5.0, 2.0],
        [-0.1, -5.0, 2.0]
    ])
    external_wrenches = np.array([
        [5.0, 10.0, 8.0],
        [-5.0, -10.0, 8.0]
    ])
    
    wrenches, info = multi_controller.compute_wrenches(
        current_poses,
        desired_poses,
        current_twists,
        desired_twists,
        external_wrenches
    )
    
    print(f"Left wrench:  {wrenches[0]}")
    print(f"Right wrench: {wrenches[1]}")
    print(f"Total fighting force: {info['total_fighting_force']:.4f} N")
    
    # Test wrench decomposition
    F_par, F_perp = multi_controller.decompose_external_wrenches(external_wrenches)
    print(f"\nExternal wrench decomposition:")
    print(f"  Left F_∥: {F_par[0]}, F_⊥: {F_perp[0]}")
    print(f"  Right F_∥: {F_par[1]}, F_⊥: {F_perp[1]}")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)
