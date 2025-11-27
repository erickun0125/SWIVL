"""
SE(2) Screw-Decomposed Impedance Controller

Implements screw-axis based decomposed impedance control on SE(2).
Decomposes impedance behavior into:
- Parallel subspace (1D): compliant/stiff along screw axis
- Perpendicular subspace (2D): compliant/stiff perpendicular to screw

This allows independent impedance tuning for different directions.

Following Modern Robotics (Lynch & Park) convention:
- Twist: V = [ω, vx, vy]^T (angular velocity first!)
- Wrench: F = [τ, fx, fy]^T (torque first!)
- Screw axis: S = [sω, sx, sy]^T (angular component first!)

Target dynamics:
- Parallel:      M_∥ θ̈ + D_∥ θ̇ + K_∥ θ = τ_ext
- Perpendicular: M_⊥ ë_⊥ + D_⊥ V_e,⊥ + K_⊥ e_⊥ = F_ext,⊥

where θ is generalized displacement along screw axis.

Reference: Based on SE(3) screw decomposition theory
          Applied to SE(2) planar robotics
          Modern Robotics, Lynch & Park, Chapter 3
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from src.se2_math import (
    SE2Pose,
    se2_log,
    se2_inverse,
    normalize_angle
)
from src.se2_dynamics import SE2Dynamics, SE2RobotParams


@dataclass
class ScrewImpedanceParams:
    """
    Screw-decomposed impedance parameters.

    Parallel subspace (1D along screw):
        - M_parallel: Inertia along screw
        - D_parallel: Damping along screw
        - K_parallel: Stiffness along screw

    Perpendicular subspace (2D perpendicular to screw):
        - M_perpendicular: Inertia perpendicular to screw
        - D_perpendicular: Damping perpendicular to screw
        - K_perpendicular: Stiffness perpendicular to screw
    """
    M_parallel: float = 1.0          # [kg] or [kg⋅m²]
    D_parallel: float = 10.0         # [N⋅s/m] or [N⋅m⋅s/rad]
    K_parallel: float = 50.0         # [N/m] or [N⋅m/rad]

    M_perpendicular: float = 1.0     # [kg] or [kg⋅m²]
    D_perpendicular: float = 10.0    # [N⋅s/m] or [N⋅m⋅s/rad]
    K_perpendicular: float = 50.0    # [N/m] or [N⋅m/rad]


class SE2ScrewDecomposedImpedanceController:
    """
    SE(2) Screw-Decomposed Impedance Controller.

    Decomposes control into 1D parallel + 2D perpendicular subspaces.

    Key features:
    - Independent impedance tuning for each subspace
    - Screw axis can be configuration-dependent or fixed
    - Proper SE(2) dynamics compensation
    - Model matching mode for passivity
    """

    def __init__(self,
                 screw_axis: np.ndarray,
                 params: ScrewImpedanceParams,
                 robot_dynamics: SE2Dynamics,
                 model_matching: bool = True,
                 use_feedforward: bool = True,
                 max_force: float = 100.0,
                 max_torque: float = 50.0):
        """
        Initialize screw-decomposed impedance controller.

        Following Modern Robotics convention:
        Screw axis has same ordering as twist: [sω, sx, sy]

        Args:
            screw_axis: SE(2) screw axis [sω, sx, sy] ∈ R³ (MR convention: angular first!)
                       Should be a unit screw (normalized appropriately)
                       Examples:
                       - Pure rotation: [1, 0, 0]
                       - Pure translation x: [0, 1, 0]
                       - Pure translation y: [0, 0, 1]
            params: Screw impedance parameters
            robot_dynamics: SE2Dynamics object
            model_matching: If True, use M_d = Lambda_b
            use_feedforward: If True, include acceleration feedforward
            max_force: Maximum force magnitude [N]
            max_torque: Maximum torque magnitude [N⋅m]
        """
        assert screw_axis.shape == (3,), "Screw axis must be 3D vector"

        self.screw_axis = screw_axis.copy()
        self.params = params
        self.robot_dynamics = robot_dynamics
        self.model_matching = model_matching
        self.use_feedforward = use_feedforward
        self.max_force = max_force
        self.max_torque = max_torque

        # Compute screw norm (S^T S)
        self.screw_norm_sq = np.dot(screw_axis, screw_axis)

        # Pre-compute projection operators
        self._compute_projection_operators()

        # Storage for debugging
        self.last_theta = None
        self.last_theta_dot = None
        self.last_e_perp = None
        self.last_V_e_perp = None
        self.last_tau_ext = None

    def _compute_projection_operators(self):
        """
        Compute projection operators for screw decomposition.

        P_∥ = (S S^T) / (S^T S)
        P_⊥ = I - P_∥
        """
        S = self.screw_axis
        S_norm_sq = self.screw_norm_sq

        # Parallel projection: P_∥ = (S S^T) / (S^T S)
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        self.P_parallel = np.outer(S, S) / (S_norm_sq + epsilon)

        # Perpendicular projection: P_⊥ = I - P_∥
        self.P_perpendicular = np.eye(3) - self.P_parallel

    def set_screw_axis(self, screw_axis: np.ndarray):
        """
        Update screw axis (for configuration-dependent screws).

        Following Modern Robotics convention: [sω, sx, sy]

        Args:
            screw_axis: New screw axis [sω, sx, sy] (MR convention: angular first!)
        """
        self.screw_axis = screw_axis.copy()
        self.screw_norm_sq = np.dot(screw_axis, screw_axis)
        self._compute_projection_operators()

    def decompose_vector(self, v: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Decompose vector into parallel and perpendicular components.

        v = v_∥ + v_⊥ = θ S + v_⊥

        Args:
            v: Vector to decompose [3,]

        Returns:
            theta: Scalar coordinate along screw
            v_parallel: Parallel component [3,]
            v_perp: Perpendicular component [3,]
        """
        # Scalar coordinate: θ = (v^T S) / (S^T S)
        epsilon = 1e-10
        theta = np.dot(v, self.screw_axis) / (self.screw_norm_sq + epsilon)

        # Parallel component: v_∥ = θ S
        v_parallel = theta * self.screw_axis

        # Perpendicular component: v_⊥ = v - θ S
        v_perp = v - v_parallel

        return theta, v_parallel, v_perp

    def compute_pose_error(self,
                          current_pose: np.ndarray,
                          desired_pose: np.ndarray) -> np.ndarray:
        """
        Compute pose error via logarithm map.

        e = log(T_bd)^∨ where T_bd = T_sb^(-1) T_sd

        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]

        Returns:
            e: Pose error [3,]
        """
        T_sb = SE2Pose.from_array(current_pose).to_matrix()
        T_sd = SE2Pose.from_array(desired_pose).to_matrix()
        T_bd = se2_inverse(T_sb) @ T_sd
        e = se2_log(T_bd)
        return e

    def compute_control(self,
                       current_pose: np.ndarray,
                       desired_pose: np.ndarray,
                       body_twist_current: np.ndarray,
                       body_twist_desired: np.ndarray,
                       body_accel_desired: Optional[np.ndarray] = None,
                       F_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Compute screw-decomposed impedance control.

        Following Modern Robotics convention:
        - Twist: [ω, vx, vy]
        - Wrench: [τ, fx, fy]
        - Screw axis: [sω, sx, sy]

        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]
            body_twist_current: Current body twist [ω, vx, vy] (MR convention!)
            body_twist_desired: Desired body twist [ω, vx, vy]
            body_accel_desired: Desired body acceleration [dω, dvx, dvy] [optional]
            F_ext: External wrench [τ, fx, fy] (MR convention!) [optional]

        Returns:
            F_cmd: Control wrench [τ, fx, fy] (MR convention!)
            info: Debug information dictionary
        """
        # Default values
        if body_accel_desired is None:
            body_accel_desired = np.zeros(3)
        if F_ext is None:
            F_ext = np.zeros(3)

        # ========================================
        # Step 1: Compute errors
        # ========================================
        e = self.compute_pose_error(current_pose, desired_pose)
        V_e = body_twist_desired - body_twist_current

        # ========================================
        # Step 2: Decompose errors using screw
        # ========================================
        theta, e_parallel, e_perp = self.decompose_vector(e)
        theta_dot, V_e_parallel, V_e_perp = self.decompose_vector(V_e)
        tau_ext, F_ext_parallel, F_ext_perp = self.decompose_vector(F_ext)

        # Store for debugging
        self.last_theta = theta
        self.last_theta_dot = theta_dot
        self.last_e_perp = e_perp
        self.last_V_e_perp = V_e_perp
        self.last_tau_ext = tau_ext

        # ========================================
        # Step 3: Decompose desired acceleration
        # ========================================
        dV_d_scalar, dV_d_parallel, dV_d_perp = self.decompose_vector(body_accel_desired)

        # ========================================
        # Step 4: Compute commanded acceleration
        # ========================================
        # From decomposed impedance dynamics:
        # θ̈_cmd = dV_d_scalar + (1/M_∥)(D_∥ θ̇ + K_∥ θ - τ_ext)
        # ë_⊥_cmd = dV_d_⊥ + (1/M_⊥)(D_⊥ V_e,⊥ + K_⊥ e_⊥ - F_ext,⊥)

        # Parallel commanded acceleration (scalar)
        if self.model_matching:
            # Model matching: simpler form
            ddot_theta_cmd = dV_d_scalar
            impedance_parallel = (self.params.D_parallel * theta_dot +
                                 self.params.K_parallel * theta)
        else:
            # General case
            impedance_parallel = (self.params.D_parallel * theta_dot +
                                 self.params.K_parallel * theta -
                                 tau_ext)
            ddot_theta_cmd = dV_d_scalar + impedance_parallel / self.params.M_parallel

        # Perpendicular commanded acceleration (2D)
        if self.model_matching:
            # Model matching: simpler form
            ddot_V_perp_cmd = dV_d_perp
            impedance_perp = (self.params.D_perpendicular * V_e_perp +
                            self.params.K_perpendicular * e_perp)
        else:
            # General case
            impedance_perp = (self.params.D_perpendicular * V_e_perp +
                            self.params.K_perpendicular * e_perp -
                            F_ext_perp)
            ddot_V_perp_cmd = dV_d_perp + impedance_perp / self.params.M_perpendicular

        # Reconstruct full commanded acceleration
        if self.use_feedforward:
            dV_cmd_parallel = ddot_theta_cmd * self.screw_axis
            dV_cmd_perp = ddot_V_perp_cmd
            dV_cmd = dV_cmd_parallel + dV_cmd_perp
        else:
            dV_cmd = np.zeros(3)

        # ========================================
        # Step 5: Get robot dynamics
        # ========================================
        Lambda_b = self.robot_dynamics.get_task_space_inertia(current_pose)
        C_b = self.robot_dynamics.compute_coriolis_matrix(body_twist_current)
        eta_b = self.robot_dynamics.gravity_wrench(current_pose)

        # ========================================
        # Step 6: Compute control wrench
        # ========================================
        if self.model_matching:
            # Model matching: F_cmd = Lambda_b dV_cmd + C_b V + eta_b + D_d V_e + K_d e
            # where D_d and K_d are decomposed

            # Impedance feedback term (decomposed)
            impedance_feedback = impedance_parallel * self.screw_axis + impedance_perp

            F_cmd = (Lambda_b @ dV_cmd +
                    C_b @ body_twist_current +
                    eta_b +
                    impedance_feedback)
        else:
            # General case: F_cmd = Lambda_b dV_cmd + C_b V + eta_b - (I + Lambda_b M_d^-1) F_ext
            # This requires computing M_d^-1 in decomposed form

            # Construct M_d^-1 using decomposition
            # M_d^-1 = (1/M_∥) P_∥ + (1/M_⊥) P_⊥
            M_d_inv = ((1.0 / self.params.M_parallel) * self.P_parallel +
                      (1.0 / self.params.M_perpendicular) * self.P_perpendicular)

            F_cmd = (Lambda_b @ dV_cmd +
                    C_b @ body_twist_current +
                    eta_b -
                    (np.eye(3) + Lambda_b @ M_d_inv) @ F_ext)

        # ========================================
        # Step 7: Saturate wrench
        # ========================================
        F_cmd = self._saturate_wrench(F_cmd)

        # ========================================
        # Collect debug info
        # ========================================
        info = {
            'pose_error': e,
            'velocity_error': V_e,
            'theta': theta,
            'theta_dot': theta_dot,
            'e_parallel': e_parallel,
            'e_perp': e_perp,
            'V_e_parallel': V_e_parallel,
            'V_e_perp': V_e_perp,
            'tau_ext': tau_ext,
            'F_ext_parallel': F_ext_parallel,
            'F_ext_perp': F_ext_perp,
            'control_wrench': F_cmd,
            'screw_axis': self.screw_axis
        }

        return F_cmd, info

    def _saturate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """
        Saturate wrench to maximum limits.

        Following Modern Robotics convention: [τ, fx, fy]

        Args:
            wrench: Input wrench [τ, fx, fy] (MR convention!)

        Returns:
            Saturated wrench [τ, fx, fy]
        """
        wrench_out = wrench.copy()

        # Saturate torque (first element in MR convention!)
        wrench_out[0] = np.clip(wrench[0], -self.max_torque, self.max_torque)

        # Saturate force magnitude (elements 1,2 in MR convention!)
        force_mag = np.linalg.norm(wrench[1:3])
        if force_mag > self.max_force:
            wrench_out[1:3] = wrench[1:3] / force_mag * self.max_force

        return wrench_out

    def reset(self):
        """Reset controller state."""
        self.last_theta = None
        self.last_theta_dot = None
        self.last_e_perp = None
        self.last_V_e_perp = None
        self.last_tau_ext = None

    @staticmethod
    def create_from_standard_params(screw_axis: np.ndarray,
                                    M_parallel: float,
                                    D_parallel: float,
                                    K_parallel: float,
                                    M_perpendicular: float,
                                    D_perpendicular: float,
                                    K_perpendicular: float,
                                    robot_params: SE2RobotParams,
                                    model_matching: bool = True,
                                    max_force: float = 100.0,
                                    max_torque: float = 50.0) -> 'SE2ScrewDecomposedImpedanceController':
        """
        Factory method to create controller from individual parameters.

        Following Modern Robotics convention: screw axis [sω, sx, sy]

        Args:
            screw_axis: Screw axis [sω, sx, sy] (MR convention: angular first!)
            M_parallel, D_parallel, K_parallel: Parallel impedance params
            M_perpendicular, D_perpendicular, K_perpendicular: Perpendicular impedance params
            robot_params: Robot physical parameters
            model_matching: Use model matching mode
            max_force, max_torque: Saturation limits

        Returns:
            Configured controller instance
        """
        params = ScrewImpedanceParams(
            M_parallel=M_parallel,
            D_parallel=D_parallel,
            K_parallel=K_parallel,
            M_perpendicular=M_perpendicular,
            D_perpendicular=D_perpendicular,
            K_perpendicular=K_perpendicular
        )

        return SE2ScrewDecomposedImpedanceController(
            screw_axis=screw_axis,
            params=params,
            robot_dynamics=SE2Dynamics(robot_params),
            model_matching=model_matching,
            use_feedforward=True,
            max_force=max_force,
            max_torque=max_torque
        )


def verify_projection_properties(P_parallel: np.ndarray, P_perp: np.ndarray) -> bool:
    """
    Verify mathematical properties of projection operators.

    Properties:
    1. Completeness: P_∥ + P_⊥ = I
    2. Idempotency: P_∥² = P_∥, P_⊥² = P_⊥
    3. Orthogonality: P_∥ P_⊥ = 0
    4. Symmetry: P_∥^T = P_∥, P_⊥^T = P_⊥

    Args:
        P_parallel: Parallel projection operator
        P_perp: Perpendicular projection operator

    Returns:
        True if all properties hold
    """
    tol = 1e-10

    # 1. Completeness
    completeness = np.allclose(P_parallel + P_perp, np.eye(3), atol=tol)

    # 2. Idempotency
    idempotent_parallel = np.allclose(P_parallel @ P_parallel, P_parallel, atol=tol)
    idempotent_perp = np.allclose(P_perp @ P_perp, P_perp, atol=tol)

    # 3. Orthogonality
    orthogonal = np.allclose(P_parallel @ P_perp, np.zeros((3, 3)), atol=tol)

    # 4. Symmetry
    symmetric_parallel = np.allclose(P_parallel.T, P_parallel, atol=tol)
    symmetric_perp = np.allclose(P_perp.T, P_perp, atol=tol)

    return (completeness and idempotent_parallel and idempotent_perp and
            orthogonal and symmetric_parallel and symmetric_perp)


# ============================================================================
# Test and demonstration code
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("SE(2) Screw-Decomposed Impedance Controller - Test Suite")
    print("=" * 70)

    # Test 1: Projection operators
    print("\nTest 1: Projection Operators")
    print("-" * 70)

    # Pure translation screw (MR convention: [sω, sx, sy])
    S_translation = np.array([0.0, 1.0, 0.0])  # Translation along x (no rotation, unit vx)
    controller_trans = SE2ScrewDecomposedImpedanceController(
        screw_axis=S_translation,
        params=ScrewImpedanceParams(),
        robot_dynamics=SE2Dynamics(SE2RobotParams(mass=1.0, inertia=0.1))
    )

    print(f"Screw (translation x, MR convention [ω, vx, vy]): {S_translation}")
    print(f"P_parallel:\n{controller_trans.P_parallel}")
    print(f"P_perpendicular:\n{controller_trans.P_perpendicular}")

    verified = verify_projection_properties(controller_trans.P_parallel,
                                           controller_trans.P_perpendicular)
    print(f"Projection properties verified: {verified}")

    # Pure rotation screw (MR convention: [sω, sx, sy])
    S_rotation = np.array([1.0, 0.0, 0.0])  # Pure rotation (unit ω, no linear velocity)
    controller_rot = SE2ScrewDecomposedImpedanceController(
        screw_axis=S_rotation,
        params=ScrewImpedanceParams(),
        robot_dynamics=SE2Dynamics(SE2RobotParams(mass=1.0, inertia=0.1))
    )

    print(f"\nScrew (rotation, MR convention [ω, vx, vy]): {S_rotation}")
    print(f"P_parallel:\n{controller_rot.P_parallel}")
    print(f"P_perpendicular:\n{controller_rot.P_perpendicular}")

    verified_rot = verify_projection_properties(controller_rot.P_parallel,
                                               controller_rot.P_perpendicular)
    print(f"Projection properties verified: {verified_rot}")

    # Test 2: Vector decomposition
    print("\n\nTest 2: Vector Decomposition")
    print("-" * 70)

    test_vector = np.array([2.0, 3.0, 0.5])
    theta, v_par, v_perp = controller_trans.decompose_vector(test_vector)

    print(f"Test vector: {test_vector}")
    print(f"Screw axis: {S_translation}")
    print(f"θ (scalar): {theta:.4f}")
    print(f"v_parallel: {v_par}")
    print(f"v_perp: {v_perp}")
    print(f"Reconstruction: {v_par + v_perp}")
    print(f"Match: {np.allclose(test_vector, v_par + v_perp)}")
    print(f"Orthogonality: v_perp^T S = {np.dot(v_perp, S_translation):.10f}")

    # Test 3: Control computation
    print("\n\nTest 3: Control Computation")
    print("-" * 70)

    # Setup
    robot_params = SE2RobotParams(mass=1.0, inertia=0.1)
    screw_params = ScrewImpedanceParams(
        M_parallel=1.0,
        D_parallel=5.0,
        K_parallel=25.0,
        M_perpendicular=1.0,
        D_perpendicular=20.0,
        K_perpendicular=100.0
    )

    # MR convention: [sω, sx, sy] - compliant along x translation
    controller = SE2ScrewDecomposedImpedanceController(
        screw_axis=np.array([0.0, 1.0, 0.0]),  # Compliant along x (no rotation, unit vx)
        params=screw_params,
        robot_dynamics=SE2Dynamics(robot_params),
        model_matching=True
    )

    # Current and desired states
    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([0.1, 0.05, 0.1])
    # MR convention: twist is [ω, vx, vy]
    current_twist = np.array([0.0, 0.0, 0.0])  # [omega, vx, vy]
    desired_twist = np.array([0.0, 0.2, 0.0])  # [omega, vx, vy]

    F_cmd, info = controller.compute_control(
        current_pose, desired_pose,
        current_twist, desired_twist
    )

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Pose error: {info['pose_error']}")
    print(f"\nDecomposition:")
    print(f"  θ = {info['theta']:.4f}")
    print(f"  e_parallel = {info['e_parallel']}")
    print(f"  e_perp = {info['e_perp']}")
    print(f"\nControl wrench: {F_cmd}")

    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)
