"""
SE(2) Impedance Controller

Implements impedance control on SE(2) for planar robots (mobile robots, 2D manipulators).
Supports both general impedance control and model-matching special case.

Reference: Modern Robotics, Lynch & Park
          Impedance Control on SE(3), Various Papers

Author: Robotics Lab
Date: 2025-11-18
"""

import numpy as np
from typing import Tuple, Optional, Dict
from se2_math import SE2Math, SE2Kinematics


class SE2ImpedanceController:
    """
    SE(2) Impedance Controller
    
    Target dynamics: M_d * dV_e + D_d * V_e + K_d * e = F_ext
    
    where:
        e: pose error in R³ (log map of T_bd)
        V_e: velocity error in R³
        M_d: desired inertia matrix (3x3)
        D_d: desired damping matrix (3x3)
        K_d: desired stiffness matrix (3x3)
        F_ext: external wrench in body frame
    """
    
    def __init__(self,
                 M_d: np.ndarray,
                 D_d: np.ndarray,
                 K_d: np.ndarray,
                 model_matching: bool = True,
                 use_feedforward: bool = True):
        """
        Initialize SE(2) Impedance Controller
        
        Args:
            M_d: Desired inertia matrix (3x3), positive definite
            D_d: Desired damping matrix (3x3), positive definite
            K_d: Desired stiffness matrix (3x3), positive semi-definite
            model_matching: If True, use M_d = Lambda_b (simplified control)
            use_feedforward: If True, include feedforward acceleration term
        """
        assert M_d.shape == (3, 3), "M_d must be 3x3"
        assert D_d.shape == (3, 3), "D_d must be 3x3"
        assert K_d.shape == (3, 3), "K_d must be 3x3"
        
        self.M_d = M_d
        self.D_d = D_d
        self.K_d = K_d
        self.model_matching = model_matching
        self.use_feedforward = use_feedforward
        
        # Pre-compute inverse of M_d for general case
        if not model_matching:
            self.M_d_inv = np.linalg.inv(M_d)
        
        # Storage for debugging/monitoring
        self.last_error = None
        self.last_velocity_error = None
        self.last_control_wrench = None
    
    @staticmethod
    def create_diagonal_impedance(I_d: float, m_d: float,
                                   d_theta: float, d_x: float, d_y: float,
                                   k_theta: float, k_x: float, k_y: float,
                                   model_matching: bool = True) -> 'SE2ImpedanceController':
        """
        Create impedance controller with diagonal impedance matrices
        
        Args:
            I_d: Rotational inertia [kg⋅m²]
            m_d: Translational mass [kg]
            d_theta: Rotational damping [N⋅m⋅s]
            d_x, d_y: Translational damping [N⋅s/m]
            k_theta: Rotational stiffness [N⋅m/rad]
            k_x, k_y: Translational stiffness [N/m]
            model_matching: Use model matching
            
        Returns:
            controller: SE2ImpedanceController instance
        """
        M_d = np.diag([I_d, m_d, m_d])
        D_d = np.diag([d_theta, d_x, d_y])
        K_d = np.diag([k_theta, k_x, k_y])
        
        return SE2ImpedanceController(M_d, D_d, K_d, model_matching)
    
    def compute_pose_error(self, T_sb: np.ndarray, T_sd: np.ndarray) -> np.ndarray:
        """
        Compute pose error in body frame
        
        e = log(T_bd)^∨ = log(T_sb^(-1) * T_sd)^∨
        
        Args:
            T_sb: Current pose (space to body)
            T_sd: Desired pose (space to desired)
            
        Returns:
            e: Pose error vector [theta_e, e_x, e_y] in R³
        """
        T_bd = SE2Math.inverse(T_sb) @ T_sd
        e = SE2Math.log(T_bd)
        
        self.last_error = e
        return e
    
    def compute_velocity_error(self, body_twist_current: np.ndarray,
                               body_twist_desired: np.ndarray) -> np.ndarray:
        """
        Compute velocity error in body frame
        
        V_e = b_V_d - b_V_b
        
        Args:
            body_twist_current: Current body twist b_V_b
            body_twist_desired: Desired body twist b_V_d
            
        Returns:
            V_e: Velocity error in R³
        """
        V_e = body_twist_desired - body_twist_current
        
        self.last_velocity_error = V_e
        return V_e
    
    def compute_control_wrench(self,
                               e: np.ndarray,
                               V_e: np.ndarray,
                               body_twist_current: np.ndarray,
                               body_accel_desired: np.ndarray,
                               Lambda_b: np.ndarray,
                               C_b: Optional[np.ndarray] = None,
                               eta_b: Optional[np.ndarray] = None,
                               F_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute control wrench for impedance control
        
        Two cases:
        1. Model Matching (M_d = Lambda_b):
           F_cmd = Lambda_b * dV_d + C_b * V + eta_b + D_d * V_e + K_d * e
           
        2. General:
           F_cmd = Lambda_b * dV_cmd + C_b * V + eta_b - F_ext
           where dV_cmd = dV_d + M_d^(-1) * (D_d * V_e + K_d * e - F_ext)
        
        Args:
            e: Pose error (3,)
            V_e: Velocity error (3,)
            body_twist_current: Current body twist b_V_b (3,)
            body_accel_desired: Desired body acceleration db_V_d (3,)
            Lambda_b: Task-space inertia matrix (3x3) [ROBOT-SPECIFIC]
            C_b: Coriolis matrix (3x3) [ROBOT-SPECIFIC, optional]
            eta_b: Gravity wrench (3,) [ROBOT-SPECIFIC, optional]
            F_ext: External wrench (3,) [optional]
            
        Returns:
            F_cmd: Control wrench in body frame (3,)
        """
        # Default values for optional dynamics terms
        if C_b is None:
            C_b = np.zeros((3, 3))
        if eta_b is None:
            eta_b = np.zeros(3)
        if F_ext is None:
            F_ext = np.zeros(3)
        
        # Impedance feedback term
        impedance_term = self.D_d @ V_e + self.K_d @ e
        
        # Dynamics feedforward terms
        inertial_term = Lambda_b @ body_accel_desired if self.use_feedforward else np.zeros(3)
        coriolis_term = C_b @ body_twist_current
        gravity_term = eta_b
        
        if self.model_matching:
            # ========================================
            # Model Matching Case: M_d = Lambda_b
            # ========================================
            # F_cmd = Lambda_b * dV_d + C_b * V + eta_b + D_d * V_e + K_d * e
            # Note: F_ext term cancels out!
            
            F_cmd = (inertial_term +
                    coriolis_term +
                    gravity_term +
                    impedance_term)
        
        else:
            # ========================================
            # General Case
            # ========================================
            # dV_cmd = dV_d + M_d^(-1) * (D_d * V_e + K_d * e - F_ext)
            commanded_accel = (body_accel_desired + 
                             self.M_d_inv @ (impedance_term - F_ext))
            
            # F_cmd = Lambda_b * dV_cmd + C_b * V + eta_b - (I + Lambda_b * M_d^(-1)) * F_ext
            F_cmd = (Lambda_b @ commanded_accel +
                    coriolis_term +
                    gravity_term -
                    (np.eye(3) + Lambda_b @ self.M_d_inv) @ F_ext)
        
        self.last_control_wrench = F_cmd
        return F_cmd
    
    def compute_control(self,
                       T_sb: np.ndarray,
                       T_sd: np.ndarray,
                       body_twist_current: np.ndarray,
                       body_twist_desired: np.ndarray,
                       body_accel_desired: np.ndarray,
                       Lambda_b: np.ndarray,
                       C_b: Optional[np.ndarray] = None,
                       eta_b: Optional[np.ndarray] = None,
                       F_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Complete impedance control computation
        
        Args:
            T_sb: Current pose
            T_sd: Desired pose
            body_twist_current: Current body twist
            body_twist_desired: Desired body twist
            body_accel_desired: Desired body acceleration
            Lambda_b: Task-space inertia matrix
            C_b: Coriolis matrix (optional)
            eta_b: Gravity wrench (optional)
            F_ext: External wrench (optional)
            
        Returns:
            F_cmd: Control wrench (3,)
            info: Dictionary with debugging information
        """
        # Compute errors
        e = self.compute_pose_error(T_sb, T_sd)
        V_e = self.compute_velocity_error(body_twist_current, body_twist_desired)
        
        # Compute control wrench
        F_cmd = self.compute_control_wrench(
            e, V_e, body_twist_current, body_accel_desired,
            Lambda_b, C_b, eta_b, F_ext
        )
        
        # Collect debug info
        info = {
            'pose_error': e,
            'velocity_error': V_e,
            'pose_error_norm': np.linalg.norm(e),
            'velocity_error_norm': np.linalg.norm(V_e),
            'control_wrench': F_cmd,
            'theta_error': e[0],
            'position_error_norm': np.linalg.norm(e[1:3])
        }
        
        return F_cmd, info
    
    def get_impedance_parameters(self) -> Dict[str, np.ndarray]:
        """Get current impedance parameters"""
        return {
            'M_d': self.M_d.copy(),
            'D_d': self.D_d.copy(),
            'K_d': self.K_d.copy()
        }
    
    def set_impedance_parameters(self,
                                M_d: Optional[np.ndarray] = None,
                                D_d: Optional[np.ndarray] = None,
                                K_d: Optional[np.ndarray] = None):
        """
        Update impedance parameters (for adaptive/variable impedance)
        
        Args:
            M_d: New inertia matrix (optional)
            D_d: New damping matrix (optional)
            K_d: New stiffness matrix (optional)
        """
        if M_d is not None:
            self.M_d = M_d
            if not self.model_matching:
                self.M_d_inv = np.linalg.inv(M_d)
        
        if D_d is not None:
            self.D_d = D_d
        
        if K_d is not None:
            self.K_d = K_d


class RobotDynamics:
    """
    Base class for robot-specific dynamics computation
    
    Each specific robot (2-link planar arm, differential drive, etc.)
    should inherit from this and implement the methods.
    """
    
    def compute_task_inertia(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute task-space inertia matrix Lambda_b(q)
        
        For a robot with Jacobian J_b and joint-space inertia M(q):
        Lambda_b = (J_b * M(q)^(-1) * J_b^T)^(-1)
        
        Args:
            q: Joint positions
            q_dot: Joint velocities
            
        Returns:
            Lambda_b: 3x3 task-space inertia matrix
            
        TODO: Implement for specific robot
        """
        raise NotImplementedError("Implement for specific robot")
    
    def compute_coriolis(self, q: np.ndarray, q_dot: np.ndarray,
                        body_twist: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis term C_b(q, q_dot) * V
        
        Args:
            q: Joint positions
            q_dot: Joint velocities
            body_twist: Body twist V
            
        Returns:
            C_b_V: 3D Coriolis wrench
            
        TODO: Implement for specific robot
        """
        raise NotImplementedError("Implement for specific robot")
    
    def compute_gravity(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation wrench eta_b(q)
        
        For planar robots in horizontal plane: typically zero
        For vertical plane robots: non-zero
        
        Args:
            q: Joint positions
            
        Returns:
            eta_b: 3D gravity wrench
            
        TODO: Implement for specific robot
        """
        raise NotImplementedError("Implement for specific robot")
    
    def wrench_to_joint_torque(self, F_cmd: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Convert body wrench to joint torques
        
        tau = J_b^T * F_cmd
        
        Args:
            F_cmd: Control wrench in body frame
            q: Joint positions
            
        Returns:
            tau: Joint torques
            
        TODO: Implement for specific robot
        """
        raise NotImplementedError("Implement for specific robot")


# ============================================================================
# Example: Simplified 2-DOF Planar Robot (for demonstration)
# ============================================================================

class SimplePlanar2DOF(RobotDynamics):
    """
    Simplified 2-DOF planar robot for demonstration
    
    Assumptions:
    - Two revolute joints
    - Links of length L1, L2 with masses m1, m2
    - Horizontal plane (no gravity in task space)
    """
    
    def __init__(self, L1: float = 1.0, L2: float = 1.0,
                 m1: float = 1.0, m2: float = 1.0):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint angles"""
        theta1, theta2 = q[0], q[1]
        
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        theta = theta1 + theta2
        
        return SE2Math.transformation_matrix(x, y, theta)
    
    def body_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute body Jacobian J_b
        
        Returns:
            J_b: 3x2 body Jacobian
            
        TODO: Implement properly for your robot
        """
        # Placeholder - simplified version
        # In reality, this requires careful derivation
        return np.ones((3, 2))  # PLACEHOLDER
    
    def compute_task_inertia(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Simplified task-space inertia
        
        For demonstration: return diagonal matrix
        In practice: compute from M(q) and J_b
        """
        # TODO: Implement proper computation
        # Lambda_b = (J_b * M(q)^(-1) * J_b^T)^(-1)
        
        # Placeholder: diagonal approximation
        I_zz = 0.5  # Rotational inertia
        m_eff = 2.0  # Effective mass
        
        return np.diag([I_zz, m_eff, m_eff])
    
    def compute_coriolis(self, q: np.ndarray, q_dot: np.ndarray,
                        body_twist: np.ndarray) -> np.ndarray:
        """
        Simplified Coriolis term
        
        TODO: Implement from robot dynamics
        """
        # For slow motion, can approximate as zero
        return np.zeros(3)
    
    def compute_gravity(self, q: np.ndarray) -> np.ndarray:
        """Horizontal plane: no gravity in task space"""
        return np.zeros(3)
    
    def wrench_to_joint_torque(self, F_cmd: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Convert wrench to joint torques via J_b^T"""
        J_b = self.body_jacobian(q)
        return J_b.T @ F_cmd


# ============================================================================
# Test and Example Usage
# ============================================================================

def test_impedance_controller():
    """Test SE(2) impedance controller"""
    print("="*60)
    print("SE(2) Impedance Controller Test")
    print("="*60)
    
    # Create controller with diagonal impedance
    controller = SE2ImpedanceController.create_diagonal_impedance(
        I_d=0.5,      # Rotational inertia [kg⋅m²]
        m_d=2.0,      # Mass [kg]
        d_theta=5.0,  # Rotational damping [N⋅m⋅s]
        d_x=20.0,     # X damping [N⋅s/m]
        d_y=20.0,     # Y damping [N⋅s/m]
        k_theta=50.0, # Rotational stiffness [N⋅m/rad]
        k_x=200.0,    # X stiffness [N/m]
        k_y=200.0,    # Y stiffness [N/m]
        model_matching=True
    )
    
    print("\n[Controller Parameters]")
    params = controller.get_impedance_parameters()
    print("M_d (Inertia):")
    print(params['M_d'])
    print("\nD_d (Damping):")
    print(params['D_d'])
    print("\nK_d (Stiffness):")
    print(params['K_d'])
    
    # Test scenario
    print("\n[Test Scenario]")
    
    # Current pose
    T_sb = SE2Math.transformation_matrix(1.0, 0.5, np.pi/6)
    print("Current pose T_sb:")
    print(SE2Math.to_xyt(T_sb))
    
    # Desired pose
    T_sd = SE2Math.transformation_matrix(1.5, 1.0, np.pi/4)
    print("Desired pose T_sd:")
    print(SE2Math.to_xyt(T_sd))
    
    # Current and desired twists
    body_twist_current = np.array([0.1, 0.2, 0.1])
    body_twist_desired = np.array([0.0, 0.0, 0.0])
    body_accel_desired = np.array([0.0, 0.0, 0.0])
    
    # Robot dynamics (simplified)
    Lambda_b = np.diag([0.5, 2.0, 2.0])  # Match M_d for model matching
    C_b = np.zeros((3, 3))
    eta_b = np.zeros(3)
    
    # Compute control
    print("\n[Computing Control Wrench]")
    F_cmd, info = controller.compute_control(
        T_sb, T_sd,
        body_twist_current,
        body_twist_desired,
        body_accel_desired,
        Lambda_b, C_b, eta_b
    )
    
    print("\nControl wrench F_cmd:")
    print(f"  Torque: {F_cmd[0]:.4f} N⋅m")
    print(f"  Force X: {F_cmd[1]:.4f} N")
    print(f"  Force Y: {F_cmd[2]:.4f} N")
    
    print("\nError information:")
    print(f"  Pose error:     {info['pose_error']}")
    print(f"  Velocity error: {info['velocity_error']}")
    print(f"  ||e||:          {info['pose_error_norm']:.4f}")
    print(f"  ||V_e||:        {info['velocity_error_norm']:.4f}")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    test_impedance_controller()