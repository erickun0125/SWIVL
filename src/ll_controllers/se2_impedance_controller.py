"""
SE(2) Impedance Controller

Implements proper impedance control on SE(2) for planar robots.
Supports both general impedance control and model-matching special case.

Target dynamics: M_d * dV_e + D_d * V_e + K_d * e = F_ext

where:
    e: pose error in R³ (log map of T_bd)
    V_e: velocity error in R³
    M_d: desired inertia matrix (3x3)
    D_d: desired damping matrix (3x3)
    K_d: desired stiffness matrix (3x3)
    F_ext: external wrench in body frame

Reference: Modern Robotics, Lynch & Park
          Impedance Control on SE(3), Various Papers
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


class SE2ImpedanceController:
    """
    SE(2) Impedance Controller with proper robot dynamics.

    Target dynamics: M_d * dV_e + D_d * V_e + K_d * e = F_ext

    Two modes:
    1. Model Matching (M_d = Lambda_b): Simplified control law
    2. General: Full dynamics compensation
    """

    def __init__(self,
                 M_d: np.ndarray,
                 D_d: np.ndarray,
                 K_d: np.ndarray,
                 robot_dynamics: SE2Dynamics,
                 model_matching: bool = True,
                 use_feedforward: bool = True,
                 max_force: float = 100.0,
                 max_torque: float = 50.0):
        """
        Initialize SE(2) Impedance Controller.

        Args:
            M_d: Desired inertia matrix (3x3), positive definite
            D_d: Desired damping matrix (3x3), positive definite
            K_d: Desired stiffness matrix (3x3), positive semi-definite
            robot_dynamics: SE2Dynamics object for robot-specific dynamics
            model_matching: If True, use M_d = Lambda_b (simplified control)
            use_feedforward: If True, include feedforward acceleration term
            max_force: Maximum force magnitude [N]
            max_torque: Maximum torque magnitude [N⋅m]
        """
        assert M_d.shape == (3, 3), "M_d must be 3x3"
        assert D_d.shape == (3, 3), "D_d must be 3x3"
        assert K_d.shape == (3, 3), "K_d must be 3x3"

        self.M_d = M_d
        self.D_d = D_d
        self.K_d = K_d
        self.robot_dynamics = robot_dynamics
        self.model_matching = model_matching
        self.use_feedforward = use_feedforward
        self.max_force = max_force
        self.max_torque = max_torque

        # Pre-compute inverse of M_d for general case
        if not model_matching:
            self.M_d_inv = np.linalg.inv(M_d)

        # Storage for debugging/monitoring
        self.last_error = None
        self.last_velocity_error = None
        self.last_control_wrench = None

    @staticmethod
    def create_diagonal_impedance(I_d: float,
                                   m_d: float,
                                   d_theta: float,
                                   d_x: float,
                                   d_y: float,
                                   k_theta: float,
                                   k_x: float,
                                   k_y: float,
                                   robot_params: SE2RobotParams,
                                   model_matching: bool = True,
                                   max_force: float = 100.0,
                                   max_torque: float = 50.0) -> 'SE2ImpedanceController':
        """
        Create impedance controller with diagonal impedance matrices.

        Args:
            I_d: Rotational inertia [kg⋅m²]
            m_d: Translational mass [kg]
            d_theta: Rotational damping [N⋅m⋅s]
            d_x, d_y: Translational damping [N⋅s/m]
            k_theta: Rotational stiffness [N⋅m/rad]
            k_x, k_y: Translational stiffness [N/m]
            robot_params: Robot physical parameters
            model_matching: Use model matching
            max_force: Maximum force
            max_torque: Maximum torque

        Returns:
            controller: SE2ImpedanceController instance
        """
        M_d = np.diag([m_d, m_d, I_d])
        D_d = np.diag([d_x, d_y, d_theta])
        K_d = np.diag([k_x, k_y, k_theta])

        robot_dynamics = SE2Dynamics(robot_params)

        return SE2ImpedanceController(
            M_d, D_d, K_d, robot_dynamics,
            model_matching=model_matching,
            max_force=max_force,
            max_torque=max_torque
        )

    def compute_pose_error(self, T_sb: np.ndarray, T_sd: np.ndarray) -> np.ndarray:
        """
        Compute pose error in body frame.

        e = log(T_bd)^∨ = log(T_sb^(-1) * T_sd)^∨

        Args:
            T_sb: Current pose (space to body) - 3x3 matrix
            T_sd: Desired pose (space to desired) - 3x3 matrix

        Returns:
            e: Pose error vector [vx, vy, omega] in R³ (se(2) coordinates)
        """
        T_bd = se2_inverse(T_sb) @ T_sd
        e = se2_log(T_bd)

        self.last_error = e
        return e

    def compute_pose_error_from_arrays(self,
                                        current_pose: np.ndarray,
                                        desired_pose: np.ndarray) -> np.ndarray:
        """
        Compute pose error from [x, y, theta] arrays.

        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]

        Returns:
            e: Pose error in R³
        """
        T_sb = SE2Pose.from_array(current_pose).to_matrix()
        T_sd = SE2Pose.from_array(desired_pose).to_matrix()
        return self.compute_pose_error(T_sb, T_sd)

    def compute_velocity_error(self,
                               body_twist_current: np.ndarray,
                               body_twist_desired: np.ndarray) -> np.ndarray:
        """
        Compute velocity error in body frame.

        V_e = b_V_d - b_V_b

        Args:
            body_twist_current: Current body twist b_V_b [vx, vy, omega]
            body_twist_desired: Desired body twist b_V_d [vx, vy, omega]

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
                               body_accel_desired: Optional[np.ndarray] = None,
                               F_ext: Optional[np.ndarray] = None,
                               current_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute control wrench for impedance control.

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
            body_accel_desired: Desired body acceleration db_V_d (3,) [optional]
            F_ext: External wrench (3,) [optional]
            current_pose: Current pose [x, y, theta] [optional, for debugging]

        Returns:
            F_cmd: Control wrench in body frame (3,) [fx, fy, tau]
        """
        # Default values for optional parameters
        if body_accel_desired is None:
            body_accel_desired = np.zeros(3)
        if F_ext is None:
            F_ext = np.zeros(3)

        # Get robot dynamics
        Lambda_b = self.robot_dynamics.get_task_space_inertia(current_pose)
        C_b = self.robot_dynamics.compute_coriolis_matrix(body_twist_current)
        eta_b = self.robot_dynamics.gravity_wrench(current_pose)

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
            # Note: F_ext term cancels out in passivity analysis!

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

        # Saturate control output
        F_cmd = self._saturate_wrench(F_cmd)

        self.last_control_wrench = F_cmd
        return F_cmd

    def _saturate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """
        Saturate wrench to maximum limits.

        Args:
            wrench: Input wrench [fx, fy, tau]

        Returns:
            Saturated wrench
        """
        wrench_out = wrench.copy()

        # Saturate force
        force_mag = np.linalg.norm(wrench[:2])
        if force_mag > self.max_force:
            wrench_out[:2] = wrench[:2] / force_mag * self.max_force

        # Saturate torque
        wrench_out[2] = np.clip(wrench[2], -self.max_torque, self.max_torque)

        return wrench_out

    def compute_control(self,
                       current_pose: np.ndarray,
                       desired_pose: np.ndarray,
                       body_twist_current: np.ndarray,
                       body_twist_desired: np.ndarray,
                       body_accel_desired: Optional[np.ndarray] = None,
                       F_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Complete impedance control computation.

        This is the main interface for the controller.

        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]
            body_twist_current: Current body twist [vx, vy, omega]
            body_twist_desired: Desired body twist [vx, vy, omega]
            body_accel_desired: Desired body acceleration [dvx, dvy, domega] [optional]
            F_ext: External wrench [fx, fy, tau] [optional]

        Returns:
            F_cmd: Control wrench (3,) [fx, fy, tau]
            info: Dictionary with debugging information
        """
        # Compute errors
        e = self.compute_pose_error_from_arrays(current_pose, desired_pose)
        V_e = self.compute_velocity_error(body_twist_current, body_twist_desired)

        # Compute control wrench
        F_cmd = self.compute_control_wrench(
            e, V_e, body_twist_current,
            body_accel_desired, F_ext, current_pose
        )

        # Collect debug info
        info = {
            'pose_error': e,
            'velocity_error': V_e,
            'pose_error_norm': np.linalg.norm(e),
            'velocity_error_norm': np.linalg.norm(V_e),
            'control_wrench': F_cmd,
            'position_error_norm': np.linalg.norm(e[:2]),
            'orientation_error': e[2]
        }

        return F_cmd, info

    def get_impedance_parameters(self) -> Dict[str, np.ndarray]:
        """Get current impedance parameters."""
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
        Update impedance parameters (for adaptive/variable impedance).

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

    def reset(self):
        """Reset controller state."""
        self.last_error = None
        self.last_velocity_error = None
        self.last_control_wrench = None


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("SE(2) Impedance Controller Test")
    print("="*60)

    # Robot parameters
    robot_params = SE2RobotParams(
        mass=1.0,     # 1 kg
        inertia=0.1   # 0.1 kg⋅m²
    )

    # Create controller with diagonal impedance
    controller = SE2ImpedanceController.create_diagonal_impedance(
        I_d=0.1,      # Rotational inertia [kg⋅m²] (match robot)
        m_d=1.0,      # Mass [kg] (match robot)
        d_theta=5.0,  # Rotational damping [N⋅m⋅s]
        d_x=10.0,     # X damping [N⋅s/m]
        d_y=10.0,     # Y damping [N⋅s/m]
        k_theta=20.0, # Rotational stiffness [N⋅m/rad]
        k_x=50.0,     # X stiffness [N/m]
        k_y=50.0,     # Y stiffness [N/m]
        robot_params=robot_params,
        model_matching=True
    )

    print("\n[Controller Parameters]")
    params = controller.get_impedance_parameters()
    print("M_d (Desired Inertia):")
    print(params['M_d'])
    print("\nD_d (Damping):")
    print(params['D_d'])
    print("\nK_d (Stiffness):")
    print(params['K_d'])

    # Test scenario
    print("\n[Test Scenario]")

    # Current pose
    current_pose = np.array([1.0, 0.5, np.pi/6])
    print(f"Current pose: {current_pose}")

    # Desired pose
    desired_pose = np.array([1.5, 1.0, np.pi/4])
    print(f"Desired pose: {desired_pose}")

    # Current and desired twists
    body_twist_current = np.array([0.1, 0.05, 0.1])
    body_twist_desired = np.array([0.0, 0.0, 0.0])
    body_accel_desired = np.array([0.0, 0.0, 0.0])

    # Compute control
    print("\n[Computing Control Wrench]")
    F_cmd, info = controller.compute_control(
        current_pose, desired_pose,
        body_twist_current,
        body_twist_desired,
        body_accel_desired
    )

    print("\nControl wrench F_cmd:")
    print(f"  Force X: {F_cmd[0]:.4f} N")
    print(f"  Force Y: {F_cmd[1]:.4f} N")
    print(f"  Torque:  {F_cmd[2]:.4f} N⋅m")

    print("\nError information:")
    print(f"  Pose error:     {info['pose_error']}")
    print(f"  Velocity error: {info['velocity_error']}")
    print(f"  ||e||:          {info['pose_error_norm']:.4f}")
    print(f"  ||V_e||:        {info['velocity_error_norm']:.4f}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
