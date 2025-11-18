"""
Task Space Impedance Controller for SE(2)

Implements proper impedance control with robot dynamics:
- Task space inertia (Lambda_b)
- Coriolis compensation (mu)
- Gravity compensation (eta_b)
- Proper SE(2) pose error via logarithm map

This is a wrapper around SE2ImpedanceController for backward compatibility.

Following Modern Robotics (Lynch & Park) convention:
- Twist: V = [ω, vx, vy]^T (angular velocity first!)
- Wrench: F = [τ, fx, fy]^T (torque first!)

Reference: Modern Robotics, Lynch & Park, Chapter 11
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.se2_math import normalize_angle, world_to_body_velocity
from src.se2_dynamics import SE2Dynamics, SE2RobotParams
from src.ll_controllers.se2_impedance_controller import SE2ImpedanceController


@dataclass
class ImpedanceGains:
    """
    Impedance controller gains (backward compatible interface).

    These gains are mapped to proper impedance parameters:
    - kp_linear, kp_angular → K_d (stiffness matrix)
    - kd_linear, kd_angular → D_d (damping matrix)
    - Robot mass/inertia → M_d (inertia matrix, for model matching)
    """
    kp_linear: float = 50.0       # Stiffness for linear motion [N/m]
    kd_linear: float = 10.0       # Damping for linear motion [N⋅s/m]
    kp_angular: float = 20.0      # Stiffness for angular motion [N⋅m/rad]
    kd_angular: float = 5.0       # Damping for angular motion [N⋅m⋅s/rad]
    force_threshold: float = 20.0  # Force threshold for compliance [N]
    compliance_factor: float = 0.5 # Compliance reduction factor

    # Robot physical parameters (for proper dynamics)
    robot_mass: float = 1.0        # Robot mass [kg]
    robot_inertia: float = 0.1     # Robot inertia [kg⋅m²]


class TaskSpaceImpedanceController:
    """
    Task space impedance controller for SE(2) with proper robot dynamics.

    This controller now implements the full impedance control law:
        F_cmd = Lambda_b * dV_d + C_b * V + eta_b + D_d * V_e + K_d * e

    Where:
    - Lambda_b: Task space inertia matrix
    - C_b: Coriolis matrix
    - eta_b: Gravity wrench (zero for planar)
    - D_d, K_d: Desired damping and stiffness
    - e: Pose error via log map
    - V_e: Velocity error in body frame
    """

    def __init__(
        self,
        gains: Optional[ImpedanceGains] = None,
        max_force: float = 100.0,
        max_torque: float = 50.0,
        use_dynamics: bool = True
    ):
        """
        Initialize impedance controller.

        Args:
            gains: Impedance gains
            max_force: Maximum force magnitude [N]
            max_torque: Maximum torque magnitude [N⋅m]
            use_dynamics: If True, use full dynamics (recommended)
                         If False, fall back to simplified PD control
        """
        self.gains = gains if gains is not None else ImpedanceGains()
        self.max_force = max_force
        self.max_torque = max_torque
        self.use_dynamics = use_dynamics

        # Create robot dynamics
        robot_params = SE2RobotParams(
            mass=self.gains.robot_mass,
            inertia=self.gains.robot_inertia
        )

        if self.use_dynamics:
            # Create proper impedance controller with dynamics
            # Convert gains to impedance matrices (MR convention: angular first!)
            K_d = np.diag([
                self.gains.kp_angular,  # K_theta (MR: angular first!)
                self.gains.kp_linear,   # K_x
                self.gains.kp_linear    # K_y
            ])

            D_d = np.diag([
                self.gains.kd_angular,  # D_theta (MR: angular first!)
                self.gains.kd_linear,   # D_x
                self.gains.kd_linear    # D_y
            ])

            # Model matching: M_d = Lambda_b (MR convention: inertia first!)
            M_d = np.diag([
                self.gains.robot_inertia,  # I_d (MR: angular first!)
                self.gains.robot_mass,     # m_d_x
                self.gains.robot_mass      # m_d_y
            ])

            self.controller = SE2ImpedanceController(
                M_d=M_d,
                D_d=D_d,
                K_d=K_d,
                robot_dynamics=SE2Dynamics(robot_params),
                model_matching=True,
                use_feedforward=True,
                max_force=max_force,
                max_torque=max_torque
            )
        else:
            self.controller = None
            # Fallback state for simplified control
            self.prev_error_pos = None
            self.prev_error_angle = None
            self.dt = 0.01

    def set_timestep(self, dt: float):
        """Set controller timestep."""
        self.dt = dt

    def set_gains(self, gains: ImpedanceGains):
        """
        Update controller gains dynamically.

        Args:
            gains: New impedance gains
        """
        self.gains = gains

        if self.use_dynamics and self.controller is not None:
            # Update impedance parameters in SE2ImpedanceController
            # MR convention: angular first!
            K_d = np.diag([
                self.gains.kp_angular,  # K_theta (MR: angular first!)
                self.gains.kp_linear,   # K_x
                self.gains.kp_linear    # K_y
            ])

            D_d = np.diag([
                self.gains.kd_angular,  # D_theta (MR: angular first!)
                self.gains.kd_linear,   # D_x
                self.gains.kd_linear    # D_y
            ])

            M_d = np.diag([
                self.gains.robot_inertia,  # I_d (MR: angular first!)
                self.gains.robot_mass,     # m_d_x
                self.gains.robot_mass      # m_d_y
            ])

            self.controller.set_impedance_parameters(M_d=M_d, D_d=D_d, K_d=K_d)

    def compute_wrench(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        measured_wrench: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        desired_velocity: Optional[np.ndarray] = None,
        desired_acceleration: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute impedance control wrench with proper dynamics.

        Following Modern Robotics convention:
        - Twist: [ω, vx, vy]
        - Wrench: [τ, fx, fy]

        Impedance control law (model matching):
            F = Lambda_b * dV_d + C_b * V + eta_b + D_d * V_e + K_d * e

        IMPORTANT - Velocity Frame Convention:
        - current_velocity: Spatial frame [vx_s, vy_s, omega] (pose time derivative)
                           This is converted to body frame internally
        - desired_velocity: Body frame [omega, vx_b, vy_b] (MR convention!)
        - desired_acceleration: Body frame [domega, dvx_b, dvy_b] (MR convention!)

        Args:
            current_pose: Current pose [x, y, theta] in spatial frame (T_si)
            desired_pose: Desired pose [x, y, theta] in spatial frame (T_si^des)
            measured_wrench: Measured external wrench [tau, fx, fy] in body frame (MR convention!)
            current_velocity: Current velocity [vx_s, vy_s, omega] in spatial frame (will be converted internally!)
            desired_velocity: Desired body twist [omega, vx_b, vy_b] in body frame (MR convention!)
            desired_acceleration: Desired body acceleration [domega, dvx_b, dvy_b] in body frame (MR convention!)

        Returns:
            Control wrench [tau, fx, fy] in body frame (MR convention!)
        """
        if self.use_dynamics and self.controller is not None:
            # ========================================
            # Proper Impedance Control with Dynamics
            # ========================================

            # Convert velocities to body frame if provided
            if current_velocity is not None:
                body_twist_current = world_to_body_velocity(current_pose, current_velocity)
            else:
                body_twist_current = np.zeros(3)

            if desired_velocity is not None:
                # desired_velocity is already in body frame
                body_twist_desired = desired_velocity
            else:
                body_twist_desired = np.zeros(3)

            if desired_acceleration is None:
                desired_acceleration = np.zeros(3)

            # Compute control wrench using proper impedance controller
            F_cmd, info = self.controller.compute_control(
                current_pose=current_pose,
                desired_pose=desired_pose,
                body_twist_current=body_twist_current,
                body_twist_desired=body_twist_desired,
                body_accel_desired=desired_acceleration,
                F_ext=measured_wrench
            )

            # Apply additional compliance modulation based on force threshold
            # (for backward compatibility with original behavior)
            # MR convention: force is at indices 1,2 (not 0,1)
            measured_force_mag = np.linalg.norm(measured_wrench[1:3])
            if measured_force_mag > self.gains.force_threshold:
                compliance_scale = self.gains.compliance_factor
                # Scale down the impedance feedback term
                # This is in addition to the F_ext term in the control law
                F_cmd *= compliance_scale

            return F_cmd

        else:
            # ========================================
            # Fallback: Simplified PD Control (Legacy)
            # ========================================
            return self._compute_wrench_simplified(
                current_pose, desired_pose, measured_wrench,
                current_velocity, desired_velocity
            )

    def _compute_wrench_simplified(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        measured_wrench: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        desired_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simplified PD control (fallback, without dynamics).

        This is the original implementation for comparison/debugging.
        """
        # 1. Compute pose error in body frame
        x, y, theta = current_pose
        x_d, y_d, theta_d = desired_pose

        # Position error in spatial/world frame
        error_pos_world = np.array([x_d - x, y_d - y])

        # Transform position error to body frame
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        error_pos_body = np.array([
            cos_theta * error_pos_world[0] + sin_theta * error_pos_world[1],
            -sin_theta * error_pos_world[0] + cos_theta * error_pos_world[1]
        ])

        # Orientation error (always scalar, same in all frames)
        error_angle = normalize_angle(theta_d - theta)

        # 2. Compute velocity/twist error in body frame
        if current_velocity is not None and desired_velocity is not None:
            # Convert current spatial velocity to body frame
            current_twist_body = world_to_body_velocity(current_pose, current_velocity)

            # desired_velocity is already in body frame (body twist)
            desired_twist_body = desired_velocity

            # Twist error in body frame (MR convention: [ω, vx, vy])
            error_twist_body = desired_twist_body - current_twist_body
            error_vel_linear = error_twist_body[1:3]  # MR: linear components at indices 1,2
            error_vel_angular = error_twist_body[0]   # MR: angular component at index 0

        elif current_velocity is not None:
            # Only current velocity available - assume desired velocity is zero
            current_twist_body = world_to_body_velocity(current_pose, current_velocity)
            error_vel_linear = -current_twist_body[1:3]  # MR: linear components at indices 1,2
            error_vel_angular = -current_twist_body[0]   # MR: angular component at index 0

        else:
            # No velocity information - use finite difference
            if self.prev_error_pos is not None:
                error_vel_linear = (error_pos_body - self.prev_error_pos) / self.dt
            else:
                error_vel_linear = np.zeros(2)

            if self.prev_error_angle is not None:
                error_vel_angular = (error_angle - self.prev_error_angle) / self.dt
            else:
                error_vel_angular = 0.0

            self.prev_error_pos = error_pos_body.copy()
            self.prev_error_angle = error_angle

        # 3. Simplified PD control law
        force_body = (
            self.gains.kp_linear * error_pos_body +
            self.gains.kd_linear * error_vel_linear
        )
        torque = (
            self.gains.kp_angular * error_angle +
            self.gains.kd_angular * error_vel_angular
        )

        # Apply compliance based on measured force
        # MR convention: force is at indices 1,2 (not 0,1)
        measured_force_mag = np.linalg.norm(measured_wrench[1:3])

        if measured_force_mag > self.gains.force_threshold:
            # Reduce stiffness for compliance
            compliance_scale = self.gains.compliance_factor
            force_body *= compliance_scale

            # Add force feedback term
            force_body -= measured_wrench[1:3] * (1.0 - compliance_scale)

        # Saturate
        force_mag = np.linalg.norm(force_body)
        if force_mag > self.max_force:
            force_body = force_body / force_mag * self.max_force

        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # MR convention: return [tau, fx, fy]
        return np.array([torque, force_body[0], force_body[1]])

    def reset(self):
        """Reset controller state."""
        if self.use_dynamics and self.controller is not None:
            self.controller.reset()
        else:
            self.prev_error_pos = None
            self.prev_error_angle = None


# Backward compatibility: keep old class name as alias
TaskSpaceImpedanceController_Legacy = TaskSpaceImpedanceController
