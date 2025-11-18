"""
Task Space Impedance Controller for SE(2)

Implements impedance control in task space with:
- Adjustable stiffness and damping
- Force feedback integration
- Compliant behavior under external forces

Reference: SE2ImpedanceController concept
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.se2_math import normalize_angle, world_to_body_velocity


@dataclass
class ImpedanceGains:
    """Impedance controller gains."""
    kp_linear: float = 50.0       # Stiffness for linear motion
    kd_linear: float = 10.0       # Damping for linear motion
    kp_angular: float = 20.0      # Stiffness for angular motion
    kd_angular: float = 5.0       # Damping for angular motion
    force_threshold: float = 20.0  # Force threshold for compliance
    compliance_factor: float = 0.5 # Compliance reduction factor


class TaskSpaceImpedanceController:
    """
    Task space impedance controller for SE(2).

    This controller modulates stiffness based on measured external forces
    to achieve compliant behavior while tracking desired pose.
    """

    def __init__(
        self,
        gains: Optional[ImpedanceGains] = None,
        max_force: float = 100.0,
        max_torque: float = 50.0
    ):
        """
        Initialize impedance controller.

        Args:
            gains: Impedance gains
            max_force: Maximum force magnitude
            max_torque: Maximum torque magnitude
        """
        self.gains = gains if gains is not None else ImpedanceGains()
        self.max_force = max_force
        self.max_torque = max_torque

        self.prev_error_pos = None
        self.prev_error_angle = None
        self.dt = 0.01

    def set_timestep(self, dt: float):
        """Set controller timestep."""
        self.dt = dt

    def compute_wrench(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        measured_wrench: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        desired_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute impedance control wrench.

        Impedance control law (in body frame):
            F = K * error_pose + D * error_twist

        Args:
            current_pose: Current pose [x, y, theta] in spatial frame (T_si)
            desired_pose: Desired pose [x, y, theta] in spatial frame (T_si^des)
            measured_wrench: Measured external wrench [fx, fy, tau] in body frame
            current_velocity: Current velocity [vx, vy, omega] in spatial frame
            desired_velocity: Desired velocity [vx, vy, omega] in body frame (body twist)

        Returns:
            Control wrench [fx, fy, tau] in body frame
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

            # Twist error in body frame
            error_twist_body = desired_twist_body - current_twist_body
            error_vel_linear = error_twist_body[:2]
            error_vel_angular = error_twist_body[2]

        elif current_velocity is not None:
            # Only current velocity available - assume desired velocity is zero
            current_twist_body = world_to_body_velocity(current_pose, current_velocity)
            error_vel_linear = -current_twist_body[:2]
            error_vel_angular = -current_twist_body[2]

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

        # 3. Impedance control law in body frame
        # F = K * error_pose + D * error_twist
        force_body = (
            self.gains.kp_linear * error_pos_body +
            self.gains.kd_linear * error_vel_linear
        )
        torque = (
            self.gains.kp_angular * error_angle +
            self.gains.kd_angular * error_vel_angular
        )

        # Apply compliance based on measured force
        measured_force_mag = np.linalg.norm(measured_wrench[:2])

        if measured_force_mag > self.gains.force_threshold:
            # Reduce stiffness for compliance
            compliance_scale = self.gains.compliance_factor
            force_body *= compliance_scale

            # Add force feedback term
            force_body -= measured_wrench[:2] * (1.0 - compliance_scale)

        # Saturate
        force_mag = np.linalg.norm(force_body)
        if force_mag > self.max_force:
            force_body = force_body / force_mag * self.max_force

        torque = np.clip(torque, -self.max_torque, self.max_torque)

        return np.array([force_body[0], force_body[1], torque])

    def reset(self):
        """Reset controller state."""
        self.prev_error_pos = None
        self.prev_error_angle = None
