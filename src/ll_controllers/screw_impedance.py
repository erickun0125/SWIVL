"""
Screw-Aware Impedance Controller for SE(2)

Implements impedance control along screw axes, providing:
- Decoupled control along and perpendicular to screw motion
- Better handling of constrained motions
- Improved force control for manipulation tasks
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.se2_math import normalize_angle, compute_screw_axis, se2_log, SE2Pose, se2_inverse


@dataclass
class ScrewImpedanceGains:
    """Screw impedance controller gains."""
    kp_along_screw: float = 50.0    # Stiffness along screw axis
    kd_along_screw: float = 10.0    # Damping along screw axis
    kp_perp_screw: float = 80.0     # Stiffness perpendicular to screw
    kd_perp_screw: float = 15.0     # Damping perpendicular to screw
    force_threshold: float = 20.0    # Force threshold for compliance
    compliance_factor: float = 0.6   # Compliance reduction factor


class ScrewImpedanceController:
    """
    Screw-aware impedance controller for SE(2).

    This controller decomposes motion into components along and perpendicular
    to the screw axis, allowing different impedance properties in each direction.
    """

    def __init__(
        self,
        gains: Optional[ScrewImpedanceGains] = None,
        max_force: float = 100.0,
        max_torque: float = 50.0
    ):
        """
        Initialize screw impedance controller.

        Args:
            gains: Screw impedance gains
            max_force: Maximum force magnitude
            max_torque: Maximum torque magnitude
        """
        self.gains = gains if gains is not None else ScrewImpedanceGains()
        self.max_force = max_force
        self.max_torque = max_torque

        self.prev_error_screw = None
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
        Compute screw-aware impedance control wrench.

        Args:
            current_pose: Current pose [x, y, theta] in spatial frame
            desired_pose: Desired pose [x, y, theta] in spatial frame
            measured_wrench: Measured external wrench [fx, fy, tau] in body frame
            current_velocity: Optional current velocity [vx, vy, omega] in spatial frame
            desired_velocity: Optional desired velocity [vx, vy, omega] in body frame

        Returns:
            Control wrench [fx, fy, tau] in body frame
        """
        # Compute screw axis from current to desired pose
        T_curr = SE2Pose.from_array(current_pose).to_matrix()
        T_des = SE2Pose.from_array(desired_pose).to_matrix()
        T_rel = se2_inverse(T_curr) @ T_des

        # Get error in se(2)
        error_screw = se2_log(T_rel)  # [vx, vy, omega] in body frame

        # Compute magnitude and direction
        error_mag = np.linalg.norm(error_screw)

        if error_mag < 1e-6:
            # Already at desired pose
            return np.zeros(3)

        # Normalized screw axis
        screw_axis = error_screw / error_mag

        # Decompose error into components
        # Along screw axis
        error_along = error_mag

        # Perpendicular components (for SE(2), we consider position perpendicular to motion)
        # This is a simplified version - proper implementation would project onto tangent space
        theta = current_pose[2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Compute velocity error in screw coordinates
        if current_velocity is not None:
            # Convert current velocity from spatial frame to body frame
            vx, vy, omega = current_velocity
            current_vel_body = np.array([
                cos_theta * vx + sin_theta * vy,
                -sin_theta * vx + cos_theta * vy,
                omega
            ])

            # Desired velocity is already in body frame
            if desired_velocity is not None:
                desired_vel_body = desired_velocity
            else:
                desired_vel_body = np.zeros(3)

            # Velocity error in body frame
            vel_error_body = desired_vel_body - current_vel_body

            # Project velocity error onto screw axis
            vel_error_along = np.dot(vel_error_body, screw_axis)
            vel_error_perp = vel_error_body - vel_error_along * screw_axis

            derror_along = vel_error_along
            derror_perp = vel_error_perp
        else:
            if self.prev_error_screw is not None:
                derror_screw = (error_screw - self.prev_error_screw) / self.dt
                derror_along = np.dot(derror_screw, screw_axis)
                derror_perp = derror_screw - derror_along * screw_axis
            else:
                derror_along = 0.0
                derror_perp = np.zeros(3)

            self.prev_error_screw = error_screw.copy()

        # Compute control along and perpendicular to screw
        control_along = (
            self.gains.kp_along_screw * error_along +
            self.gains.kd_along_screw * derror_along
        ) * screw_axis

        # For perpendicular, we use position error components not along screw
        error_perp = error_screw - error_along * screw_axis
        control_perp = (
            self.gains.kp_perp_screw * error_perp +
            self.gains.kd_perp_screw * derror_perp
        )

        # Total control wrench
        wrench = control_along + control_perp

        # Apply compliance based on measured force
        measured_force_mag = np.linalg.norm(measured_wrench[:2])

        if measured_force_mag > self.gains.force_threshold:
            # Reduce stiffness for compliance
            compliance_scale = self.gains.compliance_factor
            wrench[:2] *= compliance_scale

            # Add force feedback
            wrench[:2] -= measured_wrench[:2] * (1.0 - compliance_scale)

        # Saturate
        force_mag = np.linalg.norm(wrench[:2])
        if force_mag > self.max_force:
            wrench[:2] = wrench[:2] / force_mag * self.max_force

        wrench[2] = np.clip(wrench[2], -self.max_torque, self.max_torque)

        return wrench

    def reset(self):
        """Reset controller state."""
        self.prev_error_screw = None
