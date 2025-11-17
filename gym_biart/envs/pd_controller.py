"""
PD Controller for SE(2) Pose Control

Implements a naive Proportional-Derivative controller that:
- Takes desired pose (x, y, theta) as input
- Computes control wrench (fx, fy, tau) in body frame
- Uses PD gains to generate smooth tracking behavior

The controller computes errors in the body frame and applies
proportional and derivative terms to generate wrench commands.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PDGains:
    """PD controller gains."""
    kp_linear: float = 50.0      # Proportional gain for linear motion
    kd_linear: float = 10.0      # Derivative gain for linear motion
    kp_angular: float = 20.0     # Proportional gain for angular motion
    kd_angular: float = 5.0      # Derivative gain for angular motion


class PoseController:
    """
    PD controller for SE(2) pose tracking.

    Computes wrench commands (force + moment) to track a desired pose.
    All computations are done in the body frame of the controlled object.
    """

    def __init__(
        self,
        gains: Optional[PDGains] = None,
        max_force: float = 100.0,
        max_torque: float = 50.0
    ):
        """
        Initialize PD controller.

        Args:
            gains: PD gains (uses defaults if None)
            max_force: Maximum force magnitude
            max_torque: Maximum torque magnitude
        """
        self.gains = gains if gains is not None else PDGains()
        self.max_force = max_force
        self.max_torque = max_torque

        # State tracking
        self.prev_error_pos = None
        self.prev_error_angle = None
        self.dt = 0.01  # Default timestep

    def set_timestep(self, dt: float):
        """Set controller timestep for derivative computation."""
        self.dt = dt

    def compute_wrench(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        current_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute control wrench to track desired pose.

        Args:
            current_pose: Current pose [x, y, theta] in world frame
            desired_pose: Desired pose [x, y, theta] in world frame
            current_velocity: Optional current velocity [vx, vy, omega] in world frame

        Returns:
            Wrench [fx, fy, tau] in body frame
        """
        # Extract poses
        x, y, theta = current_pose
        x_d, y_d, theta_d = desired_pose

        # Compute position error in world frame
        error_pos_world = np.array([x_d - x, y_d - y])

        # Transform position error to body frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        error_pos_body = np.array([
            cos_theta * error_pos_world[0] + sin_theta * error_pos_world[1],
            -sin_theta * error_pos_world[0] + cos_theta * error_pos_world[1]
        ])

        # Compute angular error (shortest rotation)
        error_angle = self._normalize_angle(theta_d - theta)

        # Compute derivative terms
        if current_velocity is not None:
            # Use provided velocity for derivative term
            vx, vy, omega = current_velocity

            # Transform velocity to body frame
            vel_body = np.array([
                cos_theta * vx + sin_theta * vy,
                -sin_theta * vx + cos_theta * vy
            ])

            # Derivative of position error (negative of velocity)
            derror_pos_body = -vel_body
            derror_angle = -omega

        else:
            # Estimate derivative from finite differences
            if self.prev_error_pos is not None:
                derror_pos_body = (error_pos_body - self.prev_error_pos) / self.dt
            else:
                derror_pos_body = np.zeros(2)

            if self.prev_error_angle is not None:
                derror_angle = (error_angle - self.prev_error_angle) / self.dt
            else:
                derror_angle = 0.0

            # Store current errors for next iteration
            self.prev_error_pos = error_pos_body.copy()
            self.prev_error_angle = error_angle

        # PD control law
        # Force in body frame
        force_body = (
            self.gains.kp_linear * error_pos_body +
            self.gains.kd_linear * derror_pos_body
        )

        # Torque
        torque = (
            self.gains.kp_angular * error_angle +
            self.gains.kd_angular * derror_angle
        )

        # Saturate to limits
        force_magnitude = np.linalg.norm(force_body)
        if force_magnitude > self.max_force:
            force_body = force_body / force_magnitude * self.max_force

        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # Construct wrench [fx, fy, tau]
        wrench = np.array([force_body[0], force_body[1], torque])

        return wrench

    def reset(self):
        """Reset controller state."""
        self.prev_error_pos = None
        self.prev_error_angle = None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


class MultiGripperController:
    """
    Manages multiple PD controllers for multi-gripper systems.

    Each gripper has its own PD controller for independent pose tracking.
    """

    def __init__(
        self,
        num_grippers: int,
        gains: Optional[PDGains] = None,
        max_force: float = 100.0,
        max_torque: float = 50.0
    ):
        """
        Initialize multi-gripper controller.

        Args:
            num_grippers: Number of grippers to control
            gains: PD gains (shared across all grippers)
            max_force: Maximum force per gripper
            max_torque: Maximum torque per gripper
        """
        self.num_grippers = num_grippers
        self.controllers = [
            PoseController(gains, max_force, max_torque)
            for _ in range(num_grippers)
        ]

    def set_timestep(self, dt: float):
        """Set timestep for all controllers."""
        for controller in self.controllers:
            controller.set_timestep(dt)

    def compute_wrenches(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray,
        current_velocities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute wrenches for all grippers.

        Args:
            current_poses: Array of shape (num_grippers, 3) with current poses
            desired_poses: Array of shape (num_grippers, 3) with desired poses
            current_velocities: Optional array of shape (num_grippers, 3) with velocities

        Returns:
            Array of shape (num_grippers, 3) with wrenches [fx, fy, tau]
        """
        if current_poses.shape[0] != self.num_grippers:
            raise ValueError(f"Expected {self.num_grippers} poses, got {current_poses.shape[0]}")

        wrenches = []
        for i in range(self.num_grippers):
            current_vel = current_velocities[i] if current_velocities is not None else None

            wrench = self.controllers[i].compute_wrench(
                current_poses[i],
                desired_poses[i],
                current_vel
            )
            wrenches.append(wrench)

        return np.array(wrenches)

    def reset(self):
        """Reset all controllers."""
        for controller in self.controllers:
            controller.reset()

    def get_controller(self, gripper_idx: int) -> PoseController:
        """Get controller for a specific gripper."""
        if 0 <= gripper_idx < self.num_grippers:
            return self.controllers[gripper_idx]
        else:
            raise ValueError(f"Gripper index {gripper_idx} out of range")

    def set_gains(self, gripper_idx: int, gains: PDGains):
        """Set gains for a specific gripper controller."""
        if 0 <= gripper_idx < self.num_grippers:
            self.controllers[gripper_idx].gains = gains
        else:
            raise ValueError(f"Gripper index {gripper_idx} out of range")


class ImpedanceController(PoseController):
    """
    Impedance controller extending PD controller with force feedback.

    Modulates stiffness based on measured external forces to achieve
    compliant behavior while tracking desired pose.
    """

    def __init__(
        self,
        gains: Optional[PDGains] = None,
        max_force: float = 100.0,
        max_torque: float = 50.0,
        force_threshold: float = 20.0,
        compliance_factor: float = 0.5
    ):
        """
        Initialize impedance controller.

        Args:
            gains: PD gains
            max_force: Maximum force
            max_torque: Maximum torque
            force_threshold: Force threshold for compliance activation
            compliance_factor: Factor to reduce stiffness when force exceeds threshold
        """
        super().__init__(gains, max_force, max_torque)
        self.force_threshold = force_threshold
        self.compliance_factor = compliance_factor

    def compute_wrench_with_compliance(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        measured_wrench: np.ndarray,
        current_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute wrench with impedance control.

        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]
            measured_wrench: Measured external wrench [fx, fy, tau]
            current_velocity: Optional current velocity

        Returns:
            Control wrench [fx, fy, tau]
        """
        # Compute nominal PD wrench
        wrench = self.compute_wrench(current_pose, desired_pose, current_velocity)

        # Check if measured force exceeds threshold
        measured_force = np.linalg.norm(measured_wrench[:2])

        if measured_force > self.force_threshold:
            # Reduce stiffness (compliance behavior)
            compliance_scale = self.compliance_factor
            wrench[:2] *= compliance_scale

            # Also reduce proportional gain effect
            # This allows the system to "give in" to external forces
            wrench[:2] -= measured_wrench[:2] * (1.0 - compliance_scale)

        return wrench


# Utility functions

def compute_desired_pose_from_velocity(
    current_pose: np.ndarray,
    velocity_command: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute desired pose from velocity command (for velocity control mode).

    Args:
        current_pose: Current pose [x, y, theta]
        velocity_command: Velocity command [vx, vy, omega] in body frame
        dt: Time step

    Returns:
        Desired pose [x, y, theta]
    """
    x, y, theta = current_pose
    vx_body, vy_body, omega = velocity_command

    # Transform body frame velocity to world frame
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    vx_world = cos_theta * vx_body - sin_theta * vy_body
    vy_world = sin_theta * vx_body + cos_theta * vy_body

    # Integrate to get desired pose
    x_desired = x + vx_world * dt
    y_desired = y + vy_world * dt
    theta_desired = theta + omega * dt

    # Normalize angle
    theta_desired = np.arctan2(np.sin(theta_desired), np.cos(theta_desired))

    return np.array([x_desired, y_desired, theta_desired])


def interpolate_pose(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Interpolate between two poses using linear interpolation.

    Args:
        pose_start: Start pose [x, y, theta]
        pose_end: End pose [x, y, theta]
        alpha: Interpolation parameter in [0, 1]

    Returns:
        Interpolated pose
    """
    # Linear interpolation for position
    pos = (1 - alpha) * pose_start[:2] + alpha * pose_end[:2]

    # Spherical linear interpolation for angle
    angle_diff = np.arctan2(
        np.sin(pose_end[2] - pose_start[2]),
        np.cos(pose_end[2] - pose_start[2])
    )
    angle = pose_start[2] + alpha * angle_diff

    return np.array([pos[0], pos[1], angle])
