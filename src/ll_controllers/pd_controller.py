"""
PD Controller for SE(2) Pose Control

Implements a Position & Orientation decomposed PD controller that:
- Takes desired pose (x, y, theta) as input
- Computes control wrench in body frame
- Uses PD gains to generate smooth tracking behavior

The controller computes errors in the body frame and applies
proportional and derivative terms to generate wrench commands.

Following Modern Robotics (Lynch & Park) convention:
- Output wrench: F = [τ, fx, fy]^T (torque first!)

Note: This controller works with spatial/world frame velocities as input
(not body twists), since it's a simple PD controller without dynamics.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.se2_math import normalize_angle, spatial_to_body_twist


@dataclass
class PDGains:
    """PD controller gains."""
    kp_linear: float = 50.0      # Proportional gain for linear motion
    kd_linear: float = 10.0      # Derivative gain for linear motion
    kp_angular: float = 20.0     # Proportional gain for angular motion
    kd_angular: float = 5.0      # Derivative gain for angular motion


class PDController:
    """
    Position & Orientation decomposed PD controller for SE(2) pose tracking.

    Computes wrench commands (force + moment) to track a desired pose.
    All computations are done in the body frame of the controlled object.

    This controller treats position and orientation separately, which is
    appropriate for many manipulation tasks.
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
        desired_velocity: np.ndarray,
        desired_acceleration: Optional[np.ndarray] = None,
        current_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute control wrench to track desired trajectory.

        Following Modern Robotics convention:
        - Output wrench: [τ, fx, fy]

        Note: This controller works with spatial/world frame velocities
        (not body twists) since it's a simple PD controller.

        Args:
            current_pose: Current pose [x, y, theta] in world frame
            desired_pose: Desired pose [x, y, theta] in world frame
            desired_velocity: Desired velocity [vx, vy, omega] in spatial/world frame
            desired_acceleration: Desired acceleration [ax, ay, alpha] in spatial/world frame (usually zero)
            current_velocity: Optional current velocity [vx, vy, omega] in spatial/world frame

        Returns:
            Wrench [tau, fx, fy] in body frame (MR convention!)
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
        error_angle = normalize_angle(theta_d - theta)

        # Transform desired velocity to body frame
        # desired_velocity is [vx_s, vy_s, omega], convert to MR [omega, vx_s, vy_s]
        desired_vel_spatial_mr = np.array([desired_velocity[2], desired_velocity[0], desired_velocity[1]])
        # Use proper adjoint map to get body twist [omega, vx_b, vy_b]
        desired_twist_body = spatial_to_body_twist(desired_pose, desired_vel_spatial_mr)
        
        # MR convention to PD internal convention (linear first for calculation convenience)
        # PD uses: [vx_b, vy_b, omega]
        desired_vel_body = np.array([desired_twist_body[1], desired_twist_body[2], desired_twist_body[0]])

        # Compute velocity error
        if current_velocity is not None:
            # Use provided velocity for error computation
            # current_velocity is [vx_s, vy_s, omega]
            
            # Convert to MR [omega, vx_s, vy_s]
            current_vel_spatial_mr = np.array([current_velocity[2], current_velocity[0], current_velocity[1]])
            # Convert to body twist [omega, vx_b, vy_b]
            current_twist_body = spatial_to_body_twist(current_pose, current_vel_spatial_mr)
            
            # Velocity error in body frame (linear only: [vx_b, vy_b])
            error_vel_body = desired_vel_body[:2] - np.array([current_twist_body[1], current_twist_body[2]])
            error_omega = desired_twist_body[0] - current_twist_body[0]

        else:
            # Estimate velocity error from finite differences
            if self.prev_error_pos is not None:
                derror_pos_body = (error_pos_body - self.prev_error_pos) / self.dt
                error_vel_body = -derror_pos_body
            else:
                # Assume current vel = 0, so error = desired (linear only: [vx_b, vy_b])
                error_vel_body = desired_vel_body[:2]

            if self.prev_error_angle is not None:
                derror_angle = (error_angle - self.prev_error_angle) / self.dt
                error_omega = -derror_angle
            else:
                error_omega = desired_twist_body[0]  # Assume current omega = 0

            # Store current errors for next iteration
            self.prev_error_pos = error_pos_body.copy()
            self.prev_error_angle = error_angle

        # PD control law with feedforward
        # Force in body frame
        force_body = (
            self.gains.kp_linear * error_pos_body +
            self.gains.kd_linear * error_vel_body
        )

        # Torque
        torque = (
            self.gains.kp_angular * error_angle +
            self.gains.kd_angular * error_omega
        )

        # Add feedforward term if desired acceleration provided
        # (Usually zero, but included for completeness)
        if desired_acceleration is not None:
            ax_d, ay_d, alpha_d = desired_acceleration
            # Transform to body frame
            accel_body = np.array([
                cos_theta * ax_d + sin_theta * ay_d,
                -sin_theta * ax_d + cos_theta * ay_d
            ])
            # Feedforward assumes unit mass/inertia
            force_body += accel_body
            torque += alpha_d

        # Saturate to limits
        force_magnitude = np.linalg.norm(force_body)
        if force_magnitude > self.max_force:
            force_body = force_body / force_magnitude * self.max_force

        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # Construct wrench [tau, fx, fy] (MR convention!)
        wrench = np.array([torque, force_body[0], force_body[1]])

        return wrench

    def reset(self):
        """Reset controller state."""
        self.prev_error_pos = None
        self.prev_error_angle = None


class MultiGripperPDController:
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
            PDController(gains, max_force, max_torque)
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
        desired_velocities: np.ndarray,
        desired_accelerations: Optional[np.ndarray] = None,
        current_velocities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute wrenches for all grippers.

        Following Modern Robotics convention:
        - Output wrenches: [τ, fx, fy]

        Args:
            current_poses: Array of shape (num_grippers, 3) with current poses
            desired_poses: Array of shape (num_grippers, 3) with desired poses
            desired_velocities: Array of shape (num_grippers, 3) with desired spatial velocities
            desired_accelerations: Optional array of shape (num_grippers, 3) with desired spatial accelerations
            current_velocities: Optional array of shape (num_grippers, 3) with current spatial velocities

        Returns:
            Array of shape (num_grippers, 3) with wrenches [tau, fx, fy] (MR convention!)
        """
        if current_poses.shape[0] != self.num_grippers:
            raise ValueError(f"Expected {self.num_grippers} poses, got {current_poses.shape[0]}")

        wrenches = []
        for i in range(self.num_grippers):
            current_vel = current_velocities[i] if current_velocities is not None else None
            desired_accel = desired_accelerations[i] if desired_accelerations is not None else None

            wrench = self.controllers[i].compute_wrench(
                current_poses[i],
                desired_poses[i],
                desired_velocities[i],
                desired_accel,
                current_vel
            )
            wrenches.append(wrench)

        return np.array(wrenches)

    def reset(self):
        """Reset all controllers."""
        for controller in self.controllers:
            controller.reset()

    def get_controller(self, gripper_idx: int) -> PDController:
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
