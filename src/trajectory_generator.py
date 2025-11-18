"""
Trajectory Generation with Cubic Splines

This module generates smooth trajectories in SE(2) using cubic splines.
Position (x, y) and orientation (theta) are interpolated separately using
cubic spline interpolation for smooth, differentiable trajectories.

Following Modern Robotics (Lynch & Park) convention:
- Body twist: V = [ω, vx, vy]^T (angular velocity first!)

Key features:
- Cubic spline interpolation for position and orientation
- Separate handling of position and orientation for better control
- Support for waypoint-based trajectory generation
- Computation of velocities and accelerations along trajectory
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.se2_math import normalize_angle, SE2Pose, world_to_body_velocity


@dataclass
class TrajectoryPoint:
    """
    Single point in a trajectory.

    Following Modern Robotics convention:
    - Body twist: [ω, vx, vy]

    Attributes:
        pose: SE(2) pose [x, y, theta] in spatial frame (T_si)
        velocity_spatial: Spatial velocity [vx, vy, omega] (time derivative of pose)
        velocity_body: Body twist [omega, vx_b, vy_b] (twist in body frame, MR convention!)
        acceleration: Acceleration [ax, ay, alpha] in spatial frame
        time: Time stamp
    """
    pose: np.ndarray
    velocity_spatial: np.ndarray
    velocity_body: np.ndarray
    acceleration: np.ndarray
    time: float


class CubicSplineTrajectory:
    """
    Trajectory generator using cubic splines.

    Generates smooth trajectories by fitting cubic splines to waypoints,
    treating position and orientation separately.
    """

    def __init__(
        self,
        waypoints: List[np.ndarray],
        times: Optional[List[float]] = None,
        boundary_conditions: str = 'natural'
    ):
        """
        Initialize trajectory generator with waypoints.

        Args:
            waypoints: List of SE(2) poses [x, y, theta]
            times: Optional list of time stamps for each waypoint
                   If None, uses uniform spacing
            boundary_conditions: Boundary conditions for spline
                               ('natural', 'clamped', 'not-a-knot')
        """
        self.waypoints = np.array(waypoints)
        self.num_waypoints = len(waypoints)

        if self.num_waypoints < 2:
            raise ValueError("Need at least 2 waypoints for trajectory")

        # Set up time parametrization
        if times is None:
            # Uniform spacing based on distance between waypoints
            distances = [0.0]
            for i in range(1, self.num_waypoints):
                pos_dist = np.linalg.norm(
                    self.waypoints[i, :2] - self.waypoints[i-1, :2]
                )
                angle_dist = np.abs(normalize_angle(
                    self.waypoints[i, 2] - self.waypoints[i-1, 2]
                ))
                # Weight position and orientation distances
                dist = pos_dist + 0.5 * angle_dist
                distances.append(distances[-1] + dist)

            # Normalize to [0, 1]
            total_dist = distances[-1]
            if total_dist > 0:
                self.times = np.array(distances) / total_dist
            else:
                self.times = np.linspace(0, 1, self.num_waypoints)
        else:
            self.times = np.array(times)

        # Ensure times are normalized to [0, 1]
        self.times = (self.times - self.times[0]) / (self.times[-1] - self.times[0])

        # Build cubic splines for position (x, y)
        self.spline_x = CubicSpline(
            self.times,
            self.waypoints[:, 0],
            bc_type=boundary_conditions
        )
        self.spline_y = CubicSpline(
            self.times,
            self.waypoints[:, 1],
            bc_type=boundary_conditions
        )

        # Handle orientation specially to account for angle wrapping
        # Unwrap angles to prevent discontinuities
        theta_unwrapped = np.unwrap(self.waypoints[:, 2])
        self.spline_theta = CubicSpline(
            self.times,
            theta_unwrapped,
            bc_type=boundary_conditions
        )

        self.duration = 1.0  # Duration in normalized time

    def set_duration(self, duration: float):
        """
        Set trajectory duration in seconds.

        Args:
            duration: Total trajectory duration (seconds)
        """
        self.duration = duration

    def evaluate(self, t: float) -> TrajectoryPoint:
        """
        Evaluate trajectory at a specific time.

        Args:
            t: Time in [0, duration]

        Returns:
            TrajectoryPoint with pose, spatial velocity, body twist, and acceleration
        """
        # Normalize time to [0, 1]
        t_norm = np.clip(t / self.duration, 0.0, 1.0)

        # Evaluate splines for pose
        x = self.spline_x(t_norm)
        y = self.spline_y(t_norm)
        theta = normalize_angle(self.spline_theta(t_norm))

        # Evaluate derivatives (spatial velocities)
        # Note: derivatives are w.r.t. normalized time, so scale by duration
        vx_spatial = self.spline_x(t_norm, 1) / self.duration
        vy_spatial = self.spline_y(t_norm, 1) / self.duration
        omega = self.spline_theta(t_norm, 1) / self.duration

        # Evaluate second derivatives (accelerations)
        ax = self.spline_x(t_norm, 2) / (self.duration ** 2)
        ay = self.spline_y(t_norm, 2) / (self.duration ** 2)
        alpha = self.spline_theta(t_norm, 2) / (self.duration ** 2)

        pose = np.array([x, y, theta])
        velocity_spatial = np.array([vx_spatial, vy_spatial, omega])
        acceleration = np.array([ax, ay, alpha])

        # Convert spatial velocity to body twist
        # Body twist: velocity expressed in the body frame
        velocity_body = world_to_body_velocity(pose, velocity_spatial)

        return TrajectoryPoint(pose, velocity_spatial, velocity_body, acceleration, t)

    def sample(self, num_samples: int) -> List[TrajectoryPoint]:
        """
        Sample trajectory at evenly spaced time points.

        Args:
            num_samples: Number of samples

        Returns:
            List of TrajectoryPoint objects
        """
        times = np.linspace(0, self.duration, num_samples)
        return [self.evaluate(t) for t in times]

    def get_length(self) -> float:
        """
        Compute approximate arc length of trajectory.

        Returns:
            Approximate trajectory length
        """
        # Sample trajectory and sum distances
        samples = self.sample(100)
        length = 0.0

        for i in range(1, len(samples)):
            pos_dist = np.linalg.norm(
                samples[i].pose[:2] - samples[i-1].pose[:2]
            )
            angle_dist = np.abs(normalize_angle(
                samples[i].pose[2] - samples[i-1].pose[2]
            ))
            length += pos_dist + 0.5 * angle_dist

        return length


class MinimumJerkTrajectory:
    """
    Minimum jerk trajectory generator.

    Generates smooth trajectories by minimizing jerk (third derivative),
    which produces natural-looking motion.
    """

    def __init__(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        duration: float
    ):
        """
        Initialize minimum jerk trajectory.

        Args:
            start_pose: Starting pose [x, y, theta]
            end_pose: Ending pose [x, y, theta]
            duration: Trajectory duration (seconds)
        """
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.duration = duration

    def evaluate(self, t: float) -> TrajectoryPoint:
        """
        Evaluate minimum jerk trajectory at time t.

        Args:
            t: Time in [0, duration]

        Returns:
            TrajectoryPoint with pose, spatial velocity, body twist, and acceleration
        """
        # Clamp time
        t = np.clip(t, 0.0, self.duration)

        # Normalized time
        tau = t / self.duration

        # Minimum jerk profile: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (self.duration ** 2)

        # Interpolate pose
        pose = (1 - s) * self.start_pose + s * self.end_pose

        # Handle angle wrapping
        angle_diff = normalize_angle(self.end_pose[2] - self.start_pose[2])
        pose[2] = self.start_pose[2] + s * angle_diff
        pose[2] = normalize_angle(pose[2])

        # Compute spatial velocity
        delta_pose = self.end_pose - self.start_pose
        delta_pose[2] = angle_diff
        velocity_spatial = s_dot * delta_pose

        # Compute acceleration
        acceleration = s_ddot * delta_pose

        # Convert spatial velocity to body twist
        velocity_body = world_to_body_velocity(pose, velocity_spatial)

        return TrajectoryPoint(pose, velocity_spatial, velocity_body, acceleration, t)

    def sample(self, num_samples: int) -> List[TrajectoryPoint]:
        """Sample trajectory at evenly spaced points."""
        times = np.linspace(0, self.duration, num_samples)
        return [self.evaluate(t) for t in times]


class TrajectoryManager:
    """
    Manages multiple trajectory segments and transitions.

    Useful for chaining multiple trajectories together with smooth transitions.
    """

    def __init__(self):
        """Initialize trajectory manager."""
        self.trajectories: List[CubicSplineTrajectory] = []
        self.segment_starts: List[float] = [0.0]
        self.total_duration = 0.0

    def add_trajectory(self, trajectory: CubicSplineTrajectory):
        """
        Add a trajectory segment.

        Args:
            trajectory: Trajectory to add
        """
        self.trajectories.append(trajectory)
        self.total_duration += trajectory.duration
        self.segment_starts.append(self.total_duration)

    def evaluate(self, t: float) -> Optional[TrajectoryPoint]:
        """
        Evaluate at global time.

        Args:
            t: Time in [0, total_duration]

        Returns:
            TrajectoryPoint or None if time is out of bounds
        """
        if t < 0 or t > self.total_duration:
            return None

        # Find which segment this time belongs to
        segment_idx = 0
        for i, start_time in enumerate(self.segment_starts[1:]):
            if t < start_time:
                segment_idx = i
                break
        else:
            segment_idx = len(self.trajectories) - 1

        # Evaluate in local time
        local_t = t - self.segment_starts[segment_idx]
        return self.trajectories[segment_idx].evaluate(local_t)

    def sample(self, num_samples: int) -> List[TrajectoryPoint]:
        """Sample entire trajectory."""
        times = np.linspace(0, self.total_duration, num_samples)
        return [self.evaluate(t) for t in times if self.evaluate(t) is not None]


def generate_circular_trajectory(
    center: np.ndarray,
    radius: float,
    num_points: int = 20,
    duration: float = 5.0
) -> CubicSplineTrajectory:
    """
    Generate a circular trajectory.

    Args:
        center: Center position [x, y]
        radius: Circle radius
        num_points: Number of waypoints
        duration: Total duration

    Returns:
        CubicSplineTrajectory
    """
    angles = np.linspace(0, 2 * np.pi, num_points)

    waypoints = []
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        # Tangent orientation
        theta = angle + np.pi / 2
        waypoints.append([x, y, theta])

    traj = CubicSplineTrajectory(waypoints)
    traj.set_duration(duration)

    return traj


def generate_lemniscate_trajectory(
    center: np.ndarray,
    scale: float = 50.0,
    num_points: int = 40,
    duration: float = 8.0
) -> CubicSplineTrajectory:
    """
    Generate a lemniscate (figure-eight) trajectory.

    Args:
        center: Center position [x, y]
        scale: Scale factor
        num_points: Number of waypoints
        duration: Total duration

    Returns:
        CubicSplineTrajectory
    """
    t = np.linspace(0, 2 * np.pi, num_points)

    waypoints = []
    for ti in t:
        # Lemniscate parametric equations
        denom = 1 + np.sin(ti) ** 2
        x = center[0] + scale * np.cos(ti) / denom
        y = center[1] + scale * np.sin(ti) * np.cos(ti) / denom

        # Tangent orientation
        dx = -scale * (np.sin(ti) + 2 * np.sin(ti) * np.cos(ti) ** 2) / (denom ** 2)
        dy = scale * (np.cos(2 * ti) + np.cos(ti) ** 2 - np.sin(ti) ** 2) / (denom ** 2)
        theta = np.arctan2(dy, dx)

        waypoints.append([x, y, theta])

    traj = CubicSplineTrajectory(waypoints)
    traj.set_duration(duration)

    return traj
