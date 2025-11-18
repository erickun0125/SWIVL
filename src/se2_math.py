"""
SE(2) Mathematics: Lie Group and Lie Algebra Utilities

This module provides mathematical utilities for working with SE(2) - the Special
Euclidean group in 2D, which represents rigid body transformations (rotation + translation).

SE(2) Lie group: {(R, t) | R ∈ SO(2), t ∈ ℝ²}
se(2) Lie algebra: {(ω, v) | ω ∈ ℝ, v ∈ ℝ²}

Key operations:
- exp: se(2) → SE(2) (exponential map)
- log: SE(2) → se(2) (logarithmic map)
- Ad: Adjoint representation
- ad: adjoint representation
- Composition, inverse, interpolation
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class SE2Pose:
    """
    SE(2) pose representation.

    Attributes:
        x: X position
        y: Y position
        theta: Orientation angle (radians)
    """
    x: float
    y: float
    theta: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])

    def to_matrix(self) -> np.ndarray:
        """
        Convert to 3x3 homogeneous transformation matrix.

        Returns:
            3x3 matrix [[R, t], [0, 1]]
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0, 1]
        ])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'SE2Pose':
        """Create SE2Pose from array [x, y, theta]."""
        return SE2Pose(x=arr[0], y=arr[1], theta=arr[2])

    @staticmethod
    def from_matrix(T: np.ndarray) -> 'SE2Pose':
        """
        Create SE2Pose from 3x3 homogeneous transformation matrix.

        Args:
            T: 3x3 transformation matrix

        Returns:
            SE2Pose object
        """
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return SE2Pose(x=x, y=y, theta=theta)


@dataclass
class se2Velocity:
    """
    se(2) velocity/twist representation.

    Attributes:
        vx: Linear velocity in x
        vy: Linear velocity in y
        omega: Angular velocity
    """
    vx: float
    vy: float
    omega: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [vx, vy, omega]."""
        return np.array([self.vx, self.vy, self.omega])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'se2Velocity':
        """Create se2Velocity from array [vx, vy, omega]."""
        return se2Velocity(vx=arr[0], vy=arr[1], omega=arr[2])


def normalize_angle(theta: float) -> float:
    """
    Normalize angle to [-π, π].

    Args:
        theta: Angle in radians

    Returns:
        Normalized angle in [-π, π]
    """
    return np.arctan2(np.sin(theta), np.cos(theta))


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create 2D rotation matrix.

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def se2_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se(2) to SE(2).

    Maps a twist (linear velocity + angular velocity) to a transformation.

    Args:
        xi: se(2) element [vx, vy, omega]

    Returns:
        3x3 SE(2) transformation matrix
    """
    vx, vy, omega = xi

    if np.abs(omega) < 1e-6:
        # Small angle approximation
        return np.array([
            [1, -omega, vx],
            [omega, 1, vy],
            [0, 0, 1]
        ])

    # General case
    c, s = np.cos(omega), np.sin(omega)

    # V matrix (left Jacobian of SO(2))
    V = np.array([
        [s / omega, -(1 - c) / omega],
        [(1 - c) / omega, s / omega]
    ])

    t = V @ np.array([vx, vy])

    return np.array([
        [c, -s, t[0]],
        [s,  c, t[1]],
        [0,  0, 1]
    ])


def se2_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SE(2) to se(2).

    Maps a transformation to a twist.

    Args:
        T: 3x3 SE(2) transformation matrix

    Returns:
        se(2) element [vx, vy, omega]
    """
    # Extract rotation and translation
    R = T[:2, :2]
    t = T[:2, 2]

    # Extract angle
    omega = np.arctan2(T[1, 0], T[0, 0])

    if np.abs(omega) < 1e-6:
        # Small angle approximation
        return np.array([t[0], t[1], omega])

    # General case
    c, s = np.cos(omega), np.sin(omega)

    # Inverse of V matrix
    V_inv = (omega / (2 * (1 - c))) * np.array([
        [s, (1 - c)],
        [-(1 - c), s]
    ])

    v = V_inv @ t

    return np.array([v[0], v[1], omega])


def se2_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compose two SE(2) transformations.

    Args:
        T1: First transformation (3x3)
        T2: Second transformation (3x3)

    Returns:
        Composed transformation T1 @ T2
    """
    return T1 @ T2


def se2_inverse(T: np.ndarray) -> np.ndarray:
    """
    Compute inverse of SE(2) transformation.

    Args:
        T: 3x3 SE(2) transformation matrix

    Returns:
        Inverse transformation
    """
    R = T[:2, :2]
    t = T[:2, 2]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(3)
    T_inv[:2, :2] = R_inv
    T_inv[:2, 2] = t_inv

    return T_inv


def se2_adjoint(T: np.ndarray) -> np.ndarray:
    """
    Compute Adjoint representation of SE(2).

    Ad_T maps twists from one frame to another.

    Args:
        T: 3x3 SE(2) transformation matrix

    Returns:
        3x3 Adjoint matrix
    """
    R = T[:2, :2]
    t = T[:2, 2]

    # Skew-symmetric matrix for cross product in 2D
    t_hat = np.array([[0, -1], [1, 0]]) @ t

    Ad = np.zeros((3, 3))
    Ad[:2, :2] = R
    Ad[:2, 2] = t_hat
    Ad[2, 2] = 1

    return Ad


def se2_adjoint_algebra(xi: np.ndarray) -> np.ndarray:
    """
    Compute adjoint representation of se(2).

    ad_xi computes Lie bracket [xi, ·]

    Args:
        xi: se(2) element [vx, vy, omega]

    Returns:
        3x3 adjoint matrix
    """
    vx, vy, omega = xi

    return np.array([
        [0, -omega, vy],
        [omega, 0, -vx],
        [0, 0, 0]
    ])


def body_to_world_velocity(pose: np.ndarray, vel_body: np.ndarray) -> np.ndarray:
    """
    Transform velocity from body frame to world frame.

    Args:
        pose: Current pose [x, y, theta]
        vel_body: Body frame velocity [vx_b, vy_b, omega]

    Returns:
        World frame velocity [vx_w, vy_w, omega]
    """
    theta = pose[2]
    R = rotation_matrix(theta)

    vel_world = np.zeros(3)
    vel_world[:2] = R @ vel_body[:2]
    vel_world[2] = vel_body[2]

    return vel_world


def world_to_body_velocity(pose: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
    """
    Transform velocity from world frame to body frame.

    Args:
        pose: Current pose [x, y, theta]
        vel_world: World frame velocity [vx_w, vy_w, omega]

    Returns:
        Body frame velocity [vx_b, vy_b, omega]
    """
    theta = pose[2]
    R = rotation_matrix(theta)

    vel_body = np.zeros(3)
    vel_body[:2] = R.T @ vel_world[:2]
    vel_body[2] = vel_world[2]

    return vel_body


def world_to_body_acceleration(
    pose: np.ndarray,
    vel_world: np.ndarray,
    accel_world: np.ndarray
) -> np.ndarray:
    """
    Transform acceleration from world frame to body frame.

    For SE(2), the transformation accounts for the time derivative of the
    rotation matrix, resulting in centrifugal/Coriolis-like terms.

    Derivation:
    a_b = R^T * a_w - ω * (R^T * skew * v_w)
    where ω is angular velocity and skew is 2D rotation by π/2

    Args:
        pose: Current pose [x, y, theta]
        vel_world: World frame velocity [vx_w, vy_w, omega]
        accel_world: World frame acceleration [ax_w, ay_w, alpha]

    Returns:
        Body frame acceleration [ax_b, ay_b, alpha]
    """
    theta = pose[2]
    omega = vel_world[2]
    R = rotation_matrix(theta)

    # Linear acceleration transformation
    # a_body = R^T * a_world - ω * R^T * skew(v_world)
    # where skew(v) rotates v by π/2: [vx, vy] → [-vy, vx]

    # Transform acceleration
    accel_linear_body = R.T @ accel_world[:2]

    # Centrifugal/Coriolis correction due to rotating frame
    # skew([vx, vy]) = [-vy, vx]
    vel_skew = np.array([-vel_world[1], vel_world[0]])
    correction = omega * (R.T @ vel_skew)

    accel_body = np.zeros(3)
    accel_body[:2] = accel_linear_body - correction
    accel_body[2] = accel_world[2]  # Angular acceleration unchanged

    return accel_body


def se2_interpolate(pose_start: np.ndarray, pose_end: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two SE(2) poses using geodesic interpolation.

    This performs proper interpolation in the SE(2) manifold.

    Args:
        pose_start: Start pose [x, y, theta]
        pose_end: End pose [x, y, theta]
        alpha: Interpolation parameter ∈ [0, 1]

    Returns:
        Interpolated pose [x, y, theta]
    """
    if alpha <= 0:
        return pose_start.copy()
    if alpha >= 1:
        return pose_end.copy()

    # Convert to matrices
    T_start = SE2Pose.from_array(pose_start).to_matrix()
    T_end = SE2Pose.from_array(pose_end).to_matrix()

    # Compute relative transformation
    T_rel = se2_inverse(T_start) @ T_end

    # Take logarithm to get twist
    xi = se2_log(T_rel)

    # Scale twist by alpha
    xi_scaled = alpha * xi

    # Exponential map back to SE(2)
    T_alpha = se2_exp(xi_scaled)

    # Compose with start
    T_result = T_start @ T_alpha

    # Convert back to pose array
    result = SE2Pose.from_matrix(T_result)

    return result.to_array()


def se2_distance(pose1: np.ndarray, pose2: np.ndarray,
                 weight_position: float = 1.0,
                 weight_orientation: float = 1.0) -> float:
    """
    Compute weighted distance between two SE(2) poses.

    Args:
        pose1: First pose [x, y, theta]
        pose2: Second pose [x, y, theta]
        weight_position: Weight for position error
        weight_orientation: Weight for orientation error

    Returns:
        Weighted distance
    """
    pos_error = np.linalg.norm(pose1[:2] - pose2[:2])
    angle_error = np.abs(normalize_angle(pose1[2] - pose2[2]))

    return weight_position * pos_error + weight_orientation * angle_error


def compute_screw_axis(pose_start: np.ndarray, pose_end: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute screw axis for motion from start to end pose.

    In SE(2), a screw motion is characterized by:
    - Instantaneous center of rotation (ICR)
    - Angular displacement

    Args:
        pose_start: Start pose [x, y, theta]
        pose_end: End pose [x, y, theta]

    Returns:
        Tuple of (screw_axis [vx, vy, omega], magnitude)
    """
    T_start = SE2Pose.from_array(pose_start).to_matrix()
    T_end = SE2Pose.from_array(pose_end).to_matrix()

    T_rel = se2_inverse(T_start) @ T_end
    xi = se2_log(T_rel)

    magnitude = np.linalg.norm(xi)

    if magnitude < 1e-6:
        return xi, 0.0

    screw_axis = xi / magnitude

    return screw_axis, magnitude


def compute_wrench_transform(pose: np.ndarray) -> np.ndarray:
    """
    Compute transformation matrix for wrenches from body to world frame.

    For SE(2), wrench transformation is the same as Adjoint transpose.

    Args:
        pose: Current pose [x, y, theta]

    Returns:
        3x3 wrench transformation matrix
    """
    T = SE2Pose.from_array(pose).to_matrix()
    Ad = se2_adjoint(T)
    return Ad.T


def integrate_velocity(pose: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate velocity to update pose (proper SE(2) integration).

    Args:
        pose: Current pose [x, y, theta]
        velocity: Velocity in body frame [vx, vy, omega]
        dt: Time step

    Returns:
        Updated pose [x, y, theta]
    """
    # Scale velocity by time step
    xi = velocity * dt

    # Exponential map
    T_curr = SE2Pose.from_array(pose).to_matrix()
    T_delta = se2_exp(xi)
    T_new = T_curr @ T_delta

    # Convert back
    result = SE2Pose.from_matrix(T_new)
    return result.to_array()


# Utility functions for common operations

def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform a point from body frame to world frame.

    Args:
        pose: Pose [x, y, theta]
        point: Point in body frame [x, y]

    Returns:
        Point in world frame [x, y]
    """
    T = SE2Pose.from_array(pose).to_matrix()
    p_homo = np.array([point[0], point[1], 1])
    p_world = T @ p_homo
    return p_world[:2]


def inverse_transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform a point from world frame to body frame.

    Args:
        pose: Pose [x, y, theta]
        point: Point in world frame [x, y]

    Returns:
        Point in body frame [x, y]
    """
    T = SE2Pose.from_array(pose).to_matrix()
    T_inv = se2_inverse(T)
    p_homo = np.array([point[0], point[1], 1])
    p_body = T_inv @ p_homo
    return p_body[:2]
