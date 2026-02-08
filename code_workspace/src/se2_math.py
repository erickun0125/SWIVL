"""
SE(2) Mathematics: Lie Group and Lie Algebra Utilities

This module provides mathematical utilities for working with SE(2) - the Special
Euclidean group in 2D, which represents rigid body transformations (rotation + translation).

Following Modern Robotics (Lynch & Park) convention:
- SE(2) Lie group: {(R, t) | R ∈ SO(2), t ∈ ℝ²}
- se(2) Lie algebra: {(ω, v) | ω ∈ ℝ, v ∈ ℝ²}
- Twist representation: [ω, vx, vy]^T (angular velocity first!)

Key operations:
- exp: se(2) → SE(2) (exponential map)
- log: SE(2) → se(2) (logarithmic map)
- Ad: Adjoint representation (transforms twists between frames)
- ad: adjoint representation (Lie bracket)
- Composition, inverse, interpolation

Reference: Modern Robotics by Kevin M. Lynch and Frank C. Park, Chapter 3
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
class se2Twist:
    """
    se(2) twist/velocity representation.

    Following Modern Robotics convention:
    Twist = [ω, vx, vy]^T where:
    - ω: Angular velocity (scalar)
    - vx, vy: Linear velocity components

    Attributes:
        omega: Angular velocity
        vx: Linear velocity in x
        vy: Linear velocity in y
    """
    omega: float
    vx: float
    vy: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [omega, vx, vy]."""
        return np.array([self.omega, self.vx, self.vy])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'se2Twist':
        """Create se2Twist from array [omega, vx, vy]."""
        return se2Twist(omega=arr[0], vx=arr[1], vy=arr[2])


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

    Following Modern Robotics convention.

    Args:
        xi: se(2) element [omega, vx, vy]

    Returns:
        3x3 SE(2) transformation matrix

    Reference: Modern Robotics, Section 3.3.2
    """
    omega, vx, vy = xi

    if np.abs(omega) < 1e-6:
        # Small angle approximation: exp([0, v]) = [I, v; 0, 1]
        return np.array([
            [1, 0, vx],
            [0, 1, vy],
            [0, 0, 1]
        ])

    # General case
    c, s = np.cos(omega), np.sin(omega)

    # V(ω) matrix (left Jacobian of SO(2))
    # V(ω) = (1/ω) * [sin(ω), -(1-cos(ω)); (1-cos(ω)), sin(ω)]
    V = (1.0 / omega) * np.array([
        [s, -(1 - c)],
        [(1 - c), s]
    ])

    # Translation: t = V(ω) * [vx, vy]
    t = V @ np.array([vx, vy])

    return np.array([
        [c, -s, t[0]],
        [s,  c, t[1]],
        [0,  0, 1]
    ])


def se2_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SE(2) to se(2).

    Following Modern Robotics convention.

    Args:
        T: 3x3 SE(2) transformation matrix

    Returns:
        se(2) element [omega, vx, vy]

    Reference: Modern Robotics, Section 3.3.2
    """
    # Extract rotation and translation
    R = T[:2, :2]
    t = T[:2, 2]

    # Extract angle
    omega = np.arctan2(T[1, 0], T[0, 0])

    # Small angle threshold - be conservative near singularities
    small_angle_threshold = 1e-6
    near_pi_threshold = 1e-4  # Threshold for omega ≈ ±π

    if np.abs(omega) < small_angle_threshold:
        # Small angle approximation: log([I, t; 0, 1]) = [0, t]
        return np.array([omega, t[0], t[1]])

    # Check if near ±π singularity (cot(ω/2) diverges when ω → ±π)
    if np.abs(np.abs(omega) - np.pi) < near_pi_threshold:
        # Near ±π: use stable alternative formulation
        # When ω → ±π, half_omega → ±π/2, so tan(half_omega) → ±∞
        # Use L'Hôpital or series expansion for stability
        half_omega = omega / 2.0
        sign_omega = np.sign(omega)

        # Stable formula near singularity:
        # cot(ω/2) ≈ (π/2 - |ω|/2) / tan(π/2 - |ω|/2) near ±π
        # For small ε = π - |ω|, this gives stable computation
        epsilon = np.pi - np.abs(omega)
        # Use Taylor expansion: cot(π/2 - ε/2) ≈ ε/2 for small ε
        cot_half_stable = sign_omega * epsilon / 2.0

        V_inv = np.array([
            [half_omega * cot_half_stable, half_omega],
            [-half_omega, half_omega * cot_half_stable]
        ])
    else:
        # General case - stable range
        # Inverse of V matrix: V^{-1}(ω)
        # V^{-1}(ω) = (ω/2) * cot(ω/2) * I - (ω/2) * [0, -1; 1, 0]
        #           = [ω/2 * cot(ω/2), ω/2; -ω/2, ω/2 * cot(ω/2)]
        half_omega = omega / 2.0
        cot_half = 1.0 / np.tan(half_omega)

        V_inv = np.array([
            [half_omega * cot_half, half_omega],
            [-half_omega, half_omega * cot_half]
        ])

    # Linear velocity: [vx, vy] = V^{-1}(ω) * t
    v = V_inv @ t

    return np.array([omega, v[0], v[1]])


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

    Following Modern Robotics convention:
    For T = [R p; 0 1] where R ∈ SO(2), p = [px, py]^T:

    Ad_T = [ 1      0      0   ]
           [ py    R11    R12  ]
           [-px    R21    R22  ]

    This transforms twists [ω, vx, vy]^T from body frame to spatial frame:
    V_spatial = Ad_T * V_body

    Args:
        T: 3x3 SE(2) transformation matrix

    Returns:
        3x3 Adjoint matrix

    Reference: Modern Robotics, Section 3.3.2
    """
    R = T[:2, :2]
    p = T[:2, 2]

    Ad = np.zeros((3, 3))
    Ad[0, 0] = 1.0  # Angular velocity is frame-independent
    Ad[1, 0] = p[1]  # py
    Ad[2, 0] = -p[0]  # -px
    Ad[1:3, 1:3] = R  # Rotation for linear velocity

    return Ad


def se2_adjoint_algebra(xi: np.ndarray) -> np.ndarray:
    """
    Compute adjoint representation of se(2).

    Following Modern Robotics convention:
    For twist [ω, vx, vy]^T:

    ad_ξ = [  0    0    0  ]
           [-vy   0   -ω  ]
           [ vx   ω    0  ]

    This computes Lie bracket [xi, eta] = ad_xi * eta

    Args:
        xi: se(2) element [omega, vx, vy]

    Returns:
        3x3 adjoint matrix

    Reference: Modern Robotics, Section 3.3.3
    """
    omega, vx, vy = xi

    return np.array([
        [0, 0, 0],
        [-vy, 0, -omega],
        [vx, omega, 0]
    ])


def body_to_spatial_twist(pose: np.ndarray, body_twist: np.ndarray) -> np.ndarray:
    """
    Transform twist from body frame to spatial (world) frame.

    Following Modern Robotics convention:
    V_s = Ad_T * V_b

    For SE(2):
    ω_s = ω_b  (angular velocity is frame-independent)
    v_s = R * v_b + [py, -px]^T * ω_b  (includes p × ω term!)

    Args:
        pose: Current pose [x, y, theta]
        body_twist: Body frame twist [omega, vx_b, vy_b]

    Returns:
        Spatial frame twist [omega, vx_s, vy_s]

    Reference: Modern Robotics, Section 3.3.2
    """
    # Extract components
    x, y, theta = pose
    omega_b, vx_b, vy_b = body_twist

    # Rotation matrix
    R = rotation_matrix(theta)

    # Transform linear velocity with p × ω term
    v_b = np.array([vx_b, vy_b])
    v_s = R @ v_b + np.array([y, -x]) * omega_b  # p̂ × ω term

    # Angular velocity is frame-independent
    return np.array([omega_b, v_s[0], v_s[1]])


def integrate_velocity(pose: np.ndarray, body_twist: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate body-frame twist over a timestep to obtain the updated pose.

    Following Modern Robotics convention:
    - pose: [x, y, theta] (spatial frame)
    - body_twist: [omega, vx_b, vy_b]
    - dt: time step (seconds)
    """
    xi = body_twist * dt  # se(2) element for this step
    T_current = SE2Pose.from_array(pose).to_matrix()
    T_delta = se2_exp(xi)
    T_next = T_current @ T_delta
    return SE2Pose.from_matrix(T_next).to_array()


def spatial_to_body_twist(pose: np.ndarray, spatial_twist: np.ndarray) -> np.ndarray:
    """
    Transform twist from spatial (world) frame to body frame.

    Following Modern Robotics convention:
    V_b = Ad_{T^{-1}} * V_s

    For SE(2):
    ω_b = ω_s  (angular velocity is frame-independent)
    v_b = R^T * (v_s - [py, -px]^T * ω_s)

    Args:
        pose: Current pose [x, y, theta]
        spatial_twist: Spatial frame twist [omega, vx_s, vy_s]

    Returns:
        Body frame twist [omega, vx_b, vy_b]

    Reference: Modern Robotics, Section 3.3.2
    """
    # Extract components
    x, y, theta = pose
    omega_s, vx_s, vy_s = spatial_twist

    # Rotation matrix
    R = rotation_matrix(theta)

    # Transform linear velocity (subtract p × ω term first)
    v_s = np.array([vx_s, vy_s])
    v_b = R.T @ (v_s - np.array([y, -x]) * omega_s)

    # Angular velocity is frame-independent
    return np.array([omega_s, v_b[0], v_b[1]])


def spatial_to_body_acceleration(
    pose: np.ndarray,
    spatial_twist: np.ndarray,
    spatial_accel: np.ndarray
) -> np.ndarray:
    """
    Transform acceleration from spatial frame to body frame.

    Includes proper Coriolis/centrifugal terms due to rotating frame.

    For SE(2):
    dV_b/dt = R^T * (dV_s/dt - [py, -px]^T * α_s - ω_s * J * v_s)

    where J = [0 -1; 1 0] is the 90° rotation matrix (perpendicular operator)

    Args:
        pose: Current pose [x, y, theta]
        spatial_twist: Spatial frame twist [omega_s, vx_s, vy_s]
        spatial_accel: Spatial frame acceleration [alpha_s, ax_s, ay_s]

    Returns:
        Body frame acceleration [alpha_b, ax_b, ay_b]
    """
    # Extract components
    x, y, theta = pose
    omega_s, vx_s, vy_s = spatial_twist
    alpha_s, ax_s, ay_s = spatial_accel

    # Rotation matrix
    R = rotation_matrix(theta)

    # Linear acceleration transformation
    a_s = np.array([ax_s, ay_s])

    # Subtract p × α term
    a_s_minus_p_alpha = a_s - np.array([y, -x]) * alpha_s

    # Coriolis correction: ω × v in 2D (perpendicular operator)
    v_s = np.array([vx_s, vy_s])
    J = np.array([[0, -1], [1, 0]])  # 90° rotation
    coriolis = omega_s * (J @ v_s)  # ω × v = ω * J * v in 2D

    # Transform to body frame
    a_b = R.T @ (a_s_minus_p_alpha - coriolis)

    # Angular acceleration is frame-independent
    return np.array([alpha_s, a_b[0], a_b[1]])


def se2_interpolate(pose_start: np.ndarray, pose_end: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two SE(2) poses using geodesic interpolation.

    This performs proper interpolation in the SE(2) manifold using
    exponential coordinates.

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

    In SE(2), a screw motion is characterized by instantaneous twist.

    Args:
        pose_start: Start pose [x, y, theta]
        pose_end: End pose [x, y, theta]

    Returns:
        Tuple of (screw_axis [omega, vx, vy], magnitude)
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


def wrench_body_to_spatial(pose: np.ndarray, wrench_body: np.ndarray) -> np.ndarray:
    """
    Transform wrench from body frame to spatial frame.

    Wrench transformation uses Adjoint transpose:
    F_s = Ad_T^T * F_b

    Args:
        pose: Current pose [x, y, theta]
        wrench_body: Body frame wrench [tau, fx_b, fy_b]

    Returns:
        Spatial frame wrench [tau, fx_s, fy_s]
    """
    T = SE2Pose.from_array(pose).to_matrix()
    Ad = se2_adjoint(T)
    return Ad.T @ wrench_body


def wrench_spatial_to_body(pose: np.ndarray, wrench_spatial: np.ndarray) -> np.ndarray:
    """
    Transform wrench from spatial frame to body frame.

    F_b = Ad_{T^{-1}}^T * F_s

    Args:
        pose: Current pose [x, y, theta]
        wrench_spatial: Spatial frame wrench [tau, fx_s, fy_s]

    Returns:
        Body frame wrench [tau, fx_b, fy_b]
    """
    T_inv = se2_inverse(SE2Pose.from_array(pose).to_matrix())
    Ad_inv = se2_adjoint(T_inv)
    return Ad_inv.T @ wrench_spatial


def integrate_twist(pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate twist to update pose (proper SE(2) integration).

    Uses exponential map for exact integration on SE(2) manifold.

    Args:
        pose: Current pose [x, y, theta]
        twist: Twist in body frame [omega, vx, vy]
        dt: Time step

    Returns:
        Updated pose [x, y, theta]
    """
    # Scale twist by time step
    xi = twist * dt

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
