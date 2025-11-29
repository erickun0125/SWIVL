"""
Reward Manager

Manages reward computation for reinforcement learning.

Provides various reward components:
- Pose tracking reward
- Velocity tracking reward
- Energy efficiency reward
- Safety reward (wrench limits)
- Success/failure detection

This manager is used by BiArt environment for RL training.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from src.se2_math import normalize_angle, se2_log, se2_inverse, SE2Pose


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    pose_tracking: float = 1.0
    velocity_tracking: float = 0.5
    energy_efficiency: float = 0.1
    safety: float = 0.2
    success_bonus: float = 10.0
    failure_penalty: float = -10.0

    # Configurable parameters for reward computation
    pose_error_scale: float = 50.0  # Scale for exponential pose tracking reward
    velocity_error_scale: float = 20.0  # Scale for exponential velocity tracking reward
    angular_to_linear_ratio: float = 10.0  # Weight ratio between angular and linear error in SE(2) geodesic distance


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    
    # Reward component weights
    weights: RewardWeights = None  # Will be initialized in __post_init__
    
    # Success/failure thresholds
    success_threshold_pos: float = 7.0  # pixels
    success_threshold_angle: float = 0.2  # radians
    
    # Safety thresholds
    max_wrench_threshold: float = 200.0  # force/torque limit
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = RewardWeights()


class RewardManager:
    """
    Manager for reward computation.

    Computes reward based on multiple components for RL training.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward manager.

        Args:
            config: Reward configuration (uses defaults if None)
        """
        self.config = config if config is not None else RewardConfig()
        self.weights = self.config.weights
        self.success_threshold_pos = self.config.success_threshold_pos
        self.success_threshold_angle = self.config.success_threshold_angle
        self.max_wrench_threshold = self.config.max_wrench_threshold

    def compute_reward(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray,
        current_velocities: np.ndarray,
        desired_velocities: np.ndarray,
        applied_wrenches: np.ndarray,
        external_wrenches: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute total reward and components.

        Args:
            current_poses: Current tracked poses (num_items, 3)
            desired_poses: Desired poses (num_items, 3)
            current_velocities: Current velocities (num_items, 3)
            desired_velocities: Desired velocities (num_items, 3)
            applied_wrenches: Applied control wrenches (num_ee, 3)
            external_wrenches: Optional external wrenches (num_ee, 3)

        Returns:
            Dictionary with 'total_reward' and individual components
        """
        # 1. Pose tracking reward
        pose_reward = self._compute_pose_tracking_reward(
            current_poses, desired_poses
        )

        # 2. Velocity tracking reward
        velocity_reward = self._compute_velocity_tracking_reward(
            current_velocities, desired_velocities
        )

        # 3. Energy efficiency reward (penalize large control efforts)
        energy_reward = self._compute_energy_reward(applied_wrenches)

        # 4. Safety reward (penalize excessive wrenches)
        safety_reward = self._compute_safety_reward(
            applied_wrenches, external_wrenches
        )

        # 5. Success/failure bonus
        is_success = self._check_success(current_poses, desired_poses)
        is_failure = self._check_failure(applied_wrenches)

        bonus = 0.0
        if is_success:
            bonus = self.weights.success_bonus
        elif is_failure:
            bonus = self.weights.failure_penalty

        # Total reward
        total_reward = (
            self.weights.pose_tracking * pose_reward +
            self.weights.velocity_tracking * velocity_reward +
            self.weights.energy_efficiency * energy_reward +
            self.weights.safety * safety_reward +
            bonus
        )

        return {
            'total_reward': total_reward,
            'pose_tracking': pose_reward,
            'velocity_tracking': velocity_reward,
            'energy_efficiency': energy_reward,
            'safety': safety_reward,
            'bonus': bonus,
            'is_success': is_success,
            'is_failure': is_failure
        }

    def _compute_se2_geodesic_distance(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray
    ) -> float:
        """
        Compute SE(2) geodesic distance using logarithm map.

        This is the proper distance metric on the SE(2) manifold.

        Args:
            current_pose: Current pose [x, y, theta]
            desired_pose: Desired pose [x, y, theta]

        Returns:
            Geodesic distance (scalar)
        """
        T_current = SE2Pose.from_array(current_pose).to_matrix()
        T_desired = SE2Pose.from_array(desired_pose).to_matrix()

        # Compute relative transformation
        T_error = se2_inverse(T_current) @ T_desired

        # Logarithm map gives the error in se(2) algebra
        error = se2_log(T_error)  # [omega, vx, vy] (MR convention)

        # Compute weighted norm (weight rotation and translation differently)
        # error = [omega, vx, vy]
        angular_error = np.abs(error[0])  # MR: angular component at index 0
        linear_error = np.linalg.norm(error[1:3])  # MR: linear components at indices 1,2

        # Weighted combination (configurable weight ratio)
        total_error = linear_error + self.weights.angular_to_linear_ratio * angular_error

        return total_error

    def _compute_pose_tracking_reward(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray
    ) -> float:
        """
        Compute pose tracking reward using SE(2) geodesic distance.

        Uses exponential reward: r = exp(-error)
        """
        # Compute SE(2) geodesic distance for each EE
        errors = []
        for i in range(len(current_poses)):
            error = self._compute_se2_geodesic_distance(
                current_poses[i],
                desired_poses[i]
            )
            errors.append(error)

        # Average error
        avg_error = np.mean(errors)

        # Exponential reward (higher is better, configurable scale)
        reward = np.exp(-avg_error / self.weights.pose_error_scale)

        return reward

    def _compute_velocity_tracking_reward(
        self,
        current_velocities: np.ndarray,
        desired_velocities: np.ndarray
    ) -> float:
        """
        Compute velocity tracking reward.
        """
        # Velocity error
        vel_error = np.linalg.norm(
            current_velocities - desired_velocities,
            axis=1
        ).mean()

        # Exponential reward (configurable scale)
        reward = np.exp(-vel_error / self.weights.velocity_error_scale)

        return reward

    def _compute_energy_reward(self, wrenches: np.ndarray) -> float:
        """
        Compute energy efficiency reward.

        Penalizes large control efforts.

        Following MR convention: wrench = [tau, fx, fy]
        """
        # Average wrench magnitude
        wrench_magnitudes = np.array([
            np.abs(w[0]) + np.linalg.norm(w[1:3])  # MR: torque (index 0) + force (indices 1,2)
            for w in wrenches
        ])
        avg_magnitude = wrench_magnitudes.mean()

        # Penalty for large wrenches
        reward = -avg_magnitude / 100.0

        return reward

    def _compute_safety_reward(
        self,
        applied_wrenches: np.ndarray,
        external_wrenches: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute safety reward.

        Penalizes excessive wrenches that might damage system.

        Following MR convention: wrench = [tau, fx, fy]
        """
        # Check applied wrenches (MR: [tau, fx, fy])
        max_applied = np.array([
            np.abs(w[0]) + np.linalg.norm(w[1:3])  # MR: torque (index 0) + force (indices 1,2)
            for w in applied_wrenches
        ]).max()

        # Check external wrenches if available
        max_external = 0.0
        if external_wrenches is not None:
            max_external = np.array([
                np.abs(w[0]) + np.linalg.norm(w[1:3])  # MR: torque (index 0) + force (indices 1,2)
                for w in external_wrenches
            ]).max()

        # Penalty if exceeding threshold
        if max_applied > self.max_wrench_threshold:
            penalty = -(max_applied - self.max_wrench_threshold) / 100.0
        else:
            penalty = 0.0

        if max_external > self.max_wrench_threshold * 0.5:
            penalty += -(max_external - self.max_wrench_threshold * 0.5) / 100.0

        return penalty

    def _check_success(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray
    ) -> bool:
        """
        Check if task is successful.

        Success: all EEs within threshold of desired poses.
        """
        # Position errors
        pos_errors = np.linalg.norm(
            current_poses[:, :2] - desired_poses[:, :2],
            axis=1
        )

        # Angle errors
        angle_errors = np.array([
            np.abs(normalize_angle(current_poses[i, 2] - desired_poses[i, 2]))
            for i in range(len(current_poses))
        ])

        # Check all within threshold
        pos_ok = (pos_errors < self.success_threshold_pos).all()
        angle_ok = (angle_errors < self.success_threshold_angle).all()

        return pos_ok and angle_ok

    def _check_failure(self, wrenches: np.ndarray) -> bool:
        """
        Check if task failed.

        Failure: wrench exceeds safety limit.

        Following MR convention: wrench = [tau, fx, fy]
        """
        max_wrench = np.array([
            np.abs(w[0]) + np.linalg.norm(w[1:3])  # MR: torque (index 0) + force (indices 1,2)
            for w in wrenches
        ]).max()

        return max_wrench > self.max_wrench_threshold * 1.5 * 10

    def update_weights(self, new_weights: RewardWeights):
        """Update reward weights."""
        self.weights = new_weights

    def get_info(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray,
        applied_wrenches: np.ndarray,
        external_wrenches: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get info dictionary for logging.

        Args:
            current_poses: Current EE poses
            desired_poses: Desired EE poses
            applied_wrenches: Applied wrenches
            external_wrenches: Optional external wrenches

        Returns:
            Info dictionary
        """
        # Position error
        pos_error = np.linalg.norm(
            current_poses[:, :2] - desired_poses[:, :2],
            axis=1
        ).mean()

        # Angle error
        angle_errors = np.array([
            np.abs(normalize_angle(current_poses[i, 2] - desired_poses[i, 2]))
            for i in range(len(current_poses))
        ])
        angle_error = angle_errors.mean()

        # Wrench magnitudes (MR convention: [tau, fx, fy])
        wrench_magnitudes = np.array([
            np.abs(w[0]) + np.linalg.norm(w[1:3])  # MR: torque (index 0) + force (indices 1,2)
            for w in applied_wrenches
        ])

        info = {
            'pos_error': pos_error,
            'angle_error': angle_error,
            'max_wrench': wrench_magnitudes.max(),
            'avg_wrench': wrench_magnitudes.mean(),
        }

        if external_wrenches is not None:
            ext_magnitudes = np.array([
                np.abs(w[0]) + np.linalg.norm(w[1:3])  # MR: torque (index 0) + force (indices 1,2)
                for w in external_wrenches
            ])
            info['max_external_wrench'] = ext_magnitudes.max()

        return info
