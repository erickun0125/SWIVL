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

from src.se2_math import normalize_angle


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    pose_tracking: float = 1.0
    velocity_tracking: float = 0.5
    energy_efficiency: float = 0.1
    safety: float = 0.2
    success_bonus: float = 10.0
    failure_penalty: float = -10.0


class RewardManager:
    """
    Manager for reward computation.

    Computes reward based on multiple components for RL training.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        success_threshold_pos: float = 20.0,  # pixels
        success_threshold_angle: float = 0.2,  # radians
        max_wrench_threshold: float = 200.0,  # force/torque limit
    ):
        """
        Initialize reward manager.

        Args:
            weights: Reward component weights
            success_threshold_pos: Position error threshold for success
            success_threshold_angle: Angle error threshold for success
            max_wrench_threshold: Maximum allowed wrench magnitude
        """
        self.weights = weights if weights is not None else RewardWeights()
        self.success_threshold_pos = success_threshold_pos
        self.success_threshold_angle = success_threshold_angle
        self.max_wrench_threshold = max_wrench_threshold

    def compute_reward(
        self,
        current_ee_poses: np.ndarray,
        desired_ee_poses: np.ndarray,
        current_ee_velocities: np.ndarray,
        desired_ee_velocities: np.ndarray,
        applied_wrenches: np.ndarray,
        external_wrenches: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute total reward and components.

        Args:
            current_ee_poses: Current EE poses (num_ee, 3)
            desired_ee_poses: Desired EE poses (num_ee, 3)
            current_ee_velocities: Current EE velocities (num_ee, 3)
            desired_ee_velocities: Desired EE velocities (num_ee, 3)
            applied_wrenches: Applied control wrenches (num_ee, 3)
            external_wrenches: Optional external wrenches (num_ee, 3)

        Returns:
            Dictionary with 'total_reward' and individual components
        """
        # 1. Pose tracking reward
        pose_reward = self._compute_pose_tracking_reward(
            current_ee_poses, desired_ee_poses
        )

        # 2. Velocity tracking reward
        velocity_reward = self._compute_velocity_tracking_reward(
            current_ee_velocities, desired_ee_velocities
        )

        # 3. Energy efficiency reward (penalize large control efforts)
        energy_reward = self._compute_energy_reward(applied_wrenches)

        # 4. Safety reward (penalize excessive wrenches)
        safety_reward = self._compute_safety_reward(
            applied_wrenches, external_wrenches
        )

        # 5. Success/failure bonus
        is_success = self._check_success(current_ee_poses, desired_ee_poses)
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

    def _compute_pose_tracking_reward(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray
    ) -> float:
        """
        Compute pose tracking reward.

        Uses exponential reward: r = exp(-error)
        """
        # Compute position error (Euclidean distance)
        pos_error = np.linalg.norm(
            current_poses[:, :2] - desired_poses[:, :2],
            axis=1
        ).mean()

        # Compute angle error
        angle_errors = np.array([
            np.abs(normalize_angle(current_poses[i, 2] - desired_poses[i, 2]))
            for i in range(len(current_poses))
        ])
        angle_error = angle_errors.mean()

        # Weighted error
        total_error = pos_error + 10.0 * angle_error  # Weight angle more

        # Exponential reward (higher is better)
        reward = np.exp(-total_error / 50.0)

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

        # Exponential reward
        reward = np.exp(-vel_error / 20.0)

        return reward

    def _compute_energy_reward(self, wrenches: np.ndarray) -> float:
        """
        Compute energy efficiency reward.

        Penalizes large control efforts.
        """
        # Average wrench magnitude
        wrench_magnitudes = np.array([
            np.linalg.norm(w[:2]) + np.abs(w[2])  # Force + torque
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
        """
        # Check applied wrenches
        max_applied = np.array([
            np.linalg.norm(w[:2]) + np.abs(w[2])
            for w in applied_wrenches
        ]).max()

        # Check external wrenches if available
        max_external = 0.0
        if external_wrenches is not None:
            max_external = np.array([
                np.linalg.norm(w[:2]) + np.abs(w[2])
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
        """
        max_wrench = np.array([
            np.linalg.norm(w[:2]) + np.abs(w[2])
            for w in wrenches
        ]).max()

        return max_wrench > self.max_wrench_threshold * 1.5

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

        # Wrench magnitudes
        wrench_magnitudes = np.array([
            np.linalg.norm(w[:2]) + np.abs(w[2])
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
                np.linalg.norm(w[:2]) + np.abs(w[2])
                for w in external_wrenches
            ])
            info['max_external_wrench'] = ext_magnitudes.max()

        return info
