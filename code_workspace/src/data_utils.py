"""
Data Collection Utilities for Bimanual Manipulation Demos

Shared DataCollector for saving demonstration episodes to HDF5 format.
Used by both rule-based and teleoperation data collection scripts.
"""

import os
import numpy as np
import h5py
from datetime import datetime
from typing import Dict, Optional


class DataCollector:
    """Collects and saves demonstration data to HDF5."""

    def __init__(self, save_dir: str, filename_prefix: str = "demo", demo_type: str = "unknown"):
        """
        Initialize data collector.

        Args:
            save_dir: Directory to save HDF5 files
            filename_prefix: Prefix for saved filenames (e.g., 'rule_based', 'teleop')
            demo_type: Type of demonstration stored in HDF5 metadata
        """
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        self.demo_type = demo_type
        os.makedirs(save_dir, exist_ok=True)
        self.reset_buffer()

    def reset_buffer(self):
        """Reset data buffer."""
        self.images = []
        self.ee_poses = []
        self.ee_velocities = []
        self.external_wrenches = []
        self.joint_states = []
        self.link_poses = []
        self.actions = []

    def add_step(self, obs: Dict, action: np.ndarray, image: np.ndarray):
        """
        Add a step to the buffer.

        Args:
            obs: Observation dictionary from BiArtEnv
            action: Action taken (desired poses)
            image: Rendered image (H, W, 3)
        """
        self.images.append(image)
        self.ee_poses.append(obs['ee_poses'])
        self.ee_velocities.append(obs['ee_velocities'])
        self.external_wrenches.append(obs['external_wrenches'])
        self.link_poses.append(obs['link_poses'])

        joint_state = obs.get('joint_state', 0.0)
        self.joint_states.append(joint_state)

        self.actions.append(action)

    def save_episode(self, episode_idx: int, joint_type: str = 'revolute') -> Optional[str]:
        """Save buffered data to HDF5."""
        if len(self.images) == 0:
            print("No data to save.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{joint_type}_{timestamp}_ep{episode_idx}.h5"
        filepath = os.path.join(self.save_dir, filename)

        images = np.array(self.images, dtype=np.uint8)
        ee_poses = np.array(self.ee_poses, dtype=np.float32)
        ee_velocities = np.array(self.ee_velocities, dtype=np.float32)
        external_wrenches = np.array(self.external_wrenches, dtype=np.float32)
        link_poses = np.array(self.link_poses, dtype=np.float32)
        joint_states = np.array(self.joint_states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)

        print(f"Saving episode {episode_idx} with {len(images)} steps to {filepath}...")

        with h5py.File(filepath, 'w') as f:
            obs_group = f.create_group('obs')
            action_group = f.create_group('action')

            obs_group.create_dataset('images', data=images, compression='gzip')
            obs_group.create_dataset('ee_poses', data=ee_poses)
            obs_group.create_dataset('ee_velocities', data=ee_velocities)
            obs_group.create_dataset('external_wrenches', data=external_wrenches)
            obs_group.create_dataset('link_poses', data=link_poses)
            obs_group.create_dataset('joint_states', data=joint_states)

            action_group.create_dataset('desired_poses', data=actions)

            f.attrs['num_steps'] = len(images)
            f.attrs['joint_type'] = joint_type
            f.attrs['demo_type'] = self.demo_type

        print("Save complete.")
        self.reset_buffer()
        return filepath
