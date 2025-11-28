"""
Bimanual Manipulation Environments

This package provides SE(2) bimanual manipulation environments:
- BiArtEnv: Main bimanual manipulation environment with articulated objects
- EndEffectorManager: Manager for parallel gripper end-effectors
- ObjectManager: Manager for articulated objects
- RewardManager: Manager for RL reward computation
"""

from src.envs.biart import BiArtEnv
from src.envs.end_effector_manager import EndEffectorManager, GripperConfig
from src.envs.object_manager import ObjectManager, ObjectConfig, JointType
from src.envs.reward_manager import RewardManager, RewardWeights

__all__ = [
    'BiArtEnv',
    'EndEffectorManager',
    'GripperConfig',
    'ObjectManager',
    'ObjectConfig',
    'JointType',
    'RewardManager',
    'RewardWeights',
]
