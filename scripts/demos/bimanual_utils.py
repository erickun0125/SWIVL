"""
Shared utilities for bimanual manipulation demos.

This module contains common functionality used by both teleoperation and rule-based demos.
"""

import numpy as np
from src.envs.object_manager import JointType


def compute_constrained_velocity(
    env,
    controlled_ee_idx: int,
    controlled_velocity: np.ndarray,
    joint_velocity: float
) -> np.ndarray:
    """
    Compute the other EE's velocity based on kinematic constraints.
    
    When one EE is controlled and joint velocity is specified,
    the other EE must follow the kinematic constraint of the articulated object.
    
    Args:
        env: BiArtEnv instance
        controlled_ee_idx: 0 (left) or 1 (right)
        controlled_velocity: [vx, vy, omega] of controlled EE in world frame
        joint_velocity: Joint velocity (rad/s for revolute, pixels/s for prismatic)
    
    Returns:
        other_velocity: [vx, vy, omega] of other EE in world frame
    """
    joint_type = env.object_manager.joint_type
    link_poses = env.object_manager.get_link_poses()
    grasping_poses = env.object_manager.get_grasping_poses()
    cfg = env.object_config
    
    link1_pose = link_poses[0]
    link2_pose = link_poses[1]
    left_ee_pose = grasping_poses["left"]
    right_ee_pose = grasping_poses["right"]
    
    # Joint position (link1's right end = link2's left end)
    joint_pos = link1_pose[:2] + (cfg.link_length / 2) * np.array([
        np.cos(link1_pose[2]),
        np.sin(link1_pose[2])
    ])
    
    if joint_type == JointType.FIXED:
        return controlled_velocity.copy()
    
    elif joint_type == JointType.REVOLUTE:
        if controlled_ee_idx == 0:  # Left EE (on link1) is controlled
            omega_link1 = controlled_velocity[2]
            left_offset = left_ee_pose[:2] - link1_pose[:2]
            v_link1 = controlled_velocity[:2] - omega_link1 * np.array([-left_offset[1], left_offset[0]])
            joint_offset = joint_pos - link1_pose[:2]
            v_joint = v_link1 + omega_link1 * np.array([-joint_offset[1], joint_offset[0]])
            omega_link2 = omega_link1 + joint_velocity
            link2_offset = link2_pose[:2] - joint_pos
            v_link2 = v_joint + omega_link2 * np.array([-link2_offset[1], link2_offset[0]])
            right_offset = right_ee_pose[:2] - link2_pose[:2]
            v_right = v_link2 + omega_link2 * np.array([-right_offset[1], right_offset[0]])
            return np.array([v_right[0], v_right[1], omega_link2])
        else:  # Right EE (on link2) is controlled
            omega_link2 = controlled_velocity[2]
            right_offset = right_ee_pose[:2] - link2_pose[:2]
            v_link2 = controlled_velocity[:2] - omega_link2 * np.array([-right_offset[1], right_offset[0]])
            link2_offset = link2_pose[:2] - joint_pos
            v_joint = v_link2 - omega_link2 * np.array([-link2_offset[1], link2_offset[0]])
            omega_link1 = omega_link2 - joint_velocity
            joint_offset = joint_pos - link1_pose[:2]
            v_link1 = v_joint - omega_link1 * np.array([-joint_offset[1], joint_offset[0]])
            left_offset = left_ee_pose[:2] - link1_pose[:2]
            v_left = v_link1 + omega_link1 * np.array([-left_offset[1], left_offset[0]])
            return np.array([v_left[0], v_left[1], omega_link1])
    
    elif joint_type == JointType.PRISMATIC:
        slide_dir = np.array([np.cos(link1_pose[2]), np.sin(link1_pose[2])])
        perp_dir = np.array([-np.sin(link1_pose[2]), np.cos(link1_pose[2])])
        joint_state = env.object_manager.get_joint_state()
        center_distance = joint_state + cfg.link_length / 2
        
        if controlled_ee_idx == 0:  # Left EE controlled
            omega_link1 = controlled_velocity[2]
            left_offset = left_ee_pose[:2] - link1_pose[:2]
            v_link1 = controlled_velocity[:2] - omega_link1 * np.array([-left_offset[1], left_offset[0]])
            omega_link2 = omega_link1
            v_link2 = v_link1 + joint_velocity * slide_dir + omega_link1 * center_distance * perp_dir
            right_offset = right_ee_pose[:2] - link2_pose[:2]
            v_right = v_link2 + omega_link2 * np.array([-right_offset[1], right_offset[0]])
            return np.array([v_right[0], v_right[1], omega_link2])
        else:  # Right EE controlled
            omega_link2 = controlled_velocity[2]
            right_offset = right_ee_pose[:2] - link2_pose[:2]
            v_link2 = controlled_velocity[:2] - omega_link2 * np.array([-right_offset[1], right_offset[0]])
            omega_link1 = omega_link2
            v_link1 = v_link2 - joint_velocity * slide_dir - omega_link1 * center_distance * perp_dir
            left_offset = left_ee_pose[:2] - link1_pose[:2]
            v_left = v_link1 + omega_link1 * np.array([-left_offset[1], left_offset[0]])
            return np.array([v_left[0], v_left[1], omega_link1])
    
    return controlled_velocity.copy()

