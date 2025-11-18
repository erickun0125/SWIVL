"""
Grasping Frame Definitions and Utilities

This module defines grasping frames for articulated objects and provides
utilities for computing desired poses based on object configuration.
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

from src.se2_math import SE2Pose, normalize_angle


@dataclass
class GraspingFrame:
    """
    Grasping frame definition for a link.

    The grasping frame is defined in the link's local coordinate frame.
    When a gripper grasps at this frame, the gripper's body frame should
    align with this grasping frame.

    Attributes:
        link_id: ID of the link this frame belongs to
        local_pose: Pose in link's local frame [x, y, theta]
        name: Name of this grasping frame (e.g., "left_grasp", "right_grasp")
    """
    link_id: int
    local_pose: np.ndarray  # [x, y, theta] in link's local frame
    name: str


class GraspingFrameManager:
    """
    Manages grasping frames for articulated objects.

    This class stores predefined grasping frames for each object type
    and provides methods to compute world-frame poses based on current
    object configuration.
    """

    def __init__(self, joint_type: str, link_length: float = 40.0):
        """
        Initialize grasping frame manager.

        Args:
            joint_type: Type of joint ("revolute", "prismatic", "fixed")
            link_length: Length of each link
        """
        self.joint_type = joint_type
        self.link_length = link_length

        # Define grasping frames for this object type
        self.grasping_frames = self._define_grasping_frames()

    def _define_grasping_frames(self) -> Dict[str, GraspingFrame]:
        """
        Define grasping frames for the object.

        For a two-link object:
        - Left gripper grasps link1 at its left end (away from joint)
        - Right gripper grasps link2 at its right end (away from joint)

        Grasping frame orientation is perpendicular to the link (gripper jaws
        open/close perpendicular to link length).

        Returns:
            Dictionary of grasping frames {gripper_name: GraspingFrame}
        """
        frames = {}

        # Left gripper grasps link1's left side (offset -length/4 from center)
        # Gripper frame is rotated 90° relative to link (jaws perpendicular)
        frames["left"] = GraspingFrame(
            link_id=0,  # Link 1
            local_pose=np.array([-self.link_length / 4, 0.0, np.pi / 2]),
            name="left_grasp"
        )

        # Right gripper grasps link2's right side (offset +length/4 from center)
        # Gripper frame is rotated 90° relative to link
        frames["right"] = GraspingFrame(
            link_id=1,  # Link 2
            local_pose=np.array([self.link_length / 4, 0.0, np.pi / 2]),
            name="right_grasp"
        )

        return frames

    def get_world_frame_pose(
        self,
        gripper_name: str,
        link_pose: np.ndarray
    ) -> np.ndarray:
        """
        Compute grasping frame pose in world frame.

        Args:
            gripper_name: Name of gripper ("left" or "right")
            link_pose: Current pose of the link [x, y, theta]

        Returns:
            Grasping frame pose in world frame [x, y, theta]
        """
        if gripper_name not in self.grasping_frames:
            raise ValueError(f"Unknown gripper: {gripper_name}")

        grasp_frame = self.grasping_frames[gripper_name]

        # Convert link pose to SE2Pose
        link_T = SE2Pose.from_array(link_pose).to_matrix()

        # Convert local grasp pose to SE2Pose
        grasp_local_T = SE2Pose.from_array(grasp_frame.local_pose).to_matrix()

        # Compose to get world frame
        grasp_world_T = link_T @ grasp_local_T

        # Convert back to pose array
        grasp_world_pose = SE2Pose.from_matrix(grasp_world_T).to_array()

        return grasp_world_pose

    def get_all_grasping_poses(
        self,
        link1_pose: np.ndarray,
        link2_pose: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get all grasping frame poses in world frame.

        Args:
            link1_pose: Pose of link 1 [x, y, theta]
            link2_pose: Pose of link 2 [x, y, theta]

        Returns:
            Dictionary {gripper_name: world_pose}
        """
        link_poses = [link1_pose, link2_pose]

        grasping_poses = {}
        for gripper_name, grasp_frame in self.grasping_frames.items():
            link_pose = link_poses[grasp_frame.link_id]
            grasping_poses[gripper_name] = self.get_world_frame_pose(
                gripper_name, link_pose
            )

        return grasping_poses

    def compute_desired_gripper_velocity(
        self,
        controlled_gripper: str,
        controlled_velocity: np.ndarray,
        joint_velocity: float,
        link1_pose: np.ndarray,
        link2_pose: np.ndarray,
        link1_velocity: np.ndarray,
        link2_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute desired velocity for the non-controlled gripper.

        Given:
        - Controlled gripper velocity (vx, vy, omega)
        - Object joint velocity (joint_vel)

        Compute the other gripper's velocity such that:
        - Object constraint is satisfied
        - Controlled gripper moves with commanded velocity
        - Joint moves with commanded velocity

        Args:
            controlled_gripper: Which gripper is controlled ("left" or "right")
            controlled_velocity: Velocity of controlled gripper [vx, vy, omega] in world frame
            joint_velocity: Commanded joint velocity (rad/s for revolute, m/s for prismatic)
            link1_pose: Current pose of link 1
            link2_pose: Current pose of link 2
            link1_velocity: Current velocity of link 1
            link2_velocity: Current velocity of link 2

        Returns:
            Desired velocity for other gripper [vx, vy, omega] in world frame
        """
        # Determine which gripper is automatic
        other_gripper = "right" if controlled_gripper == "left" else "left"

        # Get grasping frames
        controlled_frame = self.grasping_frames[controlled_gripper]
        other_frame = self.grasping_frames[other_gripper]

        # Get current grasping poses
        link_poses = [link1_pose, link2_pose]
        controlled_link_pose = link_poses[controlled_frame.link_id]
        other_link_pose = link_poses[other_frame.link_id]

        # Compute link velocities based on gripper motion
        # This is a simplified version - proper implementation would use
        # object dynamics and constraint equations

        if self.joint_type == "revolute":
            # For revolute joint, the relative angular velocity is joint_velocity
            # Link velocities depend on gripper grasping and joint motion

            # Simplified: assume controlled link matches controlled gripper velocity
            # and other link velocity is computed from joint constraint

            # Get controlled link velocity from controlled gripper
            # (In reality, need to transform from grasp point to link COM)
            controlled_link_vel = controlled_velocity.copy()

            # Compute other link angular velocity
            other_link_omega = controlled_link_vel[2] + joint_velocity

            # For position velocity, use joint constraint
            # Connection point velocity must be consistent
            # v_connection = v_link1 + omega_link1 × r1 = v_link2 + omega_link2 × r2

            # Simplified: other link linear velocity similar to controlled
            # but adjusted for angular difference
            other_link_vel = np.array([
                controlled_link_vel[0],
                controlled_link_vel[1],
                other_link_omega
            ])

        elif self.joint_type == "prismatic":
            # For prismatic joint, links move together with joint sliding
            # Angular velocities are the same
            other_link_vel = controlled_velocity.copy()

            # Linear velocity includes joint sliding component
            # (Simplified - proper implementation needs constraint equation)
            slide_direction = np.array([np.cos(link1_pose[2]), np.sin(link1_pose[2])])
            other_link_vel[:2] += slide_direction * joint_velocity

        else:  # fixed
            # For fixed joint, both links move together
            other_link_vel = controlled_velocity.copy()

        # Transform link velocity to gripper velocity at grasping point
        # This requires transforming from link COM to grasp point

        # Simplified: use link velocity directly as approximation
        # Proper implementation would compute:
        # v_grasp = v_link + omega_link × r_grasp_to_com

        other_gripper_vel = other_link_vel.copy()

        return other_gripper_vel
