"""
Linkage Object Manager

Manages articulated linkage objects with different joint types:
- R (Revolute): Rotational joint
- P (Prismatic): Sliding/translational joint
- Fixed: Rigid connection

Provides utilities for:
- Computing forward kinematics
- Updating joint configurations
- Querying joint states and limits
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
import pymunk
from pymunk.vec2d import Vec2d


class JointType(Enum):
    """Supported joint types."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


class LinkageJoint:
    """
    Represents a single joint in a linkage mechanism.

    Attributes:
        joint_type: Type of joint (revolute, prismatic, or fixed)
        parent_link: Parent link index
        child_link: Child link index
        joint_value: Current joint value (angle for R, distance for P)
        joint_limits: (min, max) limits for joint value
        pymunk_joint: Reference to pymunk constraint object
    """

    def __init__(
        self,
        joint_type: JointType,
        parent_link: int,
        child_link: int,
        joint_limits: Optional[Tuple[float, float]] = None,
        pymunk_joint: Optional[pymunk.Constraint] = None
    ):
        """
        Initialize a linkage joint.

        Args:
            joint_type: Type of joint
            parent_link: Index of parent link
            child_link: Index of child link
            joint_limits: Optional (min, max) limits for joint value
            pymunk_joint: Optional reference to pymunk constraint
        """
        self.joint_type = joint_type
        self.parent_link = parent_link
        self.child_link = child_link
        self.joint_value = 0.0
        self.joint_velocity = 0.0

        # Set default limits based on joint type
        if joint_limits is None:
            if joint_type == JointType.REVOLUTE:
                self.joint_limits = (-np.pi, np.pi)
            elif joint_type == JointType.PRISMATIC:
                self.joint_limits = (-100.0, 100.0)  # Default linear limits
            else:  # FIXED
                self.joint_limits = (0.0, 0.0)
        else:
            self.joint_limits = joint_limits

        self.pymunk_joint = pymunk_joint

    def update_state(self, link_bodies: List[pymunk.Body]):
        """
        Update joint state based on current link poses.

        Args:
            link_bodies: List of pymunk body objects for all links
        """
        if self.joint_type == JointType.FIXED:
            self.joint_value = 0.0
            self.joint_velocity = 0.0
            return

        # Check if bodies are set
        if (self.parent_link >= len(link_bodies) or
            self.child_link >= len(link_bodies) or
            link_bodies[self.parent_link] is None or
            link_bodies[self.child_link] is None):
            # Bodies not set yet, skip update
            return

        parent = link_bodies[self.parent_link]
        child = link_bodies[self.child_link]

        if self.joint_type == JointType.REVOLUTE:
            # Compute relative angle
            self.joint_value = child.angle - parent.angle
            # Normalize to [-pi, pi]
            self.joint_value = np.arctan2(np.sin(self.joint_value), np.cos(self.joint_value))

            # Compute angular velocity difference
            self.joint_velocity = child.angular_velocity - parent.angular_velocity

        elif self.joint_type == JointType.PRISMATIC:
            # Compute relative position along parent's x-axis
            relative_pos = Vec2d(*child.position) - Vec2d(*parent.position)

            # Transform to parent frame
            cos_angle = np.cos(-parent.angle)
            sin_angle = np.sin(-parent.angle)
            local_x = cos_angle * relative_pos.x - sin_angle * relative_pos.y

            self.joint_value = local_x

            # Compute velocity along parent's x-axis
            relative_vel = child.velocity - parent.velocity
            local_vel_x = cos_angle * relative_vel.x - sin_angle * relative_vel.y
            self.joint_velocity = local_vel_x

    def is_within_limits(self) -> bool:
        """Check if current joint value is within limits."""
        return self.joint_limits[0] <= self.joint_value <= self.joint_limits[1]

    def clamp_to_limits(self, value: float) -> float:
        """Clamp a value to joint limits."""
        return np.clip(value, self.joint_limits[0], self.joint_limits[1])


class LinkageObject:
    """
    Manages a multi-link articulated object.

    Represents a kinematic chain of links connected by joints.
    Provides forward kinematics, state queries, and configuration updates.
    """

    def __init__(self, num_links: int):
        """
        Initialize linkage object.

        Args:
            num_links: Number of links in the kinematic chain
        """
        self.num_links = num_links
        self.links: List[Optional[pymunk.Body]] = [None] * num_links
        self.joints: List[LinkageJoint] = []

        # Link properties
        self.link_lengths: List[float] = [40.0] * num_links  # Default length
        self.link_masses: List[float] = [0.5] * num_links   # Default mass

    def add_joint(
        self,
        joint_type: JointType,
        parent_link: int,
        child_link: int,
        joint_limits: Optional[Tuple[float, float]] = None,
        pymunk_joint: Optional[pymunk.Constraint] = None
    ) -> LinkageJoint:
        """
        Add a joint connecting two links.

        Args:
            joint_type: Type of joint
            parent_link: Index of parent link
            child_link: Index of child link
            joint_limits: Optional (min, max) limits
            pymunk_joint: Optional pymunk constraint reference

        Returns:
            Created LinkageJoint object
        """
        joint = LinkageJoint(joint_type, parent_link, child_link, joint_limits, pymunk_joint)
        self.joints.append(joint)
        return joint

    def set_link_body(self, link_idx: int, body: pymunk.Body):
        """
        Set the pymunk body for a link.

        Args:
            link_idx: Link index
            body: Pymunk body object
        """
        if 0 <= link_idx < self.num_links:
            self.links[link_idx] = body
        else:
            raise ValueError(f"Link index {link_idx} out of range [0, {self.num_links})")

    def update_joint_states(self):
        """Update all joint states based on current link poses."""
        for joint in self.joints:
            joint.update_state(self.links)

    def get_configuration(self) -> np.ndarray:
        """
        Get current joint configuration.

        Returns:
            Array of joint values [q1, q2, ..., qn]
        """
        return np.array([joint.joint_value for joint in self.joints])

    def get_link_pose(self, link_idx: int) -> np.ndarray:
        """
        Get pose of a specific link.

        Args:
            link_idx: Link index

        Returns:
            Pose as [x, y, theta]
        """
        if self.links[link_idx] is None:
            raise ValueError(f"Link {link_idx} body not set")

        body = self.links[link_idx]
        return np.array([body.position.x, body.position.y, body.angle])

    def get_all_link_poses(self) -> np.ndarray:
        """
        Get poses of all links.

        Returns:
            Array of shape (num_links, 3) containing [x, y, theta] for each link
        """
        poses = []
        for i in range(self.num_links):
            poses.append(self.get_link_pose(i))
        return np.array(poses)

    def get_joint_by_index(self, joint_idx: int) -> LinkageJoint:
        """Get joint by index."""
        if 0 <= joint_idx < len(self.joints):
            return self.joints[joint_idx]
        else:
            raise ValueError(f"Joint index {joint_idx} out of range")

    def get_joints_by_type(self, joint_type: JointType) -> List[LinkageJoint]:
        """Get all joints of a specific type."""
        return [j for j in self.joints if j.joint_type == joint_type]

    def forward_kinematics(
        self,
        base_pose: np.ndarray,
        joint_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute forward kinematics given base pose and joint values.

        Args:
            base_pose: Base link pose [x, y, theta]
            joint_values: Array of joint values

        Returns:
            Array of shape (num_links, 3) with poses of all links
        """
        if len(joint_values) != len(self.joints):
            raise ValueError(f"Expected {len(self.joints)} joint values, got {len(joint_values)}")

        poses = np.zeros((self.num_links, 3))
        poses[0] = base_pose

        # Compute poses recursively through kinematic chain
        for joint_idx, joint in enumerate(self.joints):
            parent_idx = joint.parent_link
            child_idx = joint.child_link
            parent_pose = poses[parent_idx]

            if joint.joint_type == JointType.REVOLUTE:
                # Child link rotates relative to parent
                link_length = self.link_lengths[parent_idx]
                angle = parent_pose[2] + joint_values[joint_idx]

                # Position of child link
                child_x = parent_pose[0] + link_length * np.cos(parent_pose[2])
                child_y = parent_pose[1] + link_length * np.sin(parent_pose[2])

                poses[child_idx] = [child_x, child_y, angle]

            elif joint.joint_type == JointType.PRISMATIC:
                # Child link slides along parent's x-axis
                offset = joint_values[joint_idx]
                child_x = parent_pose[0] + offset * np.cos(parent_pose[2])
                child_y = parent_pose[1] + offset * np.sin(parent_pose[2])

                poses[child_idx] = [child_x, child_y, parent_pose[2]]

            elif joint.joint_type == JointType.FIXED:
                # Child link has fixed offset from parent
                link_length = self.link_lengths[parent_idx]
                child_x = parent_pose[0] + link_length * np.cos(parent_pose[2])
                child_y = parent_pose[1] + link_length * np.sin(parent_pose[2])

                poses[child_idx] = [child_x, child_y, parent_pose[2]]

        return poses

    def compute_object_centroid(self) -> np.ndarray:
        """
        Compute centroid of the entire linkage object.

        Returns:
            Centroid position [x, y]
        """
        positions = []
        total_mass = 0.0

        for i in range(self.num_links):
            if self.links[i] is not None:
                mass = self.link_masses[i]
                pos = self.links[i].position
                positions.append(np.array([pos.x, pos.y]) * mass)
                total_mass += mass

        if total_mass > 0:
            return np.sum(positions, axis=0) / total_mass
        else:
            return np.zeros(2)

    def compute_grasp_poses(
        self,
        num_grippers: int = 2
    ) -> List[np.ndarray]:
        """
        Compute suggested grasp poses for grippers.

        For a two-link object, suggests grasping at the center of each link
        perpendicular to the link orientation.

        Args:
            num_grippers: Number of grippers (default: 2)

        Returns:
            List of grasp poses [x, y, theta] for each gripper
        """
        grasp_poses = []

        # Distribute grippers across links
        for i in range(min(num_grippers, self.num_links)):
            if self.links[i] is not None:
                link = self.links[i]
                # Grasp at link center, perpendicular to link
                grasp_pose = np.array([
                    link.position.x,
                    link.position.y,
                    link.angle + np.pi / 2  # Perpendicular
                ])
                grasp_poses.append(grasp_pose)

        return grasp_poses

    def get_state_dict(self) -> dict:
        """
        Get complete state as dictionary.

        Returns:
            Dictionary containing:
                - link_poses: All link poses
                - joint_values: All joint values
                - joint_velocities: All joint velocities
                - centroid: Object centroid
        """
        self.update_joint_states()

        return {
            "link_poses": self.get_all_link_poses(),
            "joint_values": self.get_configuration(),
            "joint_velocities": np.array([j.joint_velocity for j in self.joints]),
            "centroid": self.compute_object_centroid(),
        }

    def __repr__(self) -> str:
        """String representation."""
        joint_types = [j.joint_type.value for j in self.joints]
        return f"LinkageObject(num_links={self.num_links}, joints={joint_types})"


def create_two_link_object(joint_type: JointType) -> LinkageObject:
    """
    Helper function to create a two-link object.

    Args:
        joint_type: Type of joint connecting the two links

    Returns:
        LinkageObject with two links
    """
    linkage = LinkageObject(num_links=2)

    # Set default link properties
    linkage.link_lengths = [40.0, 40.0]
    linkage.link_masses = [0.5, 0.5]

    # Add joint connecting link 0 (parent) to link 1 (child)
    if joint_type == JointType.REVOLUTE:
        limits = (-np.pi, np.pi)
    elif joint_type == JointType.PRISMATIC:
        limits = (-50.0, 50.0)
    else:  # FIXED
        limits = (0.0, 0.0)

    linkage.add_joint(
        joint_type=joint_type,
        parent_link=0,
        child_link=1,
        joint_limits=limits
    )

    return linkage
