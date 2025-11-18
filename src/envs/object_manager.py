"""
Object Manager

Manages articulated objects including:
- Multi-link objects with various joint types (revolute, prismatic, fixed)
- Pymunk physics joints (PivotJoint, GrooveJoint, etc.)
- Grasping frames for each link
- Object state queries and updates

This manager is used by BiArt environment for object-related operations.
"""

import numpy as np
import pymunk
from pymunk import Vec2d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pygame

from src.se2_math import SE2Pose, normalize_angle


class JointType(Enum):
    """Joint types for articulated objects."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


@dataclass
class GraspingFrame:
    """
    Grasping frame for a link.

    Defined in link's local coordinate frame.
    EE's body frame should align with this frame when grasping.
    """
    link_id: int
    local_pose: np.ndarray  # [x, y, theta] in link's local frame
    gripper_name: str  # Which gripper grasps at this frame


class ArticulatedObject:
    """
    Multi-link articulated object with joints.

    Currently supports two-link objects with single joint.
    """

    def __init__(
        self,
        space: pymunk.Space,
        joint_type: JointType,
        link_length: float = 40.0,
        link_width: float = 12.0,
        link_mass: float = 0.5,
        initial_pose: Optional[np.ndarray] = None,
        initial_angle: float = 0.0
    ):
        """
        Initialize articulated object.

        Args:
            space: Pymunk space
            joint_type: Type of joint connecting links
            link_length: Length of each link
            link_width: Width of each link
            link_mass: Mass of each link
            initial_pose: Initial pose of link1 center [x, y, theta]
            initial_angle: Initial angle (if initial_pose not provided)
        """
        self.space = space
        self.joint_type = joint_type
        self.link_length = link_length
        self.link_width = link_width
        self.link_mass = link_mass

        # Default initial pose
        if initial_pose is None:
            initial_pose = np.array([256.0, 256.0, initial_angle])

        # Create links and joint
        self.link1, self.link2, self.joint, self.aux_joints = self._create_links_and_joint(
            initial_pose
        )

        # Define grasping frames
        self.grasping_frames = self._define_grasping_frames()

    def _create_link(
        self,
        position: Tuple[float, float],
        angle: float,
        name: str
    ) -> Tuple[pymunk.Body, List[pymunk.Shape]]:
        """
        Create a single link with grasping part.

        Args:
            position: Position (x, y)
            angle: Angle (radians)
            name: Link name

        Returns:
            (body, shapes) tuple
        """
        # Main body vertices
        l, w = self.link_length, self.link_width
        main_verts = [
            (-l/2, -w/2),
            (l/2, -w/2),
            (l/2, w/2),
            (-l/2, w/2),
        ]

        # Calculate moment of inertia
        inertia = pymunk.moment_for_poly(self.link_mass, main_verts)

        # Create body
        body = pymunk.Body(self.link_mass, inertia, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle
        body.name = name

        # Create shape
        shape = pymunk.Poly(body, main_verts)
        shape.color = pygame.Color("LightSlateGray")
        shape.friction = 1.0
        shape.filter = pymunk.ShapeFilter(categories=0b10)  # Object category
        shape.collision_type = 2  # Object collision type

        self.space.add(body, shape)

        return body, [shape]

    def _create_links_and_joint(
        self,
        link1_pose: np.ndarray
    ) -> Tuple[pymunk.Body, pymunk.Body, pymunk.Constraint, List[pymunk.Constraint]]:
        """
        Create two links connected by a joint.

        Args:
            link1_pose: Pose of link1 [x, y, theta]

        Returns:
            (link1_body, link2_body, primary_joint, auxiliary_joints)
        """
        # Create link 1
        link1_body, link1_shapes = self._create_link(
            (link1_pose[0], link1_pose[1]),
            link1_pose[2],
            "link1"
        )

        # Calculate connection point (link1's right end in world frame)
        link1_end_local = Vec2d(self.link_length / 2, 0)
        connection_point = link1_body.local_to_world(link1_end_local)

        # Determine link2 initial configuration
        if self.joint_type == JointType.REVOLUTE:
            # Start aligned, joint will control relative angle
            link2_angle = link1_pose[2]
            link2_start_to_center = Vec2d(self.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_point + link2_start_to_center

        elif self.joint_type == JointType.PRISMATIC:
            # Links stay aligned
            link2_angle = link1_pose[2]
            link2_start_to_center = Vec2d(self.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_point + link2_start_to_center

        else:  # FIXED
            # Rigidly aligned
            link2_angle = link1_pose[2]
            link2_start_to_center = Vec2d(self.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_point + link2_start_to_center

        # Create link 2
        link2_body, link2_shapes = self._create_link(
            link2_center,
            link2_angle,
            "link2"
        )

        # Create joints using Pymunk constraints
        auxiliary_joints = []

        if self.joint_type == JointType.REVOLUTE:
            # PivotJoint: allows rotation around connection point
            primary_joint = pymunk.PivotJoint(link1_body, link2_body, connection_point)
            primary_joint.collide_bodies = False

            # Add rotary limit to enforce V-shape (-120° to -60°)
            angle_limit = pymunk.RotaryLimitJoint(
                link1_body, link2_body,
                -2*np.pi/3,  # -120°
                -np.pi/3     # -60°
            )
            angle_limit.collide_bodies = False
            self.space.add(angle_limit)
            auxiliary_joints.append(angle_limit)

        elif self.joint_type == JointType.PRISMATIC:
            # GrooveJoint: allows sliding along link1's x-axis
            groove_start = Vec2d(0, 0)  # Center of link1 in link1's frame
            groove_end = Vec2d(self.link_length, 0)  # Along link1's x-axis
            anchor = Vec2d(-self.link_length / 2, 0)  # Link2's left end

            primary_joint = pymunk.GrooveJoint(
                link1_body, link2_body,
                groove_start, groove_end,
                anchor
            )
            primary_joint.collide_bodies = False

            # PinJoint to prevent rotation (keeps links parallel)
            pin_joint = pymunk.PinJoint(link1_body, link2_body, connection_point, connection_point)
            pin_joint.distance = 0
            pin_joint.collide_bodies = False
            self.space.add(pin_joint)
            auxiliary_joints.append(pin_joint)

        else:  # FIXED
            # DampedRotarySpring: very stiff rotational constraint
            rotary_spring = pymunk.DampedRotarySpring(link1_body, link2_body, 0, 1e8, 1e6)
            self.space.add(rotary_spring)
            auxiliary_joints.append(rotary_spring)

            # PivotJoint: rigid position constraint
            primary_joint = pymunk.PivotJoint(link1_body, link2_body, connection_point)
            primary_joint.collide_bodies = False

        self.space.add(primary_joint)

        return link1_body, link2_body, primary_joint, auxiliary_joints

    def _define_grasping_frames(self) -> Dict[str, GraspingFrame]:
        """
        Define grasping frames for each link.

        For two-link object:
        - Left gripper grasps link1's left side (offset -length/4)
        - Right gripper grasps link2's right side (offset +length/4)
        - Frame orientation: 90° relative to link (jaws perpendicular to link)

        Returns:
            Dictionary {gripper_name: GraspingFrame}
        """
        frames = {}

        # Left gripper → Link1, left side, perpendicular
        frames["left"] = GraspingFrame(
            link_id=0,
            local_pose=np.array([-self.link_length / 4, 0.0, np.pi / 2]),
            gripper_name="left"
        )

        # Right gripper → Link2, right side, perpendicular
        frames["right"] = GraspingFrame(
            link_id=1,
            local_pose=np.array([self.link_length / 4, 0.0, np.pi / 2]),
            gripper_name="right"
        )

        return frames

    def get_link_poses(self) -> np.ndarray:
        """
        Get poses of all links.

        Returns:
            Array of shape (2, 3) with poses [x, y, theta]
        """
        return np.array([
            [self.link1.position.x, self.link1.position.y, self.link1.angle],
            [self.link2.position.x, self.link2.position.y, self.link2.angle]
        ])

    def get_link_velocities(self) -> np.ndarray:
        """
        Get velocities of all links.

        Returns:
            Array of shape (2, 3) with velocities [vx, vy, omega]
        """
        return np.array([
            [self.link1.velocity.x, self.link1.velocity.y, self.link1.angular_velocity],
            [self.link2.velocity.x, self.link2.velocity.y, self.link2.angular_velocity]
        ])

    def get_grasping_pose(self, gripper_name: str) -> np.ndarray:
        """
        Get world-frame pose of grasping frame for a gripper.

        Args:
            gripper_name: "left" or "right"

        Returns:
            Grasping frame pose [x, y, theta] in world frame
        """
        if gripper_name not in self.grasping_frames:
            raise ValueError(f"Unknown gripper: {gripper_name}")

        frame = self.grasping_frames[gripper_name]

        # Get link pose
        link_poses = self.get_link_poses()
        link_pose = link_poses[frame.link_id]

        # Transform local grasp pose to world frame
        link_T = SE2Pose.from_array(link_pose).to_matrix()
        grasp_local_T = SE2Pose.from_array(frame.local_pose).to_matrix()
        grasp_world_T = link_T @ grasp_local_T

        return SE2Pose.from_matrix(grasp_world_T).to_array()

    def get_all_grasping_poses(self) -> Dict[str, np.ndarray]:
        """
        Get all grasping frame poses.

        Returns:
            Dictionary {gripper_name: world_pose}
        """
        return {
            name: self.get_grasping_pose(name)
            for name in self.grasping_frames.keys()
        }

    def get_joint_state(self) -> float:
        """
        Get current joint state.

        For revolute: relative angle (radians)
        For prismatic: relative position (pixels)
        For fixed: 0

        Returns:
            Joint state value
        """
        if self.joint_type == JointType.REVOLUTE:
            # Relative angle between links
            return normalize_angle(self.link2.angle - self.link1.angle)

        elif self.joint_type == JointType.PRISMATIC:
            # Distance along link1's axis
            # Connection point in link1's frame
            connection_in_link1 = self.link1.world_to_local(self.link2.position)
            return connection_in_link1.x - self.link_length / 2

        else:  # FIXED
            return 0.0

    def set_link_poses(self, poses: np.ndarray):
        """
        Set link poses.

        Args:
            poses: Array of shape (2, 3) with poses [x, y, theta]
        """
        self.link1.position = (poses[0, 0], poses[0, 1])
        self.link1.angle = poses[0, 2]
        self.link2.position = (poses[1, 0], poses[1, 1])
        self.link2.angle = poses[1, 2]

    def set_link_velocities(self, velocities: np.ndarray):
        """
        Set link velocities.

        Args:
            velocities: Array of shape (2, 3) with velocities [vx, vy, omega]
        """
        self.link1.velocity = (velocities[0, 0], velocities[0, 1])
        self.link1.angular_velocity = velocities[0, 2]
        self.link2.velocity = (velocities[1, 0], velocities[1, 1])
        self.link2.angular_velocity = velocities[1, 2]


class ObjectManager:
    """
    Manager for articulated objects in the environment.

    Handles:
    - Object creation and initialization
    - Grasping frame queries
    - Object state queries
    """

    def __init__(
        self,
        space: pymunk.Space,
        joint_type: str = "revolute",
        object_params: Optional[dict] = None
    ):
        """
        Initialize object manager.

        Args:
            space: Pymunk space
            joint_type: Type of joint ("revolute", "prismatic", "fixed")
            object_params: Optional object parameters
        """
        self.space = space
        self.joint_type = JointType(joint_type)

        # Default object parameters
        default_params = {
            'link_length': 40.0,
            'link_width': 12.0,
            'link_mass': 0.5
        }

        self.object_params = {**default_params, **(object_params or {})}

        # Object will be created during reset
        self.object: Optional[ArticulatedObject] = None

    def reset(self, initial_pose: Optional[np.ndarray] = None):
        """
        Reset object.

        Args:
            initial_pose: Optional initial pose of link1 [x, y, theta]
        """
        # Remove old object if exists
        if self.object is not None:
            self.space.remove(self.object.link1)
            self.space.remove(self.object.link2)
            self.space.remove(self.object.joint)
            for aux_joint in self.object.aux_joints:
                self.space.remove(aux_joint)

        # Create new object
        self.object = ArticulatedObject(
            space=self.space,
            joint_type=self.joint_type,
            initial_pose=initial_pose,
            **self.object_params
        )

    def get_grasping_poses(self) -> Dict[str, np.ndarray]:
        """Get all grasping frame poses in world frame."""
        if self.object is None:
            raise RuntimeError("Object not initialized. Call reset() first.")
        return self.object.get_all_grasping_poses()

    def get_link_poses(self) -> np.ndarray:
        """Get link poses."""
        if self.object is None:
            raise RuntimeError("Object not initialized. Call reset() first.")
        return self.object.get_link_poses()

    def get_link_velocities(self) -> np.ndarray:
        """Get link velocities."""
        if self.object is None:
            raise RuntimeError("Object not initialized. Call reset() first.")
        return self.object.get_link_velocities()

    def get_joint_state(self) -> float:
        """Get joint state."""
        if self.object is None:
            raise RuntimeError("Object not initialized. Call reset() first.")
        return self.object.get_joint_state()

    def get_object(self) -> ArticulatedObject:
        """Get the articulated object."""
        if self.object is None:
            raise RuntimeError("Object not initialized. Call reset() first.")
        return self.object
