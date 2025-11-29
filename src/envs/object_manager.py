"""
Object Manager

Manages articulated objects for bimanual manipulation:
- Two-link objects with 1-DOF joints (revolute, prismatic, fixed)
- Pymunk physics constraints
- Grasping frames for gripper attachment
- Joint axis screws for screw-decomposed control

Joint Types:
- Revolute: rotation around connection point (PivotJoint)
- Prismatic: sliding along link axis (GrooveJoint)
- Fixed: rigid connection (PivotJoint + DampedRotarySpring)
"""

import numpy as np
import pymunk
from pymunk import Vec2d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pygame

from src.se2_math import SE2Pose, normalize_angle, se2_inverse


class JointType(Enum):
    """Joint types for articulated objects."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


@dataclass
class ObjectConfig:
    """Configuration for articulated object."""
    link_length: float = 40.0
    link_width: float = 11.0
    link_mass: float = 0.1


@dataclass
class GraspingFrame:
    """
    Grasping frame for gripper attachment.
    
    Defined in link's local coordinate frame.
    Gripper's body frame aligns with this frame when grasping.
    """
    link_id: int  # 0 for link1, 1 for link2
    local_pose: np.ndarray  # [x, y, theta] in link frame
    gripper_name: str  # "left" or "right"


class ArticulatedObject:
    """
    Two-link articulated object with 1-DOF joint.
    
    Link coordinate system:
    - Origin at link center
    - X-axis along link length
    - Y-axis perpendicular to link
    """

    def __init__(
        self,
        space: pymunk.Space,
        joint_type: JointType,
        config: ObjectConfig,
        initial_pose: np.ndarray
    ):
        """
        Initialize articulated object.

        Args:
            space: Pymunk space
            joint_type: Type of joint connecting links
            config: Object configuration
            initial_pose: Initial pose of link1 center [x, y, theta]
        """
        self.space = space
        self.joint_type = joint_type
        self.config = config
        
        # Create links and joints
        self.link1, self.link2, self.primary_joint, self.aux_joints = \
            self._create_links_and_joint(initial_pose)
        
        # Define grasping frames
        self.grasping_frames = self._define_grasping_frames()

    def _create_link(self, position: Tuple[float, float], angle: float, name: str) -> pymunk.Body:
        """Create a single link body with shape."""
        cfg = self.config
        
        # Link vertices (centered at origin)
        verts = [
            (-cfg.link_length/2, -cfg.link_width/2),
            (cfg.link_length/2, -cfg.link_width/2),
            (cfg.link_length/2, cfg.link_width/2),
            (-cfg.link_length/2, cfg.link_width/2),
        ]
        
        inertia = pymunk.moment_for_poly(cfg.link_mass, verts)
        body = pymunk.Body(cfg.link_mass, inertia, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle
        body.name = name
        
        shape = pymunk.Poly(body, verts)
        shape.color = pygame.Color("LightSlateGray")
        shape.friction = 100.0
        shape.filter = pymunk.ShapeFilter(categories=0b10)  # Object category
        shape.collision_type = 2  # Object collision type
        
        self.space.add(body, shape)
        return body

    def _create_links_and_joint(self, link1_pose: np.ndarray):
        """Create two links connected by joint."""
        cfg = self.config
        
        # Create link 1
        link1 = self._create_link(
            (link1_pose[0], link1_pose[1]),
            link1_pose[2],
            "link1"
        )
        
        # Connection point (link1's right end)
        connection_local = Vec2d(cfg.link_length / 2, 0)
        connection_world = link1.local_to_world(connection_local)
        
        # Determine link2 initial configuration based on joint type
        if self.joint_type == JointType.REVOLUTE:
            # Start with -90° relative angle (V-shape)
            relative_angle = -np.pi / 2
            link2_angle = link1_pose[2] + relative_angle
            link2_offset = Vec2d(cfg.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_world + link2_offset
            
        elif self.joint_type == JointType.PRISMATIC:
            # Links stay aligned, start at center of sliding range
            link2_angle = link1_pose[2]
            # Link2 left end at link1 center (joint state = 0)
            link2_center = link1.local_to_world(Vec2d(cfg.link_length / 2, 0))
            
        else:  # FIXED
            link2_angle = link1_pose[2]
            link2_offset = Vec2d(cfg.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_world + link2_offset
        
        # Create link 2
        link2 = self._create_link(link2_center, link2_angle, "link2")
        
        # Create joint constraints
        aux_joints = []
        
        if self.joint_type == JointType.REVOLUTE:
            # PivotJoint allows rotation around connection point
            primary_joint = pymunk.PivotJoint(link1, link2, connection_world)
            primary_joint.collide_bodies = False
            
            # Rotary limits: -180° to 180°
            limit = pymunk.RotaryLimitJoint(link1, link2, -np.pi, np.pi)
            limit.collide_bodies = False
            self.space.add(limit)
            aux_joints.append(limit)
            
        elif self.joint_type == JointType.PRISMATIC:
            # GrooveJoint allows sliding along link1's x-axis
            groove_start = Vec2d(-cfg.link_length / 2, 0)
            groove_end = Vec2d(cfg.link_length, 0)
            anchor = Vec2d(-cfg.link_length / 2, 0)  # Link2's left end
            
            primary_joint = pymunk.GrooveJoint(
                link1, link2,
                groove_start, groove_end,
                anchor
            )
            primary_joint.collide_bodies = False
            
            # GearJoint prevents rotation (keeps links parallel)
            gear = pymunk.GearJoint(link1, link2, 0, 1)
            gear.max_force = 1e10
            gear.collide_bodies = False
            self.space.add(gear)
            aux_joints.append(gear)
            
        else:  # FIXED
            # Rigid connection
            primary_joint = pymunk.PivotJoint(link1, link2, connection_world)
            primary_joint.collide_bodies = False
            
            # Very stiff rotational spring
            spring = pymunk.DampedRotarySpring(link1, link2, 0, 1e8, 1e6)
            self.space.add(spring)
            aux_joints.append(spring)
        
        self.space.add(primary_joint)
        return link1, link2, primary_joint, aux_joints

    def _define_grasping_frames(self) -> Dict[str, GraspingFrame]:
        """
        Define grasping frames for each gripper.
        
        Left gripper grasps link1's left side (perpendicular approach)
        Right gripper grasps link2's right side (perpendicular approach)
        
        The pose is the Gripper Base Body's pose.
        We need to apply an offset so that the Jaw Center aligns with the target grasp point on the link.
        
        Gripper Geometry:
        - Base height: 8.0
        - Jaw length: 20.0
        - Jaw center distance from base center: 4.0 + 10.0 = 14.0
        
        Offset direction is opposite to the Jaw direction (Gripper Y-axis).
        """
        cfg = self.config
        
        # Distance from Base Center to Jaw Center
        # Should be consistent with EndEffectorManager's GripperConfig
        jaw_center_offset = 14.0
        
        return {
            "left": GraspingFrame(
                link_id=0,
                # Link1 local: (-L/4, 0) is grasp point.
                # Left Gripper: theta = -pi/2. 
                # Gripper Y-axis (Jaw dir) aligns with Link X-axis.
                # Base must be shifted by -14.0 along Link X-axis.
                local_pose=np.array([-cfg.link_length / 4 - jaw_center_offset, 0.0, -np.pi / 2]),
                gripper_name="left"
            ),
            "right": GraspingFrame(
                link_id=1,
                # Link2 local: (L/4, 0) is grasp point.
                # Right Gripper: theta = pi/2.
                # Gripper Y-axis (Jaw dir) aligns with Link -X-axis.
                # Base must be shifted by +14.0 along Link X-axis (opposite to -X).
                local_pose=np.array([cfg.link_length / 4 + jaw_center_offset, 0.0, np.pi / 2]),
                gripper_name="right"
            )
        }

    def remove_from_space(self):
        """Remove all pymunk entities."""
        components = []
        components.extend(list(self.link1.shapes))
        components.extend(list(self.link2.shapes))
        components.extend(self.aux_joints)
        components.append(self.primary_joint)
        components.append(self.link1)
        components.append(self.link2)
        self.space.remove(*components)

    def get_link_poses(self) -> np.ndarray:
        """Get poses of both links (2, 3)."""
        return np.array([
            [self.link1.position.x, self.link1.position.y, self.link1.angle],
            [self.link2.position.x, self.link2.position.y, self.link2.angle]
        ])

    def get_link_velocities(self) -> np.ndarray:
        """Get velocities of both links (2, 3)."""
        return np.array([
            [self.link1.velocity.x, self.link1.velocity.y, self.link1.angular_velocity],
            [self.link2.velocity.x, self.link2.velocity.y, self.link2.angular_velocity]
        ])

    def get_grasping_pose(self, gripper_name: str) -> np.ndarray:
        """Get world-frame pose of grasping frame."""
        frame = self.grasping_frames[gripper_name]
        link_pose = self.get_link_poses()[frame.link_id]
        
        # Transform local grasp pose to world frame
        T_link = SE2Pose.from_array(link_pose).to_matrix()
        T_grasp_local = SE2Pose.from_array(frame.local_pose).to_matrix()
        T_grasp_world = T_link @ T_grasp_local
        
        return SE2Pose.from_matrix(T_grasp_world).to_array()

    def get_all_grasping_poses(self) -> Dict[str, np.ndarray]:
        """Get all grasping frame poses in world frame."""
        return {name: self.get_grasping_pose(name) for name in self.grasping_frames}

    def get_joint_state(self) -> float:
        """
        Get current joint state.
        
        Revolute: relative angle (radians)
        Prismatic: relative position (pixels)
        Fixed: 0
        """
        if self.joint_type == JointType.REVOLUTE:
            return normalize_angle(self.link2.angle - self.link1.angle)
            
        elif self.joint_type == JointType.PRISMATIC:
            # Position of link2's left end in link1's frame
            link2_left = Vec2d(-self.config.link_length / 2, 0)
            link2_left_world = self.link2.local_to_world(link2_left)
            pos_in_link1 = self.link1.world_to_local(link2_left_world)
            return pos_in_link1.x
            
        return 0.0

    def get_joint_axis_screws(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint axis as SE(2) unit screws in each grasping frame.
        
        Configuration-invariant kinematic constraint information.
        
        Revolute: B = [1, r_y, -r_x] where r is position to joint center
        Prismatic: B = [0, v_x, v_y] where v is unit sliding direction
        Fixed: B = [0, 0, 0]
        
        Returns:
            (B_left, B_right) - screws in left and right grasping frames
        """
        if self.joint_type == JointType.FIXED:
            return np.zeros(3), np.zeros(3)
        
        cfg = self.config
        
        if self.joint_type == JointType.REVOLUTE:
            # Joint at link1's right end (in link1 frame)
            joint_pos_link1 = np.array([cfg.link_length / 2, 0.0])
            # Joint at link2's left end (in link2 frame)
            joint_pos_link2 = np.array([-cfg.link_length / 2, 0.0])
            
            B_left = self._compute_revolute_screw("left", joint_pos_link1)
            B_right = self._compute_revolute_screw("right", joint_pos_link2)
            
        else:  # PRISMATIC
            # Sliding direction along x-axis in link frame
            joint_dir = np.array([1.0, 0.0])
            
            B_left = self._compute_prismatic_screw("left", joint_dir)
            B_right = self._compute_prismatic_screw("right", joint_dir)
        
        return B_left, B_right

    def _compute_revolute_screw(self, gripper_name: str, joint_pos_link: np.ndarray) -> np.ndarray:
        """Compute revolute joint screw in grasping frame."""
        frame = self.grasping_frames[gripper_name]
        
        # Transform from link frame to grasp frame
        T_link_grasp = SE2Pose.from_array(frame.local_pose).to_matrix()
        T_grasp_link = se2_inverse(T_link_grasp)
        
        R = T_grasp_link[:2, :2]
        t = T_grasp_link[:2, 2]
        
        # Joint position in grasp frame
        joint_pos_grasp = R @ joint_pos_link + t
        rx, ry = joint_pos_grasp
        
        # Screw: [omega=1, ry, -rx] (twist from unit angular velocity)
        return np.array([1.0, ry, -rx])

    def _compute_prismatic_screw(self, gripper_name: str, joint_dir_link: np.ndarray) -> np.ndarray:
        """Compute prismatic joint screw in grasping frame."""
        frame = self.grasping_frames[gripper_name]
        
        T_link_grasp = SE2Pose.from_array(frame.local_pose).to_matrix()
        T_grasp_link = se2_inverse(T_link_grasp)
        
        R = T_grasp_link[:2, :2]
        
        # Transform direction (rotation only)
        joint_dir_grasp = R @ joint_dir_link
        joint_dir_grasp /= np.linalg.norm(joint_dir_grasp)
        
        # Screw: [omega=0, vx, vy] (twist from unit linear velocity)
        return np.array([0.0, joint_dir_grasp[0], joint_dir_grasp[1]])


class ObjectManager:
    """
    Manager for articulated objects.
    
    Handles object lifecycle, state queries, and kinematic information.
    """

    def __init__(
        self,
        space: pymunk.Space,
        joint_type: str = "revolute",
        config: Optional[ObjectConfig] = None
    ):
        """
        Initialize object manager.

        Args:
            space: Pymunk space
            joint_type: "revolute", "prismatic", or "fixed"
            config: Object configuration
        """
        self.space = space
        self.joint_type = JointType(joint_type)
        self.config = config or ObjectConfig()
        self.object: Optional[ArticulatedObject] = None

    def reset(self, initial_pose: np.ndarray):
        """Reset object at specified pose."""
        if self.object is not None:
            self.object.remove_from_space()
        
        self.object = ArticulatedObject(
            space=self.space,
            joint_type=self.joint_type,
            config=self.config,
            initial_pose=initial_pose
        )

    def get_link_poses(self) -> np.ndarray:
        """Get link poses (2, 3)."""
        return self.object.get_link_poses()

    def get_link_velocities(self) -> np.ndarray:
        """Get link velocities (2, 3)."""
        return self.object.get_link_velocities()

    def get_grasping_poses(self) -> Dict[str, np.ndarray]:
        """Get grasping frame poses in world frame."""
        return self.object.get_all_grasping_poses()

    def get_joint_state(self) -> float:
        """Get joint state."""
        return self.object.get_joint_state()

    def get_joint_axis_screws(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint axis screws in grasping frames."""
        return self.object.get_joint_axis_screws()
