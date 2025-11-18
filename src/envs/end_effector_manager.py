"""
End-Effector Manager

Manages parallel gripper end-effectors including:
- Physical properties (mass, dimensions)
- 1-DOF parallel gripper mechanics
- Constant grasping force application
- External wrench sensing in body frame

This manager is used by BiArt environment for EE-related operations.
"""

import numpy as np
import pymunk
from pymunk import Vec2d
from typing import Tuple, Optional
from dataclasses import dataclass
import pygame


@dataclass
class GripperConfig:
    """Configuration for parallel gripper."""
    base_mass: float = 0.8
    jaw_mass: float = 0.1
    base_width: float = 16.0
    base_height: float = 8.0
    jaw_length: float = 20.0
    jaw_thickness: float = 4.0
    max_opening: float = 20.0
    grip_force: float = 15.0


class ParallelGripper:
    """
    1-DOF parallel gripper with constant closing force.

    Components:
    - Base body (main EE body, control target)
    - Left jaw (prismatic joint)
    - Right jaw (prismatic joint)
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        angle: float,
        name: str,
        base_mass: float = 0.8,
        jaw_mass: float = 0.1,
        base_width: float = 16.0,
        base_height: float = 8.0,
        jaw_length: float = 20.0,
        jaw_thickness: float = 4.0,
        max_opening: float = 20.0,
        grip_force: float = 15.0
    ):
        """
        Initialize parallel gripper.

        Args:
            space: Pymunk space
            position: Initial position (x, y)
            angle: Initial angle (radians)
            name: Gripper name ("left" or "right")
            base_mass: Mass of base body
            jaw_mass: Mass of each jaw
            base_width: Width of base
            base_height: Height of base
            jaw_length: Length of each jaw
            jaw_thickness: Thickness of jaws
            max_opening: Maximum opening between jaws
            grip_force: Constant closing force (Newtons)
        """
        self.space = space
        self.name = name
        self.grip_force = grip_force

        # Store dimensions
        self.base_width = base_width
        self.base_height = base_height
        self.jaw_length = jaw_length
        self.jaw_thickness = jaw_thickness
        self.max_opening = max_opening

        # Color
        color = pygame.Color("RoyalBlue") if name == "left" else pygame.Color("Crimson")

        # Create base body
        base_verts = [
            (-base_width/2, -base_height/2),
            (base_width/2, -base_height/2),
            (base_width/2, base_height/2),
            (-base_width/2, base_height/2),
        ]
        base_inertia = pymunk.moment_for_poly(base_mass, base_verts)
        self.base_body = pymunk.Body(base_mass, base_inertia, body_type=pymunk.Body.DYNAMIC)
        self.base_body.position = position
        self.base_body.angle = angle
        self.base_body.name = name

        base_shape = pymunk.Poly(self.base_body, base_verts)
        base_shape.color = color
        base_shape.friction = 0.5
        base_shape.filter = pymunk.ShapeFilter(categories=0b01)
        base_shape.collision_type = 1  # Gripper collision type

        space.add(self.base_body, base_shape)

        # Create left jaw
        left_jaw_verts = [
            (-jaw_thickness/2, 0),
            (jaw_thickness/2, 0),
            (jaw_thickness/2, jaw_length),
            (-jaw_thickness/2, jaw_length),
        ]
        left_jaw_inertia = pymunk.moment_for_poly(jaw_mass, left_jaw_verts)
        self.left_jaw = pymunk.Body(jaw_mass, left_jaw_inertia, body_type=pymunk.Body.DYNAMIC)
        self.left_jaw.position = self.base_body.local_to_world((-max_opening/4, base_height/2))
        self.left_jaw.angle = angle

        left_jaw_shape = pymunk.Poly(self.left_jaw, left_jaw_verts)
        left_jaw_shape.color = color
        left_jaw_shape.friction = 1.0
        left_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        left_jaw_shape.collision_type = 1

        space.add(self.left_jaw, left_jaw_shape)

        # Create right jaw
        right_jaw_verts = left_jaw_verts
        right_jaw_inertia = pymunk.moment_for_poly(jaw_mass, right_jaw_verts)
        self.right_jaw = pymunk.Body(jaw_mass, right_jaw_inertia, body_type=pymunk.Body.DYNAMIC)
        self.right_jaw.position = self.base_body.local_to_world((max_opening/4, base_height/2))
        self.right_jaw.angle = angle

        right_jaw_shape = pymunk.Poly(self.right_jaw, right_jaw_verts)
        right_jaw_shape.color = color
        right_jaw_shape.friction = 1.0
        right_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        right_jaw_shape.collision_type = 1

        space.add(self.right_jaw, right_jaw_shape)

        # Create prismatic joints (GrooveJoint)
        groove_start = Vec2d(-max_opening/2, base_height/2)
        groove_end = Vec2d(max_opening/2, base_height/2)

        left_anchor = Vec2d(-jaw_thickness/2, 0)
        self.left_joint = pymunk.GrooveJoint(
            self.base_body, self.left_jaw,
            groove_start, Vec2d(0, base_height/2),
            left_anchor
        )
        space.add(self.left_joint)

        right_anchor = Vec2d(jaw_thickness/2, 0)
        self.right_joint = pymunk.GrooveJoint(
            self.base_body, self.right_jaw,
            Vec2d(0, base_height/2), groove_end,
            right_anchor
        )
        space.add(self.right_joint)

        # Rotation constraint for jaws
        left_rotation = pymunk.GearJoint(self.base_body, self.left_jaw, 0, 1)
        left_rotation.max_force = 1e10
        space.add(left_rotation)

        right_rotation = pymunk.GearJoint(self.base_body, self.right_jaw, 0, 1)
        right_rotation.max_force = 1e10
        space.add(right_rotation)

        # External wrench tracking
        self.external_wrench = np.zeros(3)  # [fx, fy, tau] in body frame

    def apply_grip_force(self):
        """Apply constant closing force to both jaws."""
        # Closing direction: inward (toward center)
        left_close_dir = Vec2d(1, 0).rotated(self.base_body.angle)
        right_close_dir = Vec2d(-1, 0).rotated(self.base_body.angle)

        self.left_jaw.apply_force_at_world_point(
            left_close_dir * self.grip_force,
            self.left_jaw.position
        )
        self.right_jaw.apply_force_at_world_point(
            right_close_dir * self.grip_force,
            self.right_jaw.position
        )

    def get_pose(self) -> np.ndarray:
        """Get gripper pose [x, y, theta]."""
        return np.array([
            self.base_body.position.x,
            self.base_body.position.y,
            self.base_body.angle
        ])

    def get_velocity(self) -> np.ndarray:
        """Get gripper velocity [vx, vy, omega]."""
        return np.array([
            self.base_body.velocity.x,
            self.base_body.velocity.y,
            self.base_body.angular_velocity
        ])

    def set_pose(self, pose: np.ndarray):
        """Set gripper pose [x, y, theta]."""
        self.base_body.position = (pose[0], pose[1])
        self.base_body.angle = pose[2]

    def set_velocity(self, velocity: np.ndarray):
        """Set gripper velocity [vx, vy, omega]."""
        self.base_body.velocity = (velocity[0], velocity[1])
        self.base_body.angular_velocity = velocity[2]

    def apply_wrench(self, wrench: np.ndarray):
        """
        Apply wrench command (in body frame).

        Args:
            wrench: [fx, fy, tau] in body frame
        """
        # Transform force from body frame to world frame
        fx_body, fy_body, tau = wrench

        cos_theta = np.cos(self.base_body.angle)
        sin_theta = np.sin(self.base_body.angle)

        fx_world = cos_theta * fx_body - sin_theta * fy_body
        fy_world = sin_theta * fx_body + cos_theta * fy_body

        # Apply to base body
        self.base_body.apply_force_at_world_point(
            (fx_world, fy_world),
            self.base_body.position
        )
        self.base_body.torque += tau

    def compute_external_wrench(self, link_body: Optional[pymunk.Body] = None) -> np.ndarray:
        """
        Compute external wrench from contact forces.

        This is simplified - proper implementation would use collision callbacks.

        Args:
            link_body: Optional link body that gripper is grasping

        Returns:
            External wrench [fx, fy, tau] in body frame
        """
        # Simplified: return zero for now
        # Proper implementation would accumulate forces from contact callbacks
        return np.zeros(3)

    def get_external_wrench(self) -> np.ndarray:
        """Get most recent external wrench measurement."""
        return self.external_wrench.copy()


class EndEffectorManager:
    """
    Manager for all end-effectors in the environment.

    Handles:
    - Creation and initialization of parallel grippers
    - Gripper force application
    - External wrench sensing
    - State queries
    """

    def __init__(
        self,
        space: pymunk.Space,
        num_grippers: int = 2,
        config: Optional[GripperConfig] = None
    ):
        """
        Initialize end-effector manager.

        Args:
            space: Pymunk space
            num_grippers: Number of grippers
            config: Optional gripper configuration
        """
        self.space = space
        self.num_grippers = num_grippers

        # Use provided config or default
        if config is None:
            config = GripperConfig()

        # Store config as dict for backward compatibility
        self.gripper_params = {
            'base_mass': config.base_mass,
            'jaw_mass': config.jaw_mass,
            'base_width': config.base_width,
            'base_height': config.base_height,
            'jaw_length': config.jaw_length,
            'jaw_thickness': config.jaw_thickness,
            'max_opening': config.max_opening,
            'grip_force': config.grip_force
        }

        # Grippers will be created during reset
        self.grippers = []

    def reset(self, initial_poses: np.ndarray):
        """
        Reset end-effectors.

        Args:
            initial_poses: Array of shape (num_grippers, 3) with initial poses
        """
        # Remove old grippers if they exist
        for gripper in self.grippers:
            self.space.remove(gripper.base_body)
            self.space.remove(gripper.left_jaw)
            self.space.remove(gripper.right_jaw)
            self.space.remove(gripper.left_joint)
            self.space.remove(gripper.right_joint)

        self.grippers = []

        # Create new grippers
        names = ["left", "right"]
        for i in range(self.num_grippers):
            gripper = ParallelGripper(
                space=self.space,
                position=(initial_poses[i, 0], initial_poses[i, 1]),
                angle=initial_poses[i, 2],
                name=names[i] if i < len(names) else f"gripper_{i}",
                **self.gripper_params
            )
            self.grippers.append(gripper)

    def step(self):
        """
        Step end-effector physics.

        Applies grip forces to all grippers.
        """
        for gripper in self.grippers:
            gripper.apply_grip_force()

    def get_poses(self) -> np.ndarray:
        """Get all gripper poses."""
        return np.array([g.get_pose() for g in self.grippers])

    def get_velocities(self) -> np.ndarray:
        """Get all gripper velocities."""
        return np.array([g.get_velocity() for g in self.grippers])

    def apply_wrenches(self, wrenches: np.ndarray):
        """
        Apply wrench commands to all grippers.

        Args:
            wrenches: Array of shape (num_grippers, 3) with wrenches in body frame
        """
        for i, gripper in enumerate(self.grippers):
            gripper.apply_wrench(wrenches[i])

    def get_external_wrenches(self) -> np.ndarray:
        """Get external wrenches for all grippers."""
        return np.array([g.get_external_wrench() for g in self.grippers])

    def get_gripper(self, idx: int) -> ParallelGripper:
        """Get gripper by index."""
        return self.grippers[idx]
