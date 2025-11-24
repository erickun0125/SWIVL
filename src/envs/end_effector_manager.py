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
from typing import Tuple, Optional, List
from dataclasses import dataclass
import pygame

from src.se2_math import spatial_to_body_twist


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
    target_object_width: float = 12.0


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
        grip_force: float = 15.0,
        target_object_width: float = 12.0
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
        self.target_object_width = target_object_width

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
        self.base_shape = base_shape

        # Create left jaw
        left_jaw_verts = [
            (-jaw_thickness/2, 0),
            (jaw_thickness/2, 0),
            (jaw_thickness/2, jaw_length),
            (-jaw_thickness/2, jaw_length),
        ]
        left_jaw_inertia = pymunk.moment_for_poly(jaw_mass, left_jaw_verts)
        self.left_jaw = pymunk.Body(jaw_mass, left_jaw_inertia, body_type=pymunk.Body.DYNAMIC)
        # Initial opening respects expected object width with safety margin
        link_width = self.target_object_width
        safety_margin = 1.2
        required_opening = (link_width * safety_margin) / 2.0
        default_opening = max_opening * 0.4
        initial_half_opening = min(max_opening * 0.5, max(required_opening, default_opening))
        self.left_jaw.position = self.base_body.local_to_world((-initial_half_opening, base_height/2))
        self.left_jaw.angle = angle

        left_jaw_shape = pymunk.Poly(self.left_jaw, left_jaw_verts)
        left_jaw_shape.color = color
        left_jaw_shape.friction = 1.0
        left_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        left_jaw_shape.collision_type = 1

        space.add(self.left_jaw, left_jaw_shape)
        self.left_jaw_shape = left_jaw_shape

        # Create right jaw
        right_jaw_verts = left_jaw_verts
        right_jaw_inertia = pymunk.moment_for_poly(jaw_mass, right_jaw_verts)
        self.right_jaw = pymunk.Body(jaw_mass, right_jaw_inertia, body_type=pymunk.Body.DYNAMIC)
        self.right_jaw.position = self.base_body.local_to_world((initial_half_opening, base_height/2))
        self.right_jaw.angle = angle

        right_jaw_shape = pymunk.Poly(self.right_jaw, right_jaw_verts)
        right_jaw_shape.color = color
        right_jaw_shape.friction = 1.0
        right_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        right_jaw_shape.collision_type = 1

        space.add(self.right_jaw, right_jaw_shape)
        self.right_jaw_shape = right_jaw_shape

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
        self.left_rotation = pymunk.GearJoint(self.base_body, self.left_jaw, 0, 1)
        self.left_rotation.max_force = 1e10
        space.add(self.left_rotation)

        self.right_rotation = pymunk.GearJoint(self.base_body, self.right_jaw, 0, 1)
        self.right_rotation.max_force = 1e10
        space.add(self.right_rotation)

        # External wrench tracking
        # Following Modern Robotics convention: [tau, fx, fy]
        self.external_wrench = np.zeros(3)  # [tau, fx, fy] in body frame (MR convention!)
        self.contact_impulses = []  # Store contact impulses during step
        self.space = space

    def remove_from_space(self):
        """Remove all pymunk entities associated with this gripper."""
        components = [
            self.base_shape,
            self.left_jaw_shape,
            self.right_jaw_shape,
            self.base_body,
            self.left_jaw,
            self.right_jaw,
            self.left_joint,
            self.right_joint,
            self.left_rotation,
            self.right_rotation,
        ]
        # space.remove accepts duplicates gracefully; filter None just in case
        self.space.remove(*[c for c in components if c is not None])
        self.contact_impulses = []

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

        Following Modern Robotics convention:
        Args:
            wrench: [tau, fx, fy] in body frame (MR convention: torque first!)
        """
        # Parse wrench (MR convention: [tau, fx, fy])
        tau, fx_body, fy_body = wrench

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

    def add_contact_impulse(self, impulse: Vec2d, contact_point: Vec2d, contact_body=None):
        """
        Add a contact impulse from collision callback.

        Args:
            impulse: Contact impulse in world frame
            contact_point: Contact point in world frame
            contact_body: Pymunk body where contact occurred (base_body, left_jaw, or right_jaw)
                         If None, assumes base_body
        """
        if contact_body is None:
            contact_body = self.base_body
        self.contact_impulses.append((impulse, contact_point, contact_body))

    def compute_external_wrench(self, dt: float) -> np.ndarray:
        """
        Compute external wrench from accumulated contact impulses.

        Improved to handle jaw contacts properly by computing torque contribution
        from jaw contact points correctly.

        Args:
            dt: Physics timestep (to convert impulses to forces)

        Returns:
            External wrench [tau, fx, fy] in body frame (MR convention!)
        """
        if not self.contact_impulses:
            return np.zeros(3)

        # Accumulate forces and torques in world frame
        total_force_world = Vec2d(0, 0)
        total_torque = 0.0

        for impulse, contact_point, contact_body in self.contact_impulses:
            # Convert impulse to force: F = impulse / dt
            force = impulse / dt
            total_force_world += force

            # Compute torque about base body center
            # Note: Even if contact is on jaw, we compute torque w.r.t. base center
            # because jaw forces are transmitted to base through prismatic joint
            r = contact_point - self.base_body.position
            # Torque = r × F (cross product in 2D)
            torque = r.x * force.y - r.y * force.x
            total_torque += torque

        # Transform force from world frame to body frame
        cos_theta = np.cos(self.base_body.angle)
        sin_theta = np.sin(self.base_body.angle)

        fx_body = cos_theta * total_force_world.x + sin_theta * total_force_world.y
        fy_body = -sin_theta * total_force_world.x + cos_theta * total_force_world.y

        # Store and return
        # Following Modern Robotics convention: [τ, fx, fy] (torque first!)
        self.external_wrench = np.array([total_torque, fx_body, fy_body])

        # Clear impulses for next step
        self.contact_impulses = []

        return self.external_wrench.copy()

    def clear_contact_impulses(self):
        """Clear accumulated contact impulses."""
        self.contact_impulses = []

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
        config: Optional[GripperConfig] = None,
        dt: float = 0.01
    ):
        """
        Initialize end-effector manager.

        Args:
            space: Pymunk space
            num_grippers: Number of grippers
            config: Optional gripper configuration
            dt: Physics timestep for force computation
        """
        self.space = space
        self.num_grippers = num_grippers
        self.dt = dt

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
            'grip_force': config.grip_force,
            'target_object_width': config.target_object_width
        }

        # Grippers will be created during reset
        self.grippers = []

        # Setup collision handlers for force sensing
        self._setup_collision_handlers()

    def reset(self, initial_poses: np.ndarray):
        """
        Reset end-effectors.

        Args:
            initial_poses: Array of shape (num_grippers, 3) with initial poses
        """
        # Remove old grippers if they exist
        for gripper in self.grippers:
            gripper.remove_from_space()

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

    def _setup_collision_handlers(self):
        """Setup collision handlers for force sensing."""
        # Collision type 1 is for grippers
        # We'll handle collisions between grippers and objects (collision type 2)
        handler = self.space.add_collision_handler(1, 2)
        handler.post_solve = self._handle_collision

    def _handle_collision(self, arbiter, space, data):
        """
        Collision callback to record contact forces.

        Improved to handle jaw contacts properly by identifying which body
        (base, left_jaw, or right_jaw) the contact occurred on.

        Args:
            arbiter: Collision arbiter containing contact information
            space: Pymunk space
            data: User data
        """
        # Shared contact normal for all points in this contact set
        normal = arbiter.contact_point_set.normal
        contact_normal = Vec2d(normal.x, normal.y)

        for contact in arbiter.contact_point_set.points:
            # Get the impulse (force * dt)
            impulse = contact_normal * contact.normal_impulse
            contact_point = Vec2d(contact.point_a.x, contact.point_a.y)

            # Check both shapes involved in collision and only process gripper shapes
            for shape in arbiter.shapes:
                if shape.collision_type != 1:
                    continue
                contact_body = shape.body

                # Determine which gripper this body belongs to
                for gripper in self.grippers:
                    if contact_body in (gripper.base_body, gripper.left_jaw, gripper.right_jaw):
                        gripper.add_contact_impulse(impulse, contact_point, contact_body)
                        break

        return True

    def apply_grip_forces(self):
        """Apply grip forces for all grippers."""
        for gripper in self.grippers:
            gripper.apply_grip_force()

    def update_external_wrenches(self):
        """Compute external wrenches from accumulated contact impulses."""
        for gripper in self.grippers:
            gripper.compute_external_wrench(self.dt)

    def step(self):
        """
        Backwards-compatible helper: apply grip forces and update wrenches.
        """
        self.apply_grip_forces()
        self.update_external_wrenches()

    def get_poses(self) -> np.ndarray:
        """Get all gripper poses."""
        return np.array([g.get_pose() for g in self.grippers])

    def get_velocities(self) -> np.ndarray:
        """
        Get all gripper velocities in spatial frame.

        Returns:
            Array of shape (num_grippers, 3) with spatial velocities [vx_s, vy_s, omega]
        """
        return np.array([g.get_velocity() for g in self.grippers])

    def get_body_twists(self) -> np.ndarray:
        """
        Get all gripper body frame twists.

        Following Modern Robotics convention:
        Converts spatial velocities to body frame twists [ω, vx_b, vy_b]

        Returns:
            Array of shape (num_grippers, 3) with body twists [omega, vx_b, vy_b] (MR convention!)
        """
        body_twists = []
        for gripper in self.grippers:
            pose = gripper.get_pose()  # [x, y, theta]
            vel_spatial = gripper.get_velocity()  # [vx_s, vy_s, omega]

            # Convert to MR convention [omega, vx_s, vy_s]
            # get_velocity() returns [vx_s, vy_s, omega], but spatial_to_body_twist expects [omega, vx_s, vy_s]
            vel_spatial_mr = np.array([vel_spatial[2], vel_spatial[0], vel_spatial[1]])

            # Convert spatial velocity to body twist (MR convention)
            twist_body = spatial_to_body_twist(pose, vel_spatial_mr)
            body_twists.append(twist_body)
        return np.array(body_twists)

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
