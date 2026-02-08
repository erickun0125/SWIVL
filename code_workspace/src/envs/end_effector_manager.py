"""
End-Effector Manager

Manages parallel gripper end-effectors:
- U-shaped (ㄷ) parallel gripper with one fixed jaw and one moving jaw
- Constant grasping force on the moving jaw
- External wrench sensing in body frame (Modern Robotics convention)

Gripper Structure:
- Base body: main EE body, control target
- Fixed jaw: rigidly attached to base (as shape)
- Moving jaw: prismatic joint with base, gripping force applied

Left/Right grippers are mirror-symmetric for bimanual manipulation.
"""

import numpy as np
import pymunk
from pymunk import Vec2d
from typing import Tuple, Optional, List
from dataclasses import dataclass
import pygame

# Note: We use simple rotation for velocity frame conversion, NOT spatial_to_body_twist.
# spatial_to_body_twist includes p × ω terms which are only needed for SE(2) twists
# observed from different points, not for point velocities (body origin velocity).


@dataclass
class GripperConfig:
    """Configuration for parallel gripper."""
    # Mass properties
    base_mass: float = 1.0
    jaw_mass: float = 0.1
    
    # Dimensions (pixels)
    base_width: float = 20.0
    base_height: float = 8.0
    jaw_length: float = 20.0
    jaw_thickness: float = 4.0
    
    # Jaw positioning: distance from base edge to jaw center
    jaw_offset_from_edge: float = 2.0
    
    # Grip parameters
    grip_force: float = 15.0
    
    @property
    def initial_opening(self) -> float:
        """Initial opening between jaws."""
        # Jaws are positioned at base_width/2 - jaw_offset_from_edge from center
        jaw_position = self.base_width / 2 - self.jaw_offset_from_edge
        return 2 * jaw_position  # Distance between two jaws
    
    @property
    def min_opening(self) -> float:
        """Minimum opening (when fully closed)."""
        return self.jaw_thickness  # Can't close past jaw thickness


class ParallelGripper:
    """
    1-DOF parallel gripper with one fixed jaw and one moving jaw.
    
    Structure:
    - Base body with fixed jaw attached as shape
    - Moving jaw connected via prismatic joint (GrooveJoint)
    - Grip force applied only to moving jaw
    
    Left gripper: fixed jaw on left, moving jaw on right (closes leftward)
    Right gripper: fixed jaw on right, moving jaw on left (closes rightward)
    """

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        angle: float,
        name: str,
        config: GripperConfig
    ):
        """
        Initialize parallel gripper.

        Args:
            space: Pymunk space
            position: Initial position (x, y) of base center
            angle: Initial angle (radians)
            name: "left" or "right" - determines jaw configuration
            config: Gripper configuration
        """
        self.space = space
        self.name = name
        self.config = config
        
        # Determine which side is fixed/moving based on gripper name
        # Left gripper: fixed left, moving right (物体를 오른쪽에서 왼쪽으로 밀어 잡음)
        # Right gripper: fixed right, moving left (物体를 왼쪽에서 오른쪽으로 밀어 잡음)
        self.is_left_gripper = (name == "left")
        
        # Colors
        color = pygame.Color("RoyalBlue") if self.is_left_gripper else pygame.Color("Crimson")
        
        # Create base body with fixed jaw
        self._create_base_with_fixed_jaw(position, angle, color)
        
        # Create moving jaw
        self._create_moving_jaw(angle, color)
        
        # External wrench tracking (MR convention: [tau, fx, fy])
        self.external_wrench = np.zeros(3)
        self.contact_impulses: List[Tuple[Vec2d, Vec2d, pymunk.Body]] = []

    def _create_base_with_fixed_jaw(self, position: Tuple[float, float], angle: float, color):
        """Create base body with fixed jaw as attached shape."""
        cfg = self.config
        
        # Base body vertices (centered at origin)
        base_verts = [
            (-cfg.base_width/2, -cfg.base_height/2),
            (cfg.base_width/2, -cfg.base_height/2),
            (cfg.base_width/2, cfg.base_height/2),
            (-cfg.base_width/2, cfg.base_height/2),
        ]
        
        # Fixed jaw position (in base local frame)
        if self.is_left_gripper:
            # Left gripper: fixed jaw on left side
            fixed_jaw_x = -cfg.base_width/2 + cfg.jaw_offset_from_edge
        else:
            # Right gripper: fixed jaw on right side
            fixed_jaw_x = cfg.base_width/2 - cfg.jaw_offset_from_edge
        
        # Fixed jaw vertices (relative to base origin)
        # Jaw extends upward (+y) from base top
        fixed_jaw_verts = [
            (fixed_jaw_x - cfg.jaw_thickness/2, cfg.base_height/2),
            (fixed_jaw_x + cfg.jaw_thickness/2, cfg.base_height/2),
            (fixed_jaw_x + cfg.jaw_thickness/2, cfg.base_height/2 + cfg.jaw_length),
            (fixed_jaw_x - cfg.jaw_thickness/2, cfg.base_height/2 + cfg.jaw_length),
        ]
        
        # Calculate total mass and inertia
        base_inertia = pymunk.moment_for_poly(cfg.base_mass, base_verts)
        jaw_inertia = pymunk.moment_for_poly(cfg.jaw_mass, fixed_jaw_verts)
        total_mass = cfg.base_mass + cfg.jaw_mass
        total_inertia = base_inertia + jaw_inertia
        
        # Create body
        self.base_body = pymunk.Body(total_mass, total_inertia, body_type=pymunk.Body.DYNAMIC)
        self.base_body.position = position
        self.base_body.angle = angle
        self.base_body.name = self.name
        
        # Create base shape
        base_shape = pymunk.Poly(self.base_body, base_verts)
        base_shape.color = color
        base_shape.friction = 0.5
        base_shape.filter = pymunk.ShapeFilter(categories=0b01)
        base_shape.collision_type = 1  # Gripper collision type
        
        # Create fixed jaw shape (attached to same body)
        fixed_jaw_shape = pymunk.Poly(self.base_body, fixed_jaw_verts)
        fixed_jaw_shape.color = color
        fixed_jaw_shape.friction = 1.0
        fixed_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        fixed_jaw_shape.collision_type = 1
        
        self.space.add(self.base_body, base_shape, fixed_jaw_shape)
        self.base_shape = base_shape
        self.fixed_jaw_shape = fixed_jaw_shape
        
        # Store fixed jaw position for reference
        self.fixed_jaw_x = fixed_jaw_x

    def _create_moving_jaw(self, angle: float, color):
        """Create moving jaw with prismatic constraint."""
        cfg = self.config
        
        # Moving jaw position (opposite side from fixed jaw)
        if self.is_left_gripper:
            # Left gripper: moving jaw on right side
            moving_jaw_x = cfg.base_width/2 - cfg.jaw_offset_from_edge
            # Groove allows movement from current position toward fixed jaw
            groove_start = Vec2d(moving_jaw_x, cfg.base_height/2)
            groove_end = Vec2d(self.fixed_jaw_x + cfg.jaw_thickness, cfg.base_height/2)
            grip_direction = -1  # Move leftward (negative x)
        else:
            # Right gripper: moving jaw on left side
            moving_jaw_x = -cfg.base_width/2 + cfg.jaw_offset_from_edge
            # Groove allows movement from current position toward fixed jaw
            groove_start = Vec2d(moving_jaw_x, cfg.base_height/2)
            groove_end = Vec2d(self.fixed_jaw_x - cfg.jaw_thickness, cfg.base_height/2)
            grip_direction = 1  # Move rightward (positive x)
        
        self.grip_direction = grip_direction
        
        # Moving jaw vertices (local to jaw body, origin at bottom center)
        jaw_verts = [
            (-cfg.jaw_thickness/2, 0),
            (cfg.jaw_thickness/2, 0),
            (cfg.jaw_thickness/2, cfg.jaw_length),
            (-cfg.jaw_thickness/2, cfg.jaw_length),
        ]
        
        # Create moving jaw body
        jaw_inertia = pymunk.moment_for_poly(cfg.jaw_mass, jaw_verts)
        self.moving_jaw = pymunk.Body(cfg.jaw_mass, jaw_inertia, body_type=pymunk.Body.DYNAMIC)
        self.moving_jaw.position = self.base_body.local_to_world((moving_jaw_x, cfg.base_height/2))
        self.moving_jaw.angle = angle
        
        # Create jaw shape
        moving_jaw_shape = pymunk.Poly(self.moving_jaw, jaw_verts)
        moving_jaw_shape.color = color
        moving_jaw_shape.friction = 1.0
        moving_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        moving_jaw_shape.collision_type = 1
        
        self.space.add(self.moving_jaw, moving_jaw_shape)
        self.moving_jaw_shape = moving_jaw_shape
        
        # Create prismatic joint (GrooveJoint)
        anchor = Vec2d(0, 0)  # Jaw origin (bottom center)
        self.jaw_joint = pymunk.GrooveJoint(
            self.base_body, self.moving_jaw,
            groove_start, groove_end,
            anchor
        )
        self.jaw_joint.collide_bodies = False
        self.space.add(self.jaw_joint)
        
        # Rotation constraint (jaw stays parallel to base)
        self.rotation_constraint = pymunk.GearJoint(self.base_body, self.moving_jaw, 0, 1)
        self.rotation_constraint.max_force = 1e5
        self.rotation_constraint.collide_bodies = False
        self.space.add(self.rotation_constraint)

    def remove_from_space(self):
        """Remove all pymunk entities from space."""
        components = [
            self.base_shape,
            self.fixed_jaw_shape,
            self.moving_jaw_shape,
            self.base_body,
            self.moving_jaw,
            self.jaw_joint,
            self.rotation_constraint,
        ]
        self.space.remove(*[c for c in components if c is not None])
        self.contact_impulses = []

    def apply_grip_force(self):
        """
        Apply constant closing force to moving jaw.
        
        Implements internal force pair (action-reaction):
        - Force F applied to moving jaw (toward fixed jaw)
        - Reaction force -F applied to base body
        """
        force_mag = self.config.grip_force
        
        # Force direction in base local frame
        force_local = Vec2d(self.grip_direction * force_mag, 0)
        force_world = force_local.rotated(self.base_body.angle)
        
        # Apply to moving jaw
        self.moving_jaw.apply_force_at_world_point(force_world, self.moving_jaw.position)
        
        # Apply reaction to base body
        self.base_body.apply_force_at_world_point(-force_world, self.moving_jaw.position)

    def get_pose(self) -> np.ndarray:
        """Get gripper pose [x, y, theta]."""
        return np.array([
            self.base_body.position.x,
            self.base_body.position.y,
            self.base_body.angle
        ])

    def get_velocity(self) -> np.ndarray:
        """Get gripper spatial velocity [vx, vy, omega]."""
        return np.array([
            self.base_body.velocity.x,
            self.base_body.velocity.y,
            self.base_body.angular_velocity
        ])

    def apply_wrench(self, wrench: np.ndarray):
        """
        Apply wrench command in body frame.

        Args:
            wrench: [tau, fx, fy] in body frame (MR convention)
        """
        tau, fx_body, fy_body = wrench
        
        # Transform force to world frame
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

    def add_contact_impulse(self, impulse: Vec2d, contact_point: Vec2d, contact_body: pymunk.Body):
        """Record contact impulse for wrench computation."""
        self.contact_impulses.append((impulse, contact_point, contact_body))

    def compute_external_wrench(self, dt: float) -> np.ndarray:
        """
        Compute external wrench from accumulated contact impulses.

        Args:
            dt: Time duration for impulse-to-force conversion

        Returns:
            External wrench [tau, fx, fy] in body frame (MR convention)
        """
        if not self.contact_impulses:
            self.external_wrench = np.zeros(3)
            return self.external_wrench.copy()

        total_force_world = Vec2d(0, 0)
        total_torque = 0.0

        for impulse, contact_point, _ in self.contact_impulses:
            total_force_world += impulse
            
            # Torque about base body center
            r = contact_point - self.base_body.position
            total_torque += r.x * impulse.y - r.y * impulse.x

        # Convert impulse to force
        total_force_world /= dt
        total_torque /= dt

        # Transform to body frame
        cos_theta = np.cos(self.base_body.angle)
        sin_theta = np.sin(self.base_body.angle)
        fx_body = cos_theta * total_force_world.x + sin_theta * total_force_world.y
        fy_body = -sin_theta * total_force_world.x + cos_theta * total_force_world.y

        self.external_wrench = np.array([total_torque, fx_body, fy_body])
        self.contact_impulses = []
        
        return self.external_wrench.copy()

    def get_external_wrench(self) -> np.ndarray:
        """Get most recent external wrench [tau, fx, fy]."""
        return self.external_wrench.copy()


class EndEffectorManager:
    """
    Manager for bimanual parallel grippers.
    
    Handles creation, control, and sensing for dual grippers.
    """

    def __init__(
        self,
        space: pymunk.Space,
        config: Optional[GripperConfig] = None,
        dt: float = 0.01
    ):
        """
        Initialize end-effector manager.

        Args:
            space: Pymunk space
            config: Gripper configuration
            dt: Physics timestep
        """
        self.space = space
        self.config = config or GripperConfig()
        self.dt = dt
        self.grippers: List[ParallelGripper] = []
        
        self._setup_collision_handlers()

    def reset(self, initial_poses: np.ndarray):
        """
        Reset grippers at specified poses.

        Args:
            initial_poses: Shape (2, 3) with poses [x, y, theta] for [left, right]
        """
        # Remove existing grippers
        for gripper in self.grippers:
            gripper.remove_from_space()
        self.grippers = []

        # Create new grippers
        names = ["left", "right"]
        for i, name in enumerate(names):
            gripper = ParallelGripper(
                space=self.space,
                position=(initial_poses[i, 0], initial_poses[i, 1]),
                angle=initial_poses[i, 2],
                name=name,
                config=self.config
            )
            self.grippers.append(gripper)

    def _setup_collision_handlers(self):
        """Setup collision handlers for force sensing."""
        # Gripper (type 1) vs Object (type 2)
        # pymunk 7.x uses on_collision instead of add_collision_handler
        # Gripper (type 1) vs Object (type 2)
        handler = self.space.add_collision_handler(1, 2)
        handler.post_solve = self._handle_collision

    def _handle_collision(self, arbiter, space, data):
        """Record contact impulses from collisions."""
        impulse = Vec2d(arbiter.total_impulse.x, arbiter.total_impulse.y)
        
        if not arbiter.contact_point_set.points:
            return True
            
        contact = arbiter.contact_point_set.points[0]
        contact_point = Vec2d(contact.point_a.x, contact.point_a.y)

        # Find which gripper was involved
        for shape in arbiter.shapes:
            if shape.collision_type != 1:
                continue
            contact_body = shape.body
            
            for gripper in self.grippers:
                if contact_body in (gripper.base_body, gripper.moving_jaw):
                    gripper.add_contact_impulse(impulse, contact_point, contact_body)
                    break
        
        return True

    def apply_grip_forces(self):
        """Apply grip forces for all grippers."""
        for gripper in self.grippers:
            gripper.apply_grip_force()

    def apply_wrenches(self, wrenches: np.ndarray):
        """Apply wrench commands to grippers."""
        for i, gripper in enumerate(self.grippers):
            gripper.apply_wrench(wrenches[i])

    def update_external_wrenches(self, dt: float):
        """Compute external wrenches from contact impulses."""
        for gripper in self.grippers:
            gripper.compute_external_wrench(dt)

    def get_poses(self) -> np.ndarray:
        """Get gripper poses (2, 3)."""
        return np.array([g.get_pose() for g in self.grippers])

    def get_velocities(self) -> np.ndarray:
        """
        Get EE point velocities in world frame [vx, vy, omega] (2, 3).
        
        NOTE: This returns POINT VELOCITIES, NOT twists!
        - These are the velocities of each gripper's body origin point
        - Observed in the world/spatial frame
        - Order [vx, vy, omega] intentionally differs from twist [omega, vx, vy]
          to make clear this is NOT a twist
        
        For proper body twists, use get_body_twists() instead.
        """
        return np.array([g.get_velocity() for g in self.grippers])

    def get_body_twists(self) -> np.ndarray:
        """
        Get body frame velocities [omega, vx_b, vy_b] (MR convention) (2, 3).
        
        IMPORTANT: We use simple rotation to convert world-frame point velocity
        to body-frame velocity. This is NOT the SE(2) adjoint map!
        
        For point velocity (velocity of body origin):
            v_body = R^T @ v_world
            
        The adjoint map (with p × ω term) is only needed when transforming
        SE(2) twists observed from different points.
        """
        body_twists = []
        for gripper in self.grippers:
            pose = gripper.get_pose()
            vel = gripper.get_velocity()  # [vx, vy, omega] in world frame
            
            # Extract components
            vx_world, vy_world, omega = vel
            theta = pose[2]
            
            # Simple rotation: v_body = R^T @ v_world
            # R^T = [[cos, sin], [-sin, cos]]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            vx_body = cos_theta * vx_world + sin_theta * vy_world
            vy_body = -sin_theta * vx_world + cos_theta * vy_world
            # Angular velocity is frame-independent in 2D
            
            # MR convention: [omega, vx, vy]
            body_twists.append(np.array([omega, vx_body, vy_body]))
            
        return np.array(body_twists)

    def get_external_wrenches(self) -> np.ndarray:
        """Get external wrenches [tau, fx, fy] (2, 3)."""
        return np.array([g.get_external_wrench() for g in self.grippers])

    def get_gripper(self, idx: int) -> ParallelGripper:
        """Get gripper by index (0=left, 1=right)."""
        return self.grippers[idx]
