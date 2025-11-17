"""
BiArt (Bimanual Articulated object manipulation) Environment

SE(2) environment for bimanual manipulation of articulated objects.
Features:
- Dual arm robots with dynamic end-effectors
- Wrench command control (force, moment)
- U-shaped (ㄷ) grippers with parallel grip
- Articulated objects (revolute, prismatic, fixed joints)
- External wrench sensing
"""

import collections
import os
import warnings

import cv2
import gymnasium as gym
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning)
    import pygame

import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from .pymunk_override import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def pymunk_to_shapely(body, shapes):
    """Convert pymunk body and shapes to shapely geometry."""
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class BiArtEnv(gym.Env):
    """
    ## Description

    BiArt (Bimanual Articulated object manipulation) environment for SE(2) dual arm manipulation.

    The goal is to manipulate an articulated object with two robot grippers to reach a goal configuration.

    ## Action Space

    The action space consists of wrench commands for both grippers:
    [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
    - left_fx, left_fy: forces in body frame of left gripper
    - left_tau: moment (torque) in body frame of left gripper
    - right_fx, right_fy, right_tau: same for right gripper

    ## Observation Space

    If `obs_type` is set to `state`, the observation includes:
    - Left gripper: [x, y, theta]
    - Right gripper: [x, y, theta]
    - Object link 1: [x, y, theta]
    - Object link 2: [x, y, theta]
    - External wrenches: [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]

    ## Rewards

    The reward consists of:
    - Tracking reward: how well the object follows desired trajectory
    - Safety reward: penalizes excessive contact forces

    ## Arguments

    * `obs_type`: (str) The observation type. Can be "state", "pixels", etc.
    * `joint_type`: (str) Type of articulated joint: "revolute", "prismatic", or "fixed"
    * `render_mode`: (str) Rendering mode
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        joint_type="revolute",
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
    ):
        super().__init__()

        # Environment configuration
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.joint_type = joint_type  # "revolute", "prismatic", or "fixed"

        # Rendering settings
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Physics parameters
        self.dt = 0.01  # 100 Hz simulation
        self.control_hz = self.metadata["render_fps"]

        # Gripper parameters (parallel gripper)
        self.gripper_base_mass = 0.8
        self.gripper_jaw_mass = 0.1
        self.gripper_base_width = 25.0  # Width of base
        self.gripper_base_height = 8.0  # Height of base
        self.gripper_jaw_length = 20.0  # Length of each jaw
        self.gripper_jaw_thickness = 4.0  # Thickness of jaw
        self.gripper_max_opening = 20.0  # Maximum opening between jaws
        self.grip_force = 15.0  # Constant closing force for each jaw (reduced for stability)

        # Object parameters
        self.link_length = 40.0
        self.link_width = 12.0  # Reduced to fit between jaws better
        self.link_mass = 0.5

        # Action and observation spaces
        self._initialize_spaces()

        # Rendering
        self.window = None
        self.clock = None

        # State tracking
        self._last_action = None
        self.success_threshold = 0.95

        # External wrench tracking
        self.external_wrench_left = np.zeros(3)  # [fx, fy, tau]
        self.external_wrench_right = np.zeros(3)

        # Contact tracking
        self.n_contact_points = 0

        # Gripper jaw references (for parallel gripper)
        self.left_jaw_left = None   # Left gripper's left jaw
        self.left_jaw_right = None  # Left gripper's right jaw
        self.right_jaw_left = None  # Right gripper's left jaw
        self.right_jaw_right = None # Right gripper's right jaw

        # Jaw constraints (prismatic joints)
        self.left_jaw_left_joint = None
        self.left_jaw_right_joint = None
        self.right_jaw_left_joint = None
        self.right_jaw_right_joint = None

        # Collision data for processing
        self.collision_data = []

    def _initialize_spaces(self):
        """Initialize action and observation spaces."""
        # Action space: wrench commands for both grippers
        # [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
        max_force = 100.0
        max_torque = 50.0
        self.action_space = spaces.Box(
            low=np.array([-max_force, -max_force, -max_torque, -max_force, -max_force, -max_torque]),
            high=np.array([max_force, max_force, max_torque, max_force, max_force, max_torque]),
            dtype=np.float32
        )

        # Observation space
        if self.obs_type == "state":
            # State: [left_gripper(3), right_gripper(3), link1(3), link2(3), ext_wrench_left(3), ext_wrench_right(3)]
            # Total: 18 dimensions
            self.observation_space = spaces.Box(
                low=np.array([
                    # Left gripper (x, y, theta)
                    0, 0, -np.pi,
                    # Right gripper (x, y, theta)
                    0, 0, -np.pi,
                    # Link 1 (x, y, theta)
                    0, 0, -np.pi,
                    # Link 2 (x, y, theta)
                    0, 0, -np.pi,
                    # External wrench left (fx, fy, tau)
                    -max_force, -max_force, -max_torque,
                    # External wrench right (fx, fy, tau)
                    -max_force, -max_force, -max_torque,
                ]),
                high=np.array([
                    512, 512, np.pi,
                    512, 512, np.pi,
                    512, 512, np.pi,
                    512, 512, np.pi,
                    max_force, max_force, max_torque,
                    max_force, max_force, max_torque,
                ]),
                dtype=np.float32,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown obs_type {self.obs_type}")

    def _setup(self):
        """Setup the physics simulation."""
        self.space = pymunk.Space()
        self.space.gravity = 0, 0  # No gravity in SE(2)
        self.space.damping = 0.99  # Higher damping for better stability

        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add grippers
        self.left_gripper = self._add_gripper((150, 350), 0, "left")
        self.right_gripper = self._add_gripper((350, 350), 0, "right")

        # Add articulated object
        self.link1, self.link2, self.joint = self._add_articulated_object(
            (256, 256), 0, self.joint_type
        )

        # Setup collision handlers for wrench sensing
        self._setup_collision_handlers()

        # Goal pose for the object
        self.goal_pose = np.array([256, 200, np.pi / 4])  # [x, y, theta]

    def _add_segment(self, a, b, radius):
        """Add a static segment (wall) to the environment."""
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        shape.friction = 0.5
        return shape

    def _add_gripper(self, position, angle, name):
        """
        Add a 1-DOF parallel gripper with two jaws.

        The gripper consists of:
        - Base body: Main gripper body (control target)
        - Left jaw: Movable jaw on the left (prismatic joint)
        - Right jaw: Movable jaw on the right (prismatic joint)

        Each jaw applies a constant closing force to grasp objects.

        Returns:
            base_body: The main gripper body
        """
        color = pygame.Color("RoyalBlue") if name == "left" else pygame.Color("Crimson")
        collision_type_gripper = 1

        # === BASE BODY ===
        base_mass = self.gripper_base_mass
        base_w = self.gripper_base_width
        base_h = self.gripper_base_height

        base_verts = [
            (-base_w/2, -base_h/2),
            (base_w/2, -base_h/2),
            (base_w/2, base_h/2),
            (-base_w/2, base_h/2),
        ]

        base_inertia = pymunk.moment_for_poly(base_mass, base_verts)
        base_body = pymunk.Body(base_mass, base_inertia, body_type=pymunk.Body.DYNAMIC)
        base_body.position = position
        base_body.angle = angle

        base_shape = pymunk.Poly(base_body, base_verts)
        base_shape.color = color
        base_shape.friction = 0.5
        base_shape.filter = pymunk.ShapeFilter(categories=0b01)
        base_shape.collision_type = collision_type_gripper

        self.space.add(base_body, base_shape)

        # === LEFT JAW ===
        jaw_mass = self.gripper_jaw_mass
        jaw_length = self.gripper_jaw_length
        jaw_thickness = self.gripper_jaw_thickness

        # Left jaw vertices (extends downward from base)
        left_jaw_verts = [
            (-jaw_thickness/2, 0),
            (jaw_thickness/2, 0),
            (jaw_thickness/2, jaw_length),
            (-jaw_thickness/2, jaw_length),
        ]

        left_jaw_inertia = pymunk.moment_for_poly(jaw_mass, left_jaw_verts)
        left_jaw_body = pymunk.Body(jaw_mass, left_jaw_inertia, body_type=pymunk.Body.DYNAMIC)

        # Position left jaw to the left of base
        initial_jaw_offset = self.gripper_max_opening / 4  # Start partially open
        left_jaw_local_pos = Vec2d(-initial_jaw_offset, base_h/2)
        left_jaw_body.position = base_body.local_to_world(left_jaw_local_pos)
        left_jaw_body.angle = angle

        left_jaw_shape = pymunk.Poly(left_jaw_body, left_jaw_verts)
        left_jaw_shape.color = color
        left_jaw_shape.friction = 1.5  # High friction for grasping
        left_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        left_jaw_shape.collision_type = collision_type_gripper

        self.space.add(left_jaw_body, left_jaw_shape)

        # === RIGHT JAW ===
        right_jaw_verts = [
            (-jaw_thickness/2, 0),
            (jaw_thickness/2, 0),
            (jaw_thickness/2, jaw_length),
            (-jaw_thickness/2, jaw_length),
        ]

        right_jaw_inertia = pymunk.moment_for_poly(jaw_mass, right_jaw_verts)
        right_jaw_body = pymunk.Body(jaw_mass, right_jaw_inertia, body_type=pymunk.Body.DYNAMIC)

        # Position right jaw to the right of base
        right_jaw_local_pos = Vec2d(initial_jaw_offset, base_h/2)
        right_jaw_body.position = base_body.local_to_world(right_jaw_local_pos)
        right_jaw_body.angle = angle

        right_jaw_shape = pymunk.Poly(right_jaw_body, right_jaw_verts)
        right_jaw_shape.color = color
        right_jaw_shape.friction = 1.5  # High friction for grasping
        right_jaw_shape.filter = pymunk.ShapeFilter(categories=0b01)
        right_jaw_shape.collision_type = collision_type_gripper

        self.space.add(right_jaw_body, right_jaw_shape)

        # === JAW CONSTRAINTS (Prismatic Joints) ===
        # Left jaw: can slide along base's x-axis
        groove_length = self.gripper_max_opening / 2
        left_groove_start = Vec2d(-groove_length, base_h/2)
        left_groove_end = Vec2d(0, base_h/2)
        left_anchor = Vec2d(0, 0)  # Anchor point on jaw

        left_jaw_joint = pymunk.GrooveJoint(
            base_body, left_jaw_body,
            left_groove_start, left_groove_end, left_anchor
        )
        left_jaw_joint.collide_bodies = False

        # Right jaw: can slide along base's x-axis
        right_groove_start = Vec2d(0, base_h/2)
        right_groove_end = Vec2d(groove_length, base_h/2)
        right_anchor = Vec2d(0, 0)

        right_jaw_joint = pymunk.GrooveJoint(
            base_body, right_jaw_body,
            right_groove_start, right_groove_end, right_anchor
        )
        right_jaw_joint.collide_bodies = False

        self.space.add(left_jaw_joint, right_jaw_joint)

        # Pin joint to constrain rotation (jaws stay parallel to base)
        left_pin = pymunk.PinJoint(base_body, left_jaw_body, left_jaw_local_pos, left_anchor)
        right_pin = pymunk.PinJoint(base_body, right_jaw_body, right_jaw_local_pos, right_anchor)
        left_pin.collide_bodies = False
        right_pin.collide_bodies = False
        self.space.add(left_pin, right_pin)

        # Rotary limit to keep jaws aligned
        left_rot_limit = pymunk.RotaryLimitJoint(base_body, left_jaw_body, 0, 0)
        right_rot_limit = pymunk.RotaryLimitJoint(base_body, right_jaw_body, 0, 0)
        self.space.add(left_rot_limit, right_rot_limit)

        # Store references
        base_body.name = name
        base_body.custom_shapes = [base_shape]

        # Store jaw references for force application
        if name == "left":
            self.left_jaw_left = left_jaw_body
            self.left_jaw_right = right_jaw_body
            self.left_jaw_left_joint = left_jaw_joint
            self.left_jaw_right_joint = right_jaw_joint
        else:  # "right"
            self.right_jaw_left = left_jaw_body
            self.right_jaw_right = right_jaw_body
            self.right_jaw_left_joint = left_jaw_joint
            self.right_jaw_right_joint = right_jaw_joint

        return base_body

    def _add_articulated_object(self, position, angle, joint_type):
        """
        Add an articulated object with two links connected by a joint.

        Args:
            position: Initial position of the first link's CENTER
            angle: Initial angle of the first link
            joint_type: "revolute", "prismatic", or "fixed"

        Returns:
            link1, link2, joint

        Joint connection:
        - Link1 right end (+length/2, 0) connects to Link2 left end (-length/2, 0)
        - For revolute: allows rotation around connection point
        - For prismatic: allows sliding along link1's axis
        - For fixed: rigid connection
        """
        # Link 1 (base link) - centered at position
        link1_body, link1_shapes = self._add_link(position, angle, "link1")

        # Calculate connection point (link1's right end in world frame)
        link1_end_local = Vec2d(self.link_length / 2, 0)
        connection_point_world = link1_body.local_to_world(link1_end_local)

        # Link 2 position and angle depend on joint type
        if joint_type == "revolute":
            # For revolute, link2 can be at an angle
            # Position link2's center so its left end is at connection point
            # Initial angle offset for V-shape (will be constrained by joint)
            link2_angle = angle  # Start aligned, joint will control relative angle
            link2_start_to_center = Vec2d(self.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_point_world + link2_start_to_center

        elif joint_type == "prismatic":
            # For prismatic, links stay aligned
            link2_angle = angle
            link2_start_to_center = Vec2d(self.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_point_world + link2_start_to_center

        elif joint_type == "fixed":
            # For fixed, links are rigidly aligned
            link2_angle = angle
            link2_start_to_center = Vec2d(self.link_length / 2, 0).rotated(link2_angle)
            link2_center = connection_point_world + link2_start_to_center

        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

        # Create link 2
        link2_body, link2_shapes = self._add_link(link2_center, link2_angle, "link2")

        # Create joint using WORLD coordinates (simpler and more reliable)
        # Use the connection point calculated earlier
        if joint_type == "revolute":
            # PivotJoint: allows rotation around connection point
            # Use single-argument form with world coordinates
            joint = pymunk.PivotJoint(link1_body, link2_body, connection_point_world)
            joint.collide_bodies = False

            # Add rotary limit joint to enforce V-shape angle range
            # This constrains the relative angle between link1 and link2
            # Angle range: -120° to -60° (V-shape opening downward)
            angle_limit = pymunk.RotaryLimitJoint(
                link1_body, link2_body,
                -2*np.pi/3,  # min: -120°
                -np.pi/3     # max: -60°
            )
            angle_limit.collide_bodies = False
            self.space.add(angle_limit)

        elif joint_type == "prismatic":
            # PrismaticJoint: allows sliding along link1's x-axis
            # GrooveJoint: one body slides in a groove on the other
            # Groove is on link1, along its x-axis (in link1's local frame)
            groove_start = Vec2d(0, 0)  # Center of link1
            groove_end = Vec2d(self.link_length, 0)  # Along link1's length
            anchor = Vec2d(-self.link_length / 2, 0)  # Link2's left end in link2's local frame

            joint = pymunk.GrooveJoint(link1_body, link2_body, groove_start, groove_end, anchor)
            joint.collide_bodies = False

            # Add a PinJoint to keep link2 aligned with link1 (no rotation)
            # Use world coordinate form
            pin_joint = pymunk.PinJoint(link1_body, link2_body, connection_point_world, connection_point_world)
            pin_joint.distance = 0  # Keep at fixed distance
            pin_joint.collide_bodies = False
            self.space.add(pin_joint)

        elif joint_type == "fixed":
            # Fixed joint: no relative motion
            # Use both rotary and linear springs with very high stiffness
            joint = pymunk.DampedRotarySpring(link1_body, link2_body, 0, 1e8, 1e6)
            # Use world coordinate form for PivotJoint
            pos_joint = pymunk.PivotJoint(link1_body, link2_body, connection_point_world)
            pos_joint.collide_bodies = False
            self.space.add(pos_joint)

        self.space.add(joint)

        return link1_body, link2_body, joint

    def _add_link(self, position, angle, name):
        """
        Add a single link with a grasping part.

        Each link consists of:
        - Main body (rectangle)
        - Grasping part (small protrusion that fits in gripper)
        """
        mass = self.link_mass

        # Main body vertices
        l, w = self.link_length, self.link_width
        main_verts = [
            (-l/2, -w/2),
            (l/2, -w/2),
            (l/2, w/2),
            (-l/2, w/2),
        ]

        # Grasping part (small rectangle at the center)
        # Should fit between gripper jaws
        grasp_size = self.gripper_max_opening * 0.4  # Smaller than max jaw opening
        grasp_thickness = self.gripper_jaw_thickness * 0.8
        grasp_verts = [
            (-grasp_size/2, -grasp_thickness/2),
            (grasp_size/2, -grasp_thickness/2),
            (grasp_size/2, grasp_thickness/2),
            (-grasp_size/2, grasp_thickness/2),
        ]

        # Calculate inertia
        inertia = (
            pymunk.moment_for_poly(mass * 0.8, main_verts) +
            pymunk.moment_for_poly(mass * 0.2, grasp_verts)
        )

        # Create body
        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle

        # Create shapes
        main_shape = pymunk.Poly(body, main_verts)
        grasp_shape = pymunk.Poly(body, grasp_verts)

        # Set properties
        color = pygame.Color("LightSlateGray")
        collision_type_object = 2  # Collision type for objects
        for shape in [main_shape, grasp_shape]:
            shape.color = color
            shape.friction = 1.0
            shape.filter = pymunk.ShapeFilter(categories=0b10)  # Object category
            shape.collision_type = collision_type_object

        # Add to space
        self.space.add(body, main_shape, grasp_shape)

        # Store reference (use custom attribute, not 'shapes' which is read-only)
        body.custom_shapes = [main_shape, grasp_shape]
        body.name = name

        return body, [main_shape, grasp_shape]

    def _setup_collision_handlers(self):
        """
        Setup collision handlers for wrench sensing.

        Note: Pymunk 7.x has a different collision API than 6.x.
        For now, we use a simpler approach: measure external wrenches by
        examining contact forces using space.arbiters after each physics step.
        """
        # Reset contact tracking
        self.n_contact_points = 0
        self.collision_data = []  # Store collision data for processing

    def _accumulate_external_wrench(self, body, contact, total_impulse, is_first_contact):
        """
        Accumulate external wrench from collision contact.

        Args:
            body: The gripper body
            contact: Contact point data
            total_impulse: Total impulse from collision
            is_first_contact: Whether this is the first contact
        """
        # Convert impulse to force (F = impulse / dt)
        force_world = Vec2d(total_impulse.x, total_impulse.y) / self.dt

        # Transform force to body frame
        # World to body rotation
        cos_angle = np.cos(-body.angle)
        sin_angle = np.sin(-body.angle)

        force_body_x = cos_angle * force_world.x - sin_angle * force_world.y
        force_body_y = sin_angle * force_world.x + cos_angle * force_world.y

        # Compute moment (torque) in body frame
        # Get contact point relative to body COM in world frame
        contact_point = Vec2d(contact.point_a.x, contact.point_a.y)
        r_world = contact_point - body.position

        # Transform r to body frame
        r_body_x = cos_angle * r_world.x - sin_angle * r_world.y
        r_body_y = sin_angle * r_world.x + cos_angle * r_world.y

        # Moment = r × F (2D cross product)
        moment_body = r_body_x * force_body_y - r_body_y * force_body_x

        # Accumulate to external wrench
        if body.name == "left":
            self.external_wrench_left += np.array([force_body_x, force_body_y, moment_body])
        elif body.name == "right":
            self.external_wrench_right += np.array([force_body_x, force_body_y, moment_body])

    def _compute_external_wrenches(self):
        """
        Compute external wrenches from jaw-object contact forces.

        For parallel grippers, the external wrench comes from:
        - Closing forces applied by jaws onto grasped object
        - Reaction forces from object onto gripper base
        """
        # Reset contact tracking
        self.n_contact_points = 0

        # === Left Gripper ===
        if self.left_jaw_left is not None and self.left_jaw_right is not None:
            # Estimate force from jaw positions relative to link
            gripper_pos = Vec2d(*self.left_gripper.position)
            link_pos = Vec2d(*self.link1.position)

            # Check if jaws are in contact (link is between jaws)
            jaw_left_pos = Vec2d(*self.left_jaw_left.position)
            jaw_right_pos = Vec2d(*self.left_jaw_right.position)

            # Distance from each jaw to link (in gripper frame)
            cos_angle = np.cos(-self.left_gripper.angle)
            sin_angle = np.sin(-self.left_gripper.angle)

            # Link position in gripper frame
            rel_pos_world = link_pos - gripper_pos
            link_local_x = cos_angle * rel_pos_world.x - sin_angle * rel_pos_world.y
            link_local_y = sin_angle * rel_pos_world.x + cos_angle * rel_pos_world.y

            # Jaw positions in gripper frame
            rel_jaw_left = jaw_left_pos - gripper_pos
            jaw_left_x = cos_angle * rel_jaw_left.x - sin_angle * rel_jaw_left.y

            rel_jaw_right = jaw_right_pos - gripper_pos
            jaw_right_x = cos_angle * rel_jaw_right.x - sin_angle * rel_jaw_right.y

            # Estimate contact forces based on jaw-link distance
            # If link is between jaws, both jaws apply closing force
            contact_threshold = self.link_width * 1.2  # Threshold for contact

            total_force_x = 0.0
            total_force_y = 0.0

            # Check if link is roughly in grasping zone
            if abs(link_local_y - self.gripper_jaw_length/2) < self.gripper_jaw_length:
                # Link is in vertical range of jaws
                left_dist = link_local_x - jaw_left_x
                right_dist = jaw_right_x - link_local_x

                if left_dist > 0 and right_dist > 0:
                    # Link is between jaws - apply closing forces
                    # Forces in gripper frame: jaws push inward (±x direction)
                    # These create reaction force on gripper base
                    total_force_x = 0.0  # Forces cancel out in x
                    # Y force from friction/drag
                    total_force_y = self.grip_force * 0.1  # Small friction component

                    self.n_contact_points += 2  # Two jaws in contact

            # Transform to body frame (already in gripper frame)
            # Add small position-based spring for stability
            position_error_local_x = link_local_x
            position_error_local_y = link_local_y - self.gripper_jaw_length/2

            spring_force_x = -position_error_local_x * 5.0
            spring_force_y = -position_error_local_y * 5.0

            force_body_x = total_force_x + spring_force_x
            force_body_y = total_force_y + spring_force_y

            # Estimate moment from link-gripper angular difference
            angle_diff = self.link1.angle - self.left_gripper.angle
            moment_body = angle_diff * 10.0

            self.external_wrench_left = np.array([force_body_x, force_body_y, moment_body])

        # === Right Gripper ===
        if self.right_jaw_left is not None and self.right_jaw_right is not None:
            # Same procedure for right gripper
            gripper_pos = Vec2d(*self.right_gripper.position)
            link_pos = Vec2d(*self.link2.position)

            jaw_left_pos = Vec2d(*self.right_jaw_left.position)
            jaw_right_pos = Vec2d(*self.right_jaw_right.position)

            cos_angle = np.cos(-self.right_gripper.angle)
            sin_angle = np.sin(-self.right_gripper.angle)

            rel_pos_world = link_pos - gripper_pos
            link_local_x = cos_angle * rel_pos_world.x - sin_angle * rel_pos_world.y
            link_local_y = sin_angle * rel_pos_world.x + cos_angle * rel_pos_world.y

            rel_jaw_left = jaw_left_pos - gripper_pos
            jaw_left_x = cos_angle * rel_jaw_left.x - sin_angle * rel_jaw_left.y

            rel_jaw_right = jaw_right_pos - gripper_pos
            jaw_right_x = cos_angle * rel_jaw_right.x - sin_angle * rel_jaw_right.y

            total_force_x = 0.0
            total_force_y = 0.0

            if abs(link_local_y - self.gripper_jaw_length/2) < self.gripper_jaw_length:
                left_dist = link_local_x - jaw_left_x
                right_dist = jaw_right_x - link_local_x

                if left_dist > 0 and right_dist > 0:
                    total_force_x = 0.0
                    total_force_y = self.grip_force * 0.1
                    self.n_contact_points += 2

            position_error_local_x = link_local_x
            position_error_local_y = link_local_y - self.gripper_jaw_length/2

            spring_force_x = -position_error_local_x * 5.0
            spring_force_y = -position_error_local_y * 5.0

            force_body_x = total_force_x + spring_force_x
            force_body_y = total_force_y + spring_force_y

            angle_diff = self.link2.angle - self.right_gripper.angle
            moment_body = angle_diff * 10.0

            self.external_wrench_right = np.array([force_body_x, force_body_y, moment_body])

    def step(self, action):
        """
        Step the environment.

        Args:
            action: [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
        """
        # Reset contact tracking
        self.n_contact_points = 0
        self.external_wrench_left = np.zeros(3)
        self.external_wrench_right = np.zeros(3)

        # Store action
        self._last_action = action

        # Number of physics steps per control step
        n_steps = int(1 / (self.dt * self.control_hz))

        # Apply wrench commands
        left_wrench = action[:3]  # [fx, fy, tau]
        right_wrench = action[3:]  # [fx, fy, tau]

        for _ in range(n_steps):
            # Apply forces to grippers in body frame
            self._apply_wrench(self.left_gripper, left_wrench)
            self._apply_wrench(self.right_gripper, right_wrench)

            # Apply gripping forces (constant force to hold object)
            self._apply_grip_forces()

            # Step physics
            self.space.step(self.dt)

        # Compute external wrenches (simplified for now)
        self._compute_external_wrenches()

        # Get observation
        observation = self.get_obs()

        # Compute reward
        reward, info = self._compute_reward()

        # Check termination
        terminated = info.get("is_success", False)
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _apply_wrench(self, body, wrench):
        """
        Apply wrench (force + moment) to a body in its local frame.

        Args:
            body: pymunk.Body
            wrench: [fx, fy, tau] in body frame
        """
        # Transform force from body frame to world frame
        force_body = Vec2d(wrench[0], wrench[1])
        force_world = body.local_to_world(body.position + force_body) - body.position

        # Apply force at center of mass
        body.apply_force_at_world_point(force_world, body.position)

        # Apply torque
        body.torque += wrench[2]

    def _apply_grip_forces(self):
        """
        Apply constant closing forces to gripper jaws.

        Each jaw applies a constant force towards the center (closing direction)
        to grasp objects. This simulates a spring-loaded gripper mechanism.
        """
        # Left gripper jaws
        if self.left_jaw_left is not None and self.left_jaw_right is not None:
            # Left jaw: apply force to the right (positive x in gripper frame)
            # Right jaw: apply force to the left (negative x in gripper frame)

            # Transform closing force to world frame
            gripper_angle = self.left_gripper.angle

            # Closing direction for left jaw (towards right, in gripper frame)
            left_close_dir_local = Vec2d(1, 0)  # Positive x
            left_close_dir_world = left_close_dir_local.rotated(gripper_angle)

            # Closing direction for right jaw (towards left, in gripper frame)
            right_close_dir_local = Vec2d(-1, 0)  # Negative x
            right_close_dir_world = right_close_dir_local.rotated(gripper_angle)

            # Apply forces at jaw center of mass
            self.left_jaw_left.apply_force_at_world_point(
                left_close_dir_world * self.grip_force,
                self.left_jaw_left.position
            )

            self.left_jaw_right.apply_force_at_world_point(
                right_close_dir_world * self.grip_force,
                self.left_jaw_right.position
            )

        # Right gripper jaws
        if self.right_jaw_left is not None and self.right_jaw_right is not None:
            # Transform closing force to world frame
            gripper_angle = self.right_gripper.angle

            # Closing direction for left jaw
            left_close_dir_local = Vec2d(1, 0)
            left_close_dir_world = left_close_dir_local.rotated(gripper_angle)

            # Closing direction for right jaw
            right_close_dir_local = Vec2d(-1, 0)
            right_close_dir_world = right_close_dir_local.rotated(gripper_angle)

            # Apply forces
            self.right_jaw_left.apply_force_at_world_point(
                left_close_dir_world * self.grip_force,
                self.right_jaw_left.position
            )

            self.right_jaw_right.apply_force_at_world_point(
                right_close_dir_world * self.grip_force,
                self.right_jaw_right.position
            )

    def _compute_reward(self):
        """
        Compute reward based on tracking and safety.

        Returns:
            reward: float
            info: dict
        """
        # Tracking reward: how close is link1 to goal pose
        pos_error = np.linalg.norm(np.array(self.link1.position) - self.goal_pose[:2])
        angle_error = abs(self.link1.angle - self.goal_pose[2])
        angle_error = min(angle_error, 2*np.pi - angle_error)  # Wrap to [0, pi]

        tracking_reward = np.exp(-0.1 * pos_error) * np.exp(-angle_error)

        # Safety reward: penalize excessive forces
        max_wrench_left = np.max(np.abs(self.external_wrench_left))
        max_wrench_right = np.max(np.abs(self.external_wrench_right))
        max_wrench = max(max_wrench_left, max_wrench_right)

        safety_penalty = -0.01 * max_wrench

        # Total reward
        reward = tracking_reward + safety_penalty

        # Success criterion
        is_success = pos_error < 20 and angle_error < 0.2

        info = {
            "is_success": is_success,
            "tracking_reward": tracking_reward,
            "safety_penalty": safety_penalty,
            "pos_error": pos_error,
            "angle_error": angle_error,
            "n_contacts": self.n_contact_points,
        }

        return reward, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Setup physics
        self._setup()

        # Randomize initial state if no specific state is provided
        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
            self._set_state(state)
        else:
            # Random initialization
            self._randomize_state()

        # Get initial observation
        observation = self.get_obs()
        info = {"is_success": False}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _randomize_state(self):
        """
        Randomize initial state with grippers holding the grasping parts.

        CRITICAL: Since links are connected by joints, we must be careful not to violate constraints!
        Strategy: Set link1 position, compute desired link2 state, then let physics resolve constraints.
        """
        # Randomize link1 position and orientation
        obj_x = self.np_random.integers(220, 292)
        obj_y = self.np_random.integers(220, 292)
        obj_angle = self.np_random.uniform(-np.pi/4, np.pi/4)  # Limit rotation for stability

        self.link1.position = (obj_x, obj_y)
        self.link1.angle = obj_angle
        self.link1.velocity = Vec2d(0, 0)
        self.link1.angular_velocity = 0

        # Calculate connection point (link1's right end in world coords)
        connection_point = self.link1.local_to_world(Vec2d(self.link_length / 2, 0))

        # Compute DESIRED link2 state based on joint type
        # We'll set link2 close to this, then let the joint constraint pull it exact
        if self.joint_type == "revolute":
            # For revolute, create V-shape with link2 at angle
            joint_angle_offset = self.np_random.uniform(-2*np.pi/3, -np.pi/3)  # -120° to -60°
            desired_link2_angle = self.link1.angle + joint_angle_offset

            # Link2's center should be: connection_point + (link_length/2) in link2's direction
            link2_center_offset = Vec2d(self.link_length / 2, 0).rotated(desired_link2_angle)
            desired_link2_position = connection_point + link2_center_offset

        elif self.joint_type == "prismatic":
            # For prismatic, links stay aligned in angle
            desired_link2_angle = self.link1.angle
            # Link2 can slide along link1's axis - start at extended position
            link2_center_offset = Vec2d(self.link_length / 2, 0).rotated(desired_link2_angle)
            desired_link2_position = connection_point + link2_center_offset

        else:  # fixed
            # For fixed, links are rigidly connected
            desired_link2_angle = self.link1.angle
            link2_center_offset = Vec2d(self.link_length / 2, 0).rotated(desired_link2_angle)
            desired_link2_position = connection_point + link2_center_offset

        # Set link2 state (joint will adjust it if needed)
        self.link2.position = desired_link2_position
        self.link2.angle = desired_link2_angle
        self.link2.velocity = Vec2d(0, 0)
        self.link2.angular_velocity = 0

        # IMPORTANT: Let joint constraints resolve for a few steps BEFORE adding grippers
        # This prevents constraint fighting
        for _ in range(20):
            self.space.step(self.dt)
            # Damp any velocities from constraint resolution
            self.link1.velocity *= 0.9
            self.link1.angular_velocity *= 0.9
            self.link2.velocity *= 0.9
            self.link2.angular_velocity *= 0.9

        # NOW position grippers to grasp the links (after joint is settled)
        # Left gripper grasps link1
        self._position_gripper_to_grasp(self.left_gripper, self.link1)

        # Right gripper grasps link2
        self._position_gripper_to_grasp(self.right_gripper, self.link2)

        # Run physics steps to let grippers settle and grasp objects
        # The constant closing forces will automatically grasp the objects
        for _ in range(50):
            # Apply grip forces during settling
            self._apply_grip_forces()
            self.space.step(self.dt)

            # Apply damping to reduce oscillations
            self.left_gripper.velocity *= 0.95
            self.left_gripper.angular_velocity *= 0.95
            self.right_gripper.velocity *= 0.95
            self.right_gripper.angular_velocity *= 0.95
            self.link1.velocity *= 0.95
            self.link1.angular_velocity *= 0.95
            self.link2.velocity *= 0.95
            self.link2.angular_velocity *= 0.95

    def _position_gripper_to_grasp(self, gripper, link):
        """
        Position parallel gripper to grasp the link.

        The gripper's body frame is aligned with the object's grasping frame:
        - Gripper x-axis parallel to link's y-axis (jaws open/close perpendicular to link)
        - Gripper y-axis parallel to link's x-axis (jaws extend along link's length)
        - Link is positioned between the two jaws
        """
        # Align gripper frame with object grasping frame
        # Gripper angle = link angle + 90 degrees
        # This makes gripper's x-axis perpendicular to link, y-axis along link
        gripper_angle = link.angle + np.pi / 2

        # Position gripper base so that:
        # 1. Gripper center aligns with link center (in x-y plane)
        # 2. Jaws extend to straddle the link

        # Calculate offset to position gripper base above link
        # When jaws extend down (in gripper's +y direction), they should straddle the link
        jaw_extension_to_link_center = self.gripper_jaw_length / 2

        # Offset gripper base from link center (in gripper's y direction)
        offset_in_gripper_y = -(self.gripper_base_height/2 + jaw_extension_to_link_center)
        offset_world = Vec2d(0, offset_in_gripper_y).rotated(gripper_angle)

        # Position gripper
        gripper.position = Vec2d(*link.position) + offset_world
        gripper.angle = gripper_angle

        # Update jaw positions to match gripper
        if gripper.name == "left":
            jaw_left = self.left_jaw_left
            jaw_right = self.left_jaw_right
        else:
            jaw_left = self.right_jaw_left
            jaw_right = self.right_jaw_right

        if jaw_left is not None and jaw_right is not None:
            # Start jaws partially open (symmetric around center)
            initial_jaw_offset = self.link_width / 2 + 1.0  # Slightly wider than link
            base_h = self.gripper_base_height

            left_jaw_local_pos = Vec2d(-initial_jaw_offset, base_h/2)
            right_jaw_local_pos = Vec2d(initial_jaw_offset, base_h/2)

            jaw_left.position = gripper.local_to_world(left_jaw_local_pos)
            jaw_left.angle = gripper.angle

            jaw_right.position = gripper.local_to_world(right_jaw_local_pos)
            jaw_right.angle = gripper.angle

        # Set all velocities to zero
        gripper.velocity = Vec2d(0, 0)
        gripper.angular_velocity = 0

        if jaw_left is not None:
            jaw_left.velocity = Vec2d(0, 0)
            jaw_left.angular_velocity = 0

        if jaw_right is not None:
            jaw_right.velocity = Vec2d(0, 0)
            jaw_right.angular_velocity = 0

    def _create_grip_constraints(self):
        """
        No longer needed - parallel gripper jaws grasp automatically via constant closing forces.

        This function is kept for backward compatibility but does nothing.
        """
        pass

    def _remove_grip_constraints(self):
        """
        No longer needed - parallel gripper jaws release when closing forces are removed.

        This function is kept for backward compatibility but does nothing.
        """
        pass

    def _set_state(self, state):
        """Set environment to a specific state."""
        # State: [left_gripper(3), right_gripper(3), link1(3), link2(3)]
        # Total: 12 dimensions for pose only

        self.left_gripper.position = state[0:2].tolist()
        self.left_gripper.angle = state[2]

        self.right_gripper.position = state[3:5].tolist()
        self.right_gripper.angle = state[5]

        self.link1.position = state[6:8].tolist()
        self.link1.angle = state[8]

        self.link2.position = state[9:11].tolist()
        self.link2.angle = state[11]

        # Run physics to take effect
        self.space.step(self.dt)

    def get_obs(self):
        """Get observation."""
        if self.obs_type == "state":
            # Construct state observation
            obs = np.concatenate([
                # Left gripper
                np.array(self.left_gripper.position),
                [self.left_gripper.angle],
                # Right gripper
                np.array(self.right_gripper.position),
                [self.right_gripper.angle],
                # Link 1
                np.array(self.link1.position),
                [self.link1.angle],
                # Link 2
                np.array(self.link2.position),
                [self.link2.angle],
                # External wrenches
                self.external_wrench_left,
                self.external_wrench_right,
            ], dtype=np.float32)

            return obs

        elif self.obs_type == "pixels":
            return self._render()

        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def _draw(self):
        """Draw the environment."""
        # Create screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)

        # Draw goal pose
        goal_x, goal_y, goal_theta = self.goal_pose
        goal_size = 30
        goal_rect = [
            (goal_x - goal_size, goal_y - goal_size),
            (goal_x + goal_size, goal_y - goal_size),
            (goal_x + goal_size, goal_y + goal_size),
            (goal_x - goal_size, goal_y + goal_size),
        ]
        goal_rect_pygame = [pymunk.pygame_util.to_pygame(p, screen) for p in goal_rect]
        pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_rect_pygame, 2)

        # Draw physics objects
        self.space.debug_draw(draw_options)

        return screen

    def _render(self, visualize=False):
        """Render the environment."""
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )

        screen = self._draw()

        if self.render_mode == "rgb_array":
            img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
            img = cv2.resize(img, (width, height))
            return img

        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(screen, screen.get_rect())
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()

        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def render(self):
        """Render the environment."""
        return self._render(visualize=True)

    def close(self):
        """Close the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_info(self):
        """Get info dict."""
        return {
            "left_gripper_pose": np.array([*self.left_gripper.position, self.left_gripper.angle]),
            "right_gripper_pose": np.array([*self.right_gripper.position, self.right_gripper.angle]),
            "link1_pose": np.array([*self.link1.position, self.link1.angle]),
            "link2_pose": np.array([*self.link2.position, self.link2.angle]),
            "external_wrench_left": self.external_wrench_left.copy(),
            "external_wrench_right": self.external_wrench_right.copy(),
            "goal_pose": self.goal_pose.copy(),
        }
