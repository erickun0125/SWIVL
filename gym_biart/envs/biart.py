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

        # Gripper parameters
        self.gripper_mass = 1.0
        self.gripper_width = 20.0  # Width of U-shape
        self.gripper_height = 30.0  # Height of U-shape
        self.gripper_thickness = 5.0  # Thickness of gripper arms
        self.grip_force = 50.0  # Constant gripping force

        # Object parameters
        self.link_length = 40.0
        self.link_width = 15.0
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
        self.space.damping = 0.95  # Some damping for stability

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
        Add a U-shaped (ㄷ) gripper as a dynamic body.

        The gripper is composed of three rectangles forming a U-shape:
        - Left arm
        - Bottom
        - Right arm
        """
        mass = self.gripper_mass

        # Define vertices for U-shape (three rectangles)
        # Bottom rectangle
        w, h, t = self.gripper_width, self.gripper_height, self.gripper_thickness

        # Left arm
        left_arm_verts = [
            (-w/2, -h/2),
            (-w/2 + t, -h/2),
            (-w/2 + t, h/2),
            (-w/2, h/2),
        ]

        # Bottom
        bottom_verts = [
            (-w/2, -h/2),
            (w/2, -h/2),
            (w/2, -h/2 + t),
            (-w/2, -h/2 + t),
        ]

        # Right arm
        right_arm_verts = [
            (w/2 - t, -h/2),
            (w/2, -h/2),
            (w/2, h/2),
            (w/2 - t, h/2),
        ]

        # Calculate moment of inertia for all three shapes
        inertia = (
            pymunk.moment_for_poly(mass/3, left_arm_verts) +
            pymunk.moment_for_poly(mass/3, bottom_verts) +
            pymunk.moment_for_poly(mass/3, right_arm_verts)
        )

        # Create dynamic body
        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle

        # Create shapes
        left_arm = pymunk.Poly(body, left_arm_verts)
        bottom = pymunk.Poly(body, bottom_verts)
        right_arm = pymunk.Poly(body, right_arm_verts)

        # Set properties
        color = pygame.Color("RoyalBlue") if name == "left" else pygame.Color("Crimson")
        for shape in [left_arm, bottom, right_arm]:
            shape.color = color
            shape.friction = 1.0
            shape.filter = pymunk.ShapeFilter(categories=0b01)  # Gripper category

        # Add to space
        self.space.add(body, left_arm, bottom, right_arm)

        # Store shapes for later reference (use custom attribute, not 'shapes' which is read-only)
        body.custom_shapes = [left_arm, bottom, right_arm]
        body.name = name

        return body

    def _add_articulated_object(self, position, angle, joint_type):
        """
        Add an articulated object with two links connected by a joint.

        Args:
            position: Initial position of the first link
            angle: Initial angle of the first link
            joint_type: "revolute", "prismatic", or "fixed"

        Returns:
            link1, link2, joint
        """
        # Link 1 (base link)
        link1_body, link1_shapes = self._add_link(position, angle, "link1")

        # Link 2 position depends on link 1
        link2_pos = Vec2d(position[0] + self.link_length * np.cos(angle),
                          position[1] + self.link_length * np.sin(angle))
        link2_body, link2_shapes = self._add_link(link2_pos, angle, "link2")

        # Create joint based on type
        if joint_type == "revolute":
            # Revolute joint at the connection point
            joint_pos = link2_pos
            joint = pymunk.PivotJoint(link1_body, link2_body, joint_pos)
            joint.collide_bodies = False

        elif joint_type == "prismatic":
            # Prismatic joint (sliding)
            groove_start = Vec2d(-self.link_length/2, 0)
            groove_end = Vec2d(self.link_length/2, 0)
            anchor = Vec2d(0, 0)
            joint = pymunk.GrooveJoint(link1_body, link2_body, groove_start, groove_end, anchor)
            joint.collide_bodies = False

        elif joint_type == "fixed":
            # Fixed joint (no relative motion)
            joint = pymunk.DampedRotarySpring(link1_body, link2_body, 0, 1e6, 1e6)
            pos_joint = pymunk.PivotJoint(link1_body, link2_body, link2_pos)
            self.space.add(pos_joint)

        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

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
        grasp_size = self.gripper_width * 0.6  # Slightly smaller than gripper opening
        grasp_thickness = self.gripper_thickness * 0.8
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
        for shape in [main_shape, grasp_shape]:
            shape.color = color
            shape.friction = 1.0
            shape.filter = pymunk.ShapeFilter(categories=0b10)  # Object category

        # Add to space
        self.space.add(body, main_shape, grasp_shape)

        # Store reference (use custom attribute, not 'shapes' which is read-only)
        body.custom_shapes = [main_shape, grasp_shape]
        body.name = name

        return body, [main_shape, grasp_shape]

    def _setup_collision_handlers(self):
        """Setup collision handlers for wrench sensing."""
        # TODO: Implement proper collision handling for pymunk 7.x
        # For now, we'll compute external wrenches from body forces
        self.n_contact_points = 0

    def _handle_collision(self, arbiter, space, data):
        """Handle collisions and compute external wrenches."""
        # This is for older pymunk versions
        # In pymunk 7.x, we need to use different API
        self.n_contact_points += 1

    def _update_external_wrench(self, body, contact, impulse):
        """Update external wrench measurements in body frame."""
        # TODO: Implement proper external wrench sensing with pymunk 7.x collision API
        # For now, this is a placeholder
        pass

    def _compute_external_wrenches(self):
        """
        Compute external wrenches from body velocities and applied forces.

        This is a simplified version that estimates external forces from
        velocity changes and applied control forces.
        """
        # For now, we'll compute external wrenches as the difference between
        # observed acceleration and expected acceleration from applied forces

        # This is a simplified implementation
        # In a full implementation, we would use collision callbacks to measure contact forces

        # Reset external wrenches
        self.external_wrench_left = np.zeros(3)
        self.external_wrench_right = np.zeros(3)

        # TODO: Implement proper force sensing using collision detection
        # For now, we'll use zero external wrenches as a placeholder

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
        """Apply constant gripping forces to hold the object."""
        # This is a simplified version - in reality, we would check if the
        # gripper is close to the grasping part and apply forces accordingly

        # For now, we apply a small attractive force between grippers and links
        # This simulates the parallel grip holding the object

        def apply_grip_to_link(gripper, link):
            """Apply grip force if gripper is close to link."""
            distance = (Vec2d(*gripper.position) - Vec2d(*link.position)).length

            if distance < 50:  # Within gripping range
                # Direction from link to gripper
                direction = (Vec2d(*gripper.position) - Vec2d(*link.position)).normalized()

                # Apply attractive force
                force = direction * (-self.grip_force)  # Negative to attract
                link.apply_force_at_world_point(force, link.position)
                gripper.apply_force_at_world_point(-force, gripper.position)

        # Apply grip forces
        apply_grip_to_link(self.left_gripper, self.link1)
        apply_grip_to_link(self.right_gripper, self.link2)

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
        """Randomize initial state."""
        # Randomize gripper positions
        self.left_gripper.position = (
            self.np_random.integers(100, 200),
            self.np_random.integers(300, 400)
        )
        self.left_gripper.angle = self.np_random.uniform(-np.pi, np.pi)

        self.right_gripper.position = (
            self.np_random.integers(300, 400),
            self.np_random.integers(300, 400)
        )
        self.right_gripper.angle = self.np_random.uniform(-np.pi, np.pi)

        # Randomize object position
        self.link1.position = (
            self.np_random.integers(200, 300),
            self.np_random.integers(200, 300)
        )
        self.link1.angle = self.np_random.uniform(-np.pi, np.pi)

        # Link2 position depends on link1
        offset = Vec2d(self.link_length, 0).rotated(self.link1.angle)
        self.link2.position = Vec2d(*self.link1.position) + offset
        self.link2.angle = self.link1.angle

        # Run a few physics steps to settle
        for _ in range(10):
            self.space.step(self.dt)

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
