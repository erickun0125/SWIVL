"""
BiArt (Bimanual Articulated object manipulation) Environment

SE(2) environment for bimanual manipulation of articulated objects.
Features:
- Dual arm robots with dynamic end-effectors
- Wrench command control (force, moment)
- U-shaped (ㄷ) grippers with parallel grip
- Articulated objects (revolute, prismatic, fixed joints)
- External wrench sensing

Architecture:
- EndEffectorManager: Manages parallel grippers with jaw mechanics and wrench sensing
- ObjectManager: Manages articulated objects with pymunk joints and grasping frames
- RewardManager: Computes rewards for RL training
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
from .end_effector_manager import EndEffectorManager, GripperConfig
from .object_manager import ObjectManager
from .reward_manager import RewardManager, RewardWeights

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

    If `obs_type` is set to `state`, the observation is a dictionary with:
    - 'ee_poses': (2, 3) array - poses of both grippers [x, y, theta] in spatial frame
    - 'ee_twists': (2, 3) array - velocities of both grippers [vx, vy, omega] in spatial frame
    - 'link_poses': (2, 3) array - poses of object links [x, y, theta]
    - 'external_wrenches': (2, 3) array - external wrenches [fx, fy, tau] in body frame

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
        control_hz=10,
        physics_hz=100,
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
        self.control_hz = control_hz
        self.physics_hz = physics_hz
        self.dt = 1.0 / physics_hz  # Physics timestep
        self.physics_steps_per_control = physics_hz // control_hz

        # Object parameters (for ObjectManager)
        self.link_length = 40.0
        self.link_width = 12.0
        self.link_mass = 0.5

        # Gripper parameters (for EndEffectorManager)
        self.gripper_config = GripperConfig(
            base_mass=0.8,
            jaw_mass=0.1,
            base_width=25.0,
            base_height=8.0,
            jaw_length=20.0,
            jaw_thickness=4.0,
            max_opening=20.0,
            grip_force=15.0
        )

        # Managers (initialized in reset)
        self.ee_manager = None
        self.object_manager = None
        self.reward_manager = None

        # Physics space (initialized in reset)
        self.space = None

        # Action and observation spaces
        self._initialize_spaces()

        # Rendering
        self.window = None
        self.clock = None

        # State tracking
        self._last_action = None
        self.goal_pose = None  # Set in reset

    def _initialize_spaces(self):
        """Initialize action and observation spaces."""
        # Action space: wrench commands for both grippers
        # [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
        max_force = 100.0
        max_torque = 50.0
        max_velocity = 500.0
        max_angular_velocity = 10.0

        self.action_space = spaces.Box(
            low=np.array([-max_force, -max_force, -max_torque, -max_force, -max_force, -max_torque]),
            high=np.array([max_force, max_force, max_torque, max_force, max_force, max_torque]),
            dtype=np.float32
        )

        # Observation space
        if self.obs_type == "state":
            # Dictionary observation with separate fields
            self.observation_space = spaces.Dict({
                'ee_poses': spaces.Box(
                    low=np.array([[0, 0, -np.pi], [0, 0, -np.pi]]),
                    high=np.array([[512, 512, np.pi], [512, 512, np.pi]]),
                    dtype=np.float32
                ),
                'ee_twists': spaces.Box(
                    low=np.array([
                        [-max_velocity, -max_velocity, -max_angular_velocity],
                        [-max_velocity, -max_velocity, -max_angular_velocity]
                    ]),
                    high=np.array([
                        [max_velocity, max_velocity, max_angular_velocity],
                        [max_velocity, max_velocity, max_angular_velocity]
                    ]),
                    dtype=np.float32
                ),
                'link_poses': spaces.Box(
                    low=np.array([[0, 0, -np.pi], [0, 0, -np.pi]]),
                    high=np.array([[512, 512, np.pi], [512, 512, np.pi]]),
                    dtype=np.float32
                ),
                'external_wrenches': spaces.Box(
                    low=np.array([
                        [-max_force, -max_force, -max_torque],
                        [-max_force, -max_force, -max_torque]
                    ]),
                    high=np.array([
                        [max_force, max_force, max_torque],
                        [max_force, max_force, max_torque]
                    ]),
                    dtype=np.float32
                )
            })
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
        # Create physics space
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

        # Initialize managers
        # Create ObjectManager
        object_params = {
            'link_length': self.link_length,
            'link_width': self.link_width,
            'link_mass': self.link_mass
        }

        self.object_manager = ObjectManager(
            space=self.space,
            joint_type=self.joint_type,  # Pass as string
            object_params=object_params
        )

        # Create EndEffectorManager
        self.ee_manager = EndEffectorManager(
            space=self.space,
            num_grippers=2,
            config=self.gripper_config,
            dt=self.dt
        )

        # Create RewardManager
        self.reward_manager = RewardManager(
            weights=RewardWeights(),
            success_threshold_pos=20.0,
            success_threshold_angle=0.2,
            max_wrench_threshold=200.0
        )

        # Goal pose for the object
        self.goal_pose = np.array([256, 200, np.pi / 4])  # [x, y, theta]

    def _add_segment(self, a, b, radius):
        """Add a static segment (wall) to the environment."""
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        shape.friction = 0.5
        return shape

    def step(self, action):
        """
        Step the environment.

        Args:
            action: [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
        """
        # Store action
        self._last_action = action

        # Parse wrenches from action
        wrenches = np.array([
            action[:3],   # Left gripper wrench
            action[3:]    # Right gripper wrench
        ])

        # Apply wrenches to grippers
        self.ee_manager.apply_wrenches(wrenches)

        # Physics simulation
        for _ in range(self.physics_steps_per_control):
            # Apply grip forces
            self.ee_manager.step()

            # Step physics
            self.space.step(self.dt)

        # Get current states
        current_ee_poses = self.ee_manager.get_poses()
        current_link_poses = self.object_manager.get_link_poses()
        external_wrenches = self.ee_manager.get_external_wrenches()

        # Compute reward using RewardManager
        # For now, use link1 pose as current, goal pose as desired
        desired_ee_poses = np.array([self.goal_pose, self.goal_pose])  # Dummy for now
        desired_ee_velocities = np.zeros((2, 3))
        applied_wrenches = wrenches

        reward_info = self.reward_manager.compute_reward(
            current_ee_poses=current_ee_poses,
            desired_ee_poses=desired_ee_poses,
            current_ee_velocities=np.zeros((2, 3)),  # Not tracked yet
            desired_ee_velocities=desired_ee_velocities,
            applied_wrenches=applied_wrenches,
            external_wrenches=external_wrenches
        )

        # Get observation
        observation = self.get_obs()

        # Build info dict
        info = {
            "is_success": reward_info["is_success"],
            "is_failure": reward_info["is_failure"],
            "total_reward": reward_info["total_reward"],
            "pose_tracking": reward_info["pose_tracking"],
            "velocity_tracking": reward_info["velocity_tracking"],
            "energy_efficiency": reward_info["energy_efficiency"],
            "safety": reward_info["safety"],
            "bonus": reward_info["bonus"],
        }

        # Check termination
        terminated = reward_info["is_success"] or reward_info["is_failure"]
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward_info["total_reward"], terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Setup physics and managers
        self._setup()

        # Randomize object position and orientation
        obj_x = self.np_random.integers(220, 292) if seed is not None else 256
        obj_y = self.np_random.integers(220, 292) if seed is not None else 256
        obj_angle = self.np_random.uniform(-np.pi/4, np.pi/4) if seed is not None else 0.0

        initial_pose = np.array([obj_x, obj_y, obj_angle])

        # Reset object (creates links and joints)
        self.object_manager.reset(initial_pose)

        # Let object settle with joint constraints
        for _ in range(20):
            self.space.step(self.dt)

        # Get grasping poses from object manager
        grasping_poses = self.object_manager.get_grasping_poses()

        # Initialize EEs at grasping frames
        ee_initial_poses = np.array([
            grasping_poses["left"],
            grasping_poses["right"]
        ])

        self.ee_manager.reset(ee_initial_poses)

        # Settle grippers with grip forces
        for _ in range(50):
            self.ee_manager.step()
            self.space.step(self.dt)

        # Get initial observation
        observation = self.get_obs()
        info = {"is_success": False}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def get_obs(self):
        """Get observation."""
        if self.obs_type == "state":
            # Get states from managers
            ee_poses = self.ee_manager.get_poses()  # (2, 3) - spatial frame
            ee_twists = self.ee_manager.get_velocities()  # (2, 3) - spatial frame velocities
            link_poses = self.object_manager.get_link_poses()  # (2, 3)
            external_wrenches = self.ee_manager.get_external_wrenches()  # (2, 3)

            # Return dictionary observation
            obs = {
                'ee_poses': ee_poses.astype(np.float32),
                'ee_twists': ee_twists.astype(np.float32),
                'link_poses': link_poses.astype(np.float32),
                'external_wrenches': external_wrenches.astype(np.float32)
            }

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
        ee_poses = self.ee_manager.get_poses()
        link_poses = self.object_manager.get_link_poses()
        external_wrenches = self.ee_manager.get_external_wrenches()

        return {
            "left_gripper_pose": ee_poses[0],
            "right_gripper_pose": ee_poses[1],
            "link1_pose": link_poses[0],
            "link2_pose": link_poses[1],
            "external_wrench_left": external_wrenches[0],
            "external_wrench_right": external_wrenches[1],
            "goal_pose": self.goal_pose.copy(),
        }
