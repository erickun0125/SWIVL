"""
BiArt (Bimanual Articulated object manipulation) Environment

SE(2) environment for bimanual manipulation of articulated objects.

Features:
- Dual parallel grippers with 1-DOF jaw mechanism
- Articulated objects with revolute, prismatic, or fixed joints
- Wrench-based control (Modern Robotics convention)
- External wrench sensing from contact forces

Physical Units (Pymunk convention):
- Length: pixels (512×512 workspace)
- Mass: arbitrary units
- Time: seconds
- Force/Torque: mass × pixels/second²
"""

import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", DeprecationWarning)
    import pygame

import pymunk
import pymunk.pygame_util
from gymnasium import spaces

from .pymunk_override import DrawOptions
from .end_effector_manager import EndEffectorManager, GripperConfig
from .object_manager import ObjectManager, ObjectConfig, JointType
from .reward_manager import RewardManager, RewardWeights


RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


class BiArtEnv(gym.Env):
    """
    Bimanual Articulated object manipulation Environment.
    
    Action Space:
        Wrench commands for both grippers (MR convention):
        [left_tau, left_fx, left_fy, right_tau, right_fx, right_fy]
    
    Observation Space (state mode):
        Dictionary with:
        - ee_poses: (2, 3) gripper poses [x, y, theta]
        - ee_twists: (2, 3) spatial velocities [vx, vy, omega]
        - ee_body_twists: (2, 3) body twists [omega, vx, vy] (MR convention)
        - link_poses: (2, 3) object link poses
        - external_wrenches: (2, 3) body wrenches [tau, fx, fy] (MR convention)
    """

    metadata = {"render_modes": RENDER_MODES, "render_fps": 10}

    def __init__(
        self,
        obs_type: str = "state",
        render_mode: str = "rgb_array",
        joint_type: str = "revolute",
        observation_width: int = 96,
        observation_height: int = 96,
        visualization_width: int = 680,
        visualization_height: int = 680,
        control_hz: int = 10,
        physics_hz: int = 100,
    ):
        """
        Initialize environment.

        Args:
            obs_type: "state" or "pixels"
            render_mode: "rgb_array" or "human"
            joint_type: "revolute", "prismatic", "fixed", or "none"
            control_hz: Control frequency
            physics_hz: Physics simulation frequency
        """
        super().__init__()

        self.obs_type = obs_type
        self.render_mode = render_mode
        self.joint_type = joint_type

        # Rendering settings
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Physics parameters
        self.control_hz = control_hz
        self.physics_hz = physics_hz
        self.dt = 1.0 / physics_hz
        self.physics_steps_per_control = physics_hz // control_hz

        # Configuration
        self.gripper_config = GripperConfig()
        self.object_config = ObjectConfig()

        # Managers (initialized in _setup)
        self.space: Optional[pymunk.Space] = None
        self.ee_manager: Optional[EndEffectorManager] = None
        self.object_manager: Optional[ObjectManager] = None
        self.reward_manager: Optional[RewardManager] = None

        # State
        self.goal_pose: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None

        # Rendering
        self.window = None
        self.clock = None

        # Initialize spaces and physics
        self._initialize_spaces()
        self._setup()

    def _initialize_spaces(self):
        """Initialize action and observation spaces."""
        max_force = 100.0
        max_torque = 50.0
        max_velocity = 500.0
        max_angular_velocity = 10.0

        # Action: wrench for both grippers [tau, fx, fy] × 2
        self.action_space = spaces.Box(
            low=np.array([-max_torque, -max_force, -max_force] * 2),
            high=np.array([max_torque, max_force, max_force] * 2),
            dtype=np.float32
        )

        if self.obs_type == "state":
            self.observation_space = spaces.Dict({
                'ee_poses': spaces.Box(
                    low=np.array([[0, 0, -np.pi]] * 2),
                    high=np.array([[512, 512, np.pi]] * 2),
                    dtype=np.float32
                ),
                'ee_twists': spaces.Box(
                    low=np.array([[-max_velocity, -max_velocity, -max_angular_velocity]] * 2),
                    high=np.array([[max_velocity, max_velocity, max_angular_velocity]] * 2),
                    dtype=np.float32
                ),
                'ee_body_twists': spaces.Box(
                    low=np.array([[-max_angular_velocity, -max_velocity, -max_velocity]] * 2),
                    high=np.array([[max_angular_velocity, max_velocity, max_velocity]] * 2),
                    dtype=np.float32
                ),
                'link_poses': spaces.Box(
                    low=np.array([[0, 0, -np.pi]] * 2),
                    high=np.array([[512, 512, np.pi]] * 2),
                    dtype=np.float32
                ),
                'external_wrenches': spaces.Box(
                    low=np.array([[-max_torque, -max_force, -max_force]] * 2),
                    high=np.array([[max_torque, max_force, max_force]] * 2),
                    dtype=np.float32
                )
            })
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def _setup(self):
        """Setup physics simulation."""
        if self.space is not None:
            return

        # Create physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.95

        # Add boundary walls
        walls = [
            pymunk.Segment(self.space.static_body, (5, 506), (5, 5), 2),
            pymunk.Segment(self.space.static_body, (5, 5), (506, 5), 2),
            pymunk.Segment(self.space.static_body, (506, 5), (506, 506), 2),
            pymunk.Segment(self.space.static_body, (5, 506), (506, 506), 2),
        ]
        for wall in walls:
            wall.color = pygame.Color("LightGray")
            wall.friction = 0.5
        self.space.add(*walls)

        # Create managers
        self.ee_manager = EndEffectorManager(
            space=self.space,
            config=self.gripper_config,
            dt=self.dt
        )

        if self.joint_type != "none":
            self.object_manager = ObjectManager(
                space=self.space,
                joint_type=self.joint_type,
                config=self.object_config
            )
        else:
            self.object_manager = None

        self.reward_manager = RewardManager(
            weights=RewardWeights(),
            success_threshold_pos=20.0,
            success_threshold_angle=0.2,
            max_wrench_threshold=200.0
        )

    def step(self, action: np.ndarray):
        """
        Step environment.

        Args:
            action: [left_tau, left_fx, left_fy, right_tau, right_fx, right_fy]
        """
        self._last_action = action
        
        # Parse wrenches
        action = np.asarray(action)
        if action.shape == (2, 3):
            wrenches = action
        elif action.shape == (6,):
            wrenches = action.reshape(2, 3)
        else:
            raise ValueError(f"Invalid action shape: {action.shape}")

        # Physics simulation
        for _ in range(self.physics_steps_per_control):
            self.ee_manager.apply_wrenches(wrenches)
            self.ee_manager.apply_grip_forces()
            self.space.step(self.dt)

        # Update wrench measurements
        control_dt = self.dt * self.physics_steps_per_control
        self.ee_manager.update_external_wrenches(control_dt)

        # Get current states
        ee_poses = self.ee_manager.get_poses()
        link_poses = self.object_manager.get_link_poses() if self.object_manager else ee_poses.copy()
        external_wrenches = self.ee_manager.get_external_wrenches()

        # Compute reward
        reward_info = self._compute_reward(link_poses, wrenches, external_wrenches)

        # Check safety and termination
        is_safe, safety_msg = self._check_safety()
        
        if not is_safe:
            terminated = True
            reward_info["total_reward"] -= 100.0
            info = {"safety_violation": safety_msg}
        else:
            terminated = reward_info["is_success"] or reward_info["is_failure"]
            info = {}

        info.update({
            "is_success": reward_info["is_success"],
            "is_failure": reward_info["is_failure"],
        })

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), reward_info["total_reward"], terminated, False, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Randomize goal pose
        self.goal_pose = np.array([
            self.np_random.integers(200, 312),
            self.np_random.integers(200, 312),
            self.np_random.uniform(-np.pi/2, np.pi/2)
        ], dtype=np.float32)

        if self.object_manager:
            # Initialize object
            initial_pose = np.array([
                self.np_random.integers(220, 292),
                self.np_random.integers(220, 292),
                self.np_random.uniform(-np.pi/4, np.pi/4)
            ])
            self.object_manager.reset(initial_pose)

            # Let object settle
            for _ in range(20):
                self.space.step(self.dt)

            # Initialize grippers at grasping poses
            grasping_poses = self.object_manager.get_grasping_poses()
            ee_poses = np.array([grasping_poses["left"], grasping_poses["right"]])
            self.ee_manager.reset(ee_poses)

            # Let grippers settle
            for _ in range(50):
                self.ee_manager.apply_grip_forces()
                self.space.step(self.dt)
            self.ee_manager.update_external_wrenches(self.dt * 50)
        else:
            # No object: random gripper positions
            ee_poses = np.array([
                [self.np_random.integers(100, 200),
                 self.np_random.integers(200, 300),
                 self.np_random.uniform(-np.pi, np.pi)],
                [self.np_random.integers(312, 412),
                 self.np_random.integers(200, 300),
                 self.np_random.uniform(-np.pi, np.pi)]
            ])
            self.ee_manager.reset(ee_poses)
            
            for _ in range(10):
                self.space.step(self.dt)
            self.ee_manager.update_external_wrenches(self.dt * 10)

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), {"is_success": False}

    def get_obs(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Get observation."""
        if self.obs_type == "state":
            ee_poses = self.ee_manager.get_poses()
            link_poses = self.object_manager.get_link_poses() if self.object_manager else ee_poses.copy()
            
            return {
                'ee_poses': ee_poses.astype(np.float32),
                'ee_twists': self.ee_manager.get_velocities().astype(np.float32),
                'ee_body_twists': self.ee_manager.get_body_twists().astype(np.float32),
                'link_poses': link_poses.astype(np.float32),
                'external_wrenches': self.ee_manager.get_external_wrenches().astype(np.float32)
            }
        else:
            return self._render_frame()

    def get_joint_axis_screws(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get joint axis screws for screw-decomposed control."""
        if self.object_manager is None:
            return None
        return self.object_manager.get_joint_axis_screws()

    def _compute_reward(self, current_poses, wrenches, external_wrenches) -> Dict:
        """Compute reward using RewardManager."""
        if self.object_manager:
            link2_goal = self._compute_link2_goal(self.goal_pose)
            desired_poses = np.array([self.goal_pose, link2_goal])
            velocities = self.object_manager.get_link_velocities()
        else:
            desired_poses = np.array([self.goal_pose, self.goal_pose])
            velocities = self.ee_manager.get_velocities()

        return self.reward_manager.compute_reward(
            current_poses=current_poses,
            desired_poses=desired_poses,
            current_velocities=velocities,
            desired_velocities=np.zeros((2, 3)),
            applied_wrenches=wrenches,
            external_wrenches=external_wrenches
        )

    def _compute_link2_goal(self, link1_goal: np.ndarray, joint_state: float = 0.0) -> np.ndarray:
        """Compute link2 goal pose from link1 goal."""
        cfg = self.object_config
        joint_type = self.object_manager.joint_type

        if joint_type == JointType.REVOLUTE:
            link2_theta = link1_goal[2] + joint_state
            joint_x = link1_goal[0] + (cfg.link_length / 2) * np.cos(link1_goal[2])
            joint_y = link1_goal[1] + (cfg.link_length / 2) * np.sin(link1_goal[2])
            link2_x = joint_x + (cfg.link_length / 2) * np.cos(link2_theta)
            link2_y = joint_y + (cfg.link_length / 2) * np.sin(link2_theta)
            return np.array([link2_x, link2_y, link2_theta])

        elif joint_type == JointType.PRISMATIC:
            link2_x = link1_goal[0] + joint_state * np.cos(link1_goal[2])
            link2_y = link1_goal[1] + joint_state * np.sin(link1_goal[2])
            return np.array([link2_x, link2_y, link1_goal[2]])

        else:  # FIXED
            offset_x = cfg.link_length * np.cos(link1_goal[2])
            offset_y = cfg.link_length * np.sin(link1_goal[2])
            return np.array([link1_goal[0] + offset_x, link1_goal[1] + offset_y, link1_goal[2]])

    def _check_safety(self) -> Tuple[bool, str]:
        """Check safety constraints."""
        ws_min, ws_max = 10.0, 502.0

        # Check gripper positions
        for i, pose in enumerate(self.ee_manager.get_poses()):
            if not (ws_min <= pose[0] <= ws_max and ws_min <= pose[1] <= ws_max):
                return False, f"EE {i} out of workspace"

        if self.object_manager is None:
            return True, ""

        # Check joint limits
        joint_state = self.object_manager.get_joint_state()
        joint_type = self.object_manager.joint_type

        if joint_type == JointType.REVOLUTE:
            if not (-np.pi <= joint_state <= np.pi):
                return False, f"Joint angle out of limits: {np.rad2deg(joint_state):.1f}°"
                
        elif joint_type == JointType.PRISMATIC:
            limit = self.object_config.link_length * 0.5
            if not (-limit <= joint_state <= limit):
                return False, f"Joint position out of limits: {joint_state:.1f}"

        # Check link positions
        for i, pose in enumerate(self.object_manager.get_link_poses()):
            if not (ws_min <= pose[0] <= ws_max and ws_min <= pose[1] <= ws_max):
                return False, f"Link {i} out of workspace"

        return True, ""

    def _draw(self) -> pygame.Surface:
        """Draw environment to pygame surface (for external rendering)."""
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))

        # Draw goal
        if self.goal_pose is not None:
            gx, gy, _ = self.goal_pose
            goal_rect = pygame.Rect(int(gx) - 30, int(gy) - 30, 60, 60)
            pygame.draw.rect(screen, pygame.Color("LightGreen"), goal_rect, 2)

        # Draw physics objects
        draw_options = DrawOptions(screen)
        self.space.debug_draw(draw_options)

        return screen

    def _render_frame(self, visualize: bool = False) -> np.ndarray:
        """Render environment to image."""
        width = self.visualization_width if visualize else self.observation_width
        height = self.visualization_height if visualize else self.observation_height

        # Create surface
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))

        # Draw goal
        gx, gy, _ = self.goal_pose
        goal_rect = pygame.Rect(int(gx) - 30, int(gy) - 30, 60, 60)
        pygame.draw.rect(screen, pygame.Color("LightGreen"), goal_rect, 2)

        # Draw physics objects
        draw_options = DrawOptions(screen)
        self.space.debug_draw(draw_options)

        # Convert to numpy and resize
        img = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
        return cv2.resize(img, (width, height))

    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame(visualize=True)
            
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            screen = pygame.Surface((512, 512))
            screen.fill((255, 255, 255))
            
            # Draw goal
            gx, gy, _ = self.goal_pose
            pygame.draw.rect(screen, pygame.Color("LightGreen"), 
                           pygame.Rect(int(gx) - 30, int(gy) - 30, 60, 60), 2)
            
            # Draw physics
            draw_options = DrawOptions(screen)
            self.space.debug_draw(draw_options)

            self.window.blit(screen, screen.get_rect())
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()
            
        return None

    def close(self):
        """Close environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
