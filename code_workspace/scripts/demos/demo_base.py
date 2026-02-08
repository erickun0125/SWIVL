"""
Base Demo Class for Bimanual Manipulation

Provides common functionality for all low-level controller demos:
- AutoVelocityController for goal-directed motion
- Environment setup and reset
- Rendering and info overlay
- Main loop structure

Each demo subclass only needs to implement controller-specific methods.
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Add project root to path for direct script execution
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pygame

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.se2_math import normalize_angle
from src.bimanual_utils import compute_constrained_velocity


class AutoVelocityController:
    """
    Automatic velocity controller that generates velocity commands toward a goal.
    Generates smooth trajectories using proportional control with saturation.
    """
    
    def __init__(self, linear_speed=30.0, angular_speed=1.0, joint_speed=0.5):
        """
        Initialize automatic velocity controller.
        
        Args:
            linear_speed: Maximum linear velocity magnitude (pixels/s)
            angular_speed: Maximum angular velocity magnitude (rad/s)
            joint_speed: Maximum joint velocity magnitude
        """
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.joint_speed = joint_speed
        self.velocity = np.zeros(3)  # [vx, vy, omega] world frame
        self.joint_velocity = 0.0
    
    def compute_velocity_toward_goal(self, current_pose, goal_pose, gain=0.5):
        """
        Compute velocity command to move from current pose toward goal pose.
        
        Args:
            current_pose: Current [x, y, theta]
            goal_pose: Goal [x, y, theta]
            gain: Proportional gain
        
        Returns:
            velocity: [vx, vy, omega] in world frame
        """
        # Position error
        pos_error = goal_pose[:2] - current_pose[:2]
        pos_dist = np.linalg.norm(pos_error)
        
        # Proportional control with saturation
        if pos_dist > 0.5:
            vel_mag = min(max(pos_dist * gain, self.linear_speed * 0.5), self.linear_speed)
            linear_vel = (pos_error / pos_dist) * vel_mag
        else:
            linear_vel = np.zeros(2)
        
        # Angular velocity
        ang_error = normalize_angle(goal_pose[2] - current_pose[2])
        angular_vel = np.clip(ang_error * gain, -self.angular_speed, self.angular_speed)
        
        self.velocity = np.array([linear_vel[0], linear_vel[1], angular_vel])
        return self.velocity
    
    def compute_joint_velocity_toward_goal(self, current_q, goal_q, gain=0.5):
        """Compute joint velocity toward goal joint state."""
        q_error = goal_q - current_q
        self.joint_velocity = np.clip(q_error * gain, -self.joint_speed, self.joint_speed)
        return self.joint_velocity
    
    def get_velocity_array(self):
        return self.velocity.copy()
    
    def get_joint_velocity(self):
        return self.joint_velocity


@dataclass
class DemoStats:
    """Statistics tracked during demo execution."""
    step_count: int = 0
    total_reward: float = 0.0
    episode_count: int = 0
    total_fighting_force: float = 0.0
    total_wrench_magnitude: float = 0.0


class BaseBimanualDemo(ABC):
    """
    Abstract base class for bimanual manipulation demos.
    
    Subclasses must implement:
    - _create_controller(): Create the low-level controller
    - _compute_wrenches(): Compute control wrenches from state
    - controller_name: Property returning controller display name
    
    Optionally override:
    - _on_reset(): Additional reset logic
    - _get_controller_info(): Additional info for display
    """
    
    # Controller-specific color for title
    TITLE_COLOR = (0, 0, 128)  # Override in subclass
    
    def __init__(self, joint_type='revolute', render_mode='human'):
        self.joint_type = joint_type
        self.render_mode = render_mode
        
        # Map joint type string to enum
        self.joint_type_map = {
            'revolute': JointType.REVOLUTE,
            'prismatic': JointType.PRISMATIC,
            'fixed': JointType.FIXED
        }
        self.joint_enum = self.joint_type_map[joint_type]
        
        # Create environment
        print(f"Creating BiArt environment with {joint_type} joint...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode=render_mode,
            joint_type=joint_type
        )
        
        # Create controller (implemented by subclass)
        self.controller = self._create_controller()
        
        # Auto velocity controller
        joint_speed = 0.5
        if self.joint_enum == JointType.PRISMATIC:
            joint_speed *= 5.0
        self.auto_controller = AutoVelocityController(
            linear_speed=30.0,
            angular_speed=1.0,
            joint_speed=joint_speed
        )
        
        self.controlled_ee_idx = 0
        self.desired_poses = None
        self.goal_ee_poses = None
        self.goal_joint_state = None
        
        # Display settings
        self.font = None
        self.font_small = None
        
        # Statistics
        self.stats = DemoStats()
        
        # Last control info (for display)
        self.last_control_info: Optional[Dict] = None
        
        # Initialize
        self.reset_environment()
    
    @property
    @abstractmethod
    def controller_name(self) -> str:
        """Display name for the controller."""
        pass
    
    @abstractmethod
    def _create_controller(self):
        """Create and return the low-level controller."""
        pass
    
    @abstractmethod
    def _compute_wrenches(
        self,
        state: Dict,
        desired_poses: np.ndarray,
        desired_velocities: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Compute control wrenches.
        
        Args:
            state: Current state dictionary
            desired_poses: (2, 3) desired poses
            desired_velocities: (2, 3) desired velocities [vx, vy, omega] world frame
            
        Returns:
            wrenches: (2, 3) control wrenches [τ, fx, fy]
            info: Optional info dictionary for display
        """
        pass
    
    def _on_reset(self):
        """Called after environment reset. Override for controller-specific logic."""
        pass
    
    def _get_controller_info(self, info: Optional[Dict]) -> Dict[str, str]:
        """
        Get controller-specific info for display.
        
        Args:
            info: Info dictionary from _compute_wrenches
            
        Returns:
            Dictionary of label -> value strings
        """
        return {}
    
    def reset_environment(self):
        """Reset environment and controllers."""
        print(f"\nResetting environment (Episode {self.stats.episode_count + 1})...")
        
        obs, _ = self.env.reset()
        
        self.desired_poses = obs['ee_poses'].copy()
        self.goal_ee_poses = self.env.goal_manager.get_goal_ee_poses()
        self.goal_joint_state = self.env.goal_manager.get_goal_joint_state()
        
        self.controller.reset()
        
        # Reset stats
        self.stats.step_count = 0
        self.stats.total_reward = 0.0
        self.stats.total_fighting_force = 0.0
        self.stats.total_wrench_magnitude = 0.0
        self.stats.episode_count += 1
        
        self.last_control_info = None
        
        # Controller-specific reset
        self._on_reset()
        
        print("Environment reset complete!")
    
    def get_current_state(self, obs) -> Dict:
        """Extract state from observation."""
        state = {
            'ee_poses': obs['ee_poses'],
            'ee_velocities': obs['ee_velocities'],
            'external_wrenches': obs['external_wrenches'],
            'link_poses': obs['link_poses'],
            'joint_state': self.env.object_manager.get_joint_state(),
        }
        
        # Add body twists if available
        if 'ee_body_twists' in obs:
            state['ee_body_twists'] = obs['ee_body_twists']
        
        return state
    
    @staticmethod
    def point_velocity_to_body_twist(pose, point_velocity):
        """
        Convert world frame point velocity [vx, vy, omega] to body twist [omega, vx_b, vy_b].
        
        Uses simple rotation (NOT adjoint map) since this is point velocity,
        not SE(2) twist observed from different points.
        """
        vx_world, vy_world, omega = point_velocity
        theta = pose[2]
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        vx_body = cos_theta * vx_world + sin_theta * vy_world
        vy_body = -sin_theta * vx_world + cos_theta * vy_world
        
        return np.array([omega, vx_body, vy_body])
    
    def compute_fighting_force(self, external_wrenches: np.ndarray) -> float:
        """
        Compute fighting force from external wrenches.
        
        For controllers without screw decomposition, uses total wrench magnitude.
        Subclasses with screw decomposition should override this.
        """
        return np.sum([np.linalg.norm(w[1:3]) for w in external_wrenches])
    
    def draw_info_overlay(self, screen, state: Dict):
        """Draw unified information overlay."""
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 14, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 12)
        
        y_offset = 10
        line_height = 18
        x_margin = 10
        
        def draw_text(text, y, font=None, color=(0, 0, 0)):
            if font is None:
                font = self.font_small
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (x_margin, y)
            bg_surface = pygame.Surface((text_rect.width + 10, text_rect.height + 4))
            bg_surface.set_alpha(200)
            bg_surface.fill((255, 255, 255))
            screen.blit(bg_surface, (x_margin - 5, y - 2))
            screen.blit(text_surface, text_rect)
            return y + line_height
        
        # Title
        title = f"=== {self.controller_name} ({self.joint_type.upper()}) ==="
        y_offset = draw_text(title, y_offset, self.font, self.TITLE_COLOR)
        y_offset += 5
        
        # Basic stats
        y_offset = draw_text(f"Step: {self.stats.step_count} | Reward: {self.stats.total_reward:.2f}", y_offset)
        
        # Goal error
        err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
        err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
        color = (200, 0, 0) if err_left > 50 else (0, 100, 0)
        y_offset = draw_text(f"Goal Error: L={err_left:.1f}px R={err_right:.1f}px", y_offset, color=color)
        
        y_offset += 5
        
        # Fighting force (common to all controllers)
        fighting_force = self.compute_fighting_force(state['external_wrenches'])
        avg_ff = self.stats.total_fighting_force / max(1, self.stats.step_count)
        color = (200, 0, 0) if fighting_force > 20 else (0, 100, 0)
        y_offset = draw_text(f"Fighting Force: {fighting_force:.2f} N (avg: {avg_ff:.2f})", y_offset, color=color)
        
        y_offset += 5
        
        # EE info with external wrenches
        for i in range(2):
            pose = state['ee_poses'][i]
            y_offset = draw_text(f"EE {i}: x={pose[0]:.1f} y={pose[1]:.1f} θ={pose[2]:.2f}", y_offset)
            
            wrench = state['external_wrenches'][i]
            y_offset = draw_text(
                f"  F_ext: τ={wrench[0]:.1f} fx={wrench[1]:.1f} fy={wrench[2]:.1f}",
                y_offset, color=(100, 0, 100)
            )
        
        # Controller-specific info
        controller_info = self._get_controller_info(self.last_control_info)
        if controller_info:
            y_offset += 5
            for label, value in controller_info.items():
                y_offset = draw_text(f"{label}: {value}", y_offset, color=(0, 100, 100))
        
        y_offset += 10
        y_offset = draw_text("R: Reset | ESC: Exit", y_offset, color=(100, 100, 100))
    
    def run(self):
        """Main demo loop."""
        print("\n" + "=" * 60)
        print(f"Starting {self.controller_name}")
        print("=" * 60)
        print("Controls: R = Reset, ESC = Exit")
        print("=" * 60)
        
        pygame.init()
        running = True
        clock = pygame.time.Clock()
        
        while running:
            pygame.event.pump()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_environment()
            
            obs = self.env.get_obs()
            state = self.get_current_state(obs)
            current_q = state['joint_state']
            
            # Compute desired velocity toward goal
            controlled_velocity = self.auto_controller.compute_velocity_toward_goal(
                state['ee_poses'][self.controlled_ee_idx],
                self.goal_ee_poses[self.controlled_ee_idx],
                gain=0.5
            )
            joint_velocity_cmd = self.auto_controller.compute_joint_velocity_toward_goal(
                current_q,
                self.goal_joint_state,
                gain=0.5
            )
            
            dt = 1.0 / self.env.control_hz
            other_ee_idx = 1 - self.controlled_ee_idx
            
            # Other EE velocity from kinematic constraint
            other_velocity = compute_constrained_velocity(
                self.env,
                self.controlled_ee_idx,
                controlled_velocity,
                joint_velocity_cmd
            )
            
            # Update desired poses
            self.desired_poses[self.controlled_ee_idx] += np.array([
                controlled_velocity[0] * dt,
                controlled_velocity[1] * dt,
                controlled_velocity[2] * dt
            ])
            self.desired_poses[other_ee_idx] += np.array([
                other_velocity[0] * dt,
                other_velocity[1] * dt,
                other_velocity[2] * dt
            ])
            self.desired_poses[:, :2] = np.clip(self.desired_poses[:, :2], 20.0, 492.0)
            
            # Prepare desired velocities
            desired_velocities = np.zeros((2, 3))
            desired_velocities[self.controlled_ee_idx] = controlled_velocity
            desired_velocities[other_ee_idx] = other_velocity
            
            # Compute wrenches (controller-specific)
            wrenches, info = self._compute_wrenches(state, self.desired_poses, desired_velocities)
            self.last_control_info = info
            
            # Step environment
            action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Update stats
            self.stats.step_count += 1
            self.stats.total_reward += reward
            self.stats.total_fighting_force += self.compute_fighting_force(state['external_wrenches'])
            self.stats.total_wrench_magnitude += np.sum([np.linalg.norm(w) for w in wrenches])
            
            # Render
            if self.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption(f"SWIVL - {self.controller_name}")
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()
                
                screen_surface = self.env._draw()
                self.env.window.blit(screen_surface, screen_surface.get_rect())
                self.draw_info_overlay(self.env.window, state)
                pygame.display.flip()
            
            # Print periodic status
            if self.stats.step_count % 100 == 0:
                err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
                err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
                avg_ff = self.stats.total_fighting_force / self.stats.step_count
                print(f"[Step {self.stats.step_count:4d}] Goal Error: L={err_left:.1f}px R={err_right:.1f}px | "
                      f"Avg FF: {avg_ff:.2f} N | q={current_q:.2f}")
            
            clock.tick(self.env.metadata['render_fps'])
            
            if terminated or truncated:
                print(f"\nEpisode finished!")
                self.reset_environment()
        
        self.env.close()
        pygame.quit()
        print("\nDemo completed!")


def run_demo(demo_class, description: str):
    """Common entry point for demos."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('joint_type', nargs='?', default='revolute',
                        choices=['revolute', 'prismatic', 'fixed'])
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(description.upper())
    print("=" * 60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print("=" * 60)
    
    demo = demo_class(joint_type=args.joint_type, render_mode='human')
    demo.run()

