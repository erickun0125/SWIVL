"""
Interactive Teleoperation Demo for Bimanual Manipulation with Data Collection

This demo integrates all components:
- Linkage object management (R, P, Fixed joints)
- PD controller for pose tracking
- Keyboard teleoperation with kinematic constraints
- Real-time visualization with wrench display
- Data collection for imitation learning

Usage:
    python demo_teleoperation_for_data.py [joint_type] [--save-data] [--data-dir DIR]

Arguments:
    joint_type: 'revolute', 'prismatic', or 'fixed' (default: revolute)

Keyboard Controls:
    Arrow Keys: Move controlled end-effector (linear velocity)
    Q/W: Rotate counterclockwise/clockwise (angular velocity)
    A/S: Object joint velocity (negative/positive)
    1/2: Switch controlled end-effector (0 or 1)
    Space: Reset velocity to zero
    R: Reset environment (saves current episode if data collection enabled)
    H: Toggle help display
    ESC: Exit
"""

import argparse
import numpy as np
import pygame
import h5py
import os
from datetime import datetime
from typing import Dict, Optional

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains
from src.bimanual_utils import compute_constrained_velocity
from src.data_utils import DataCollector

# Import KeyboardVelocityController from demo_keyboard_teleoperation (canonical location)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from demos.demo_keyboard_teleoperation import KeyboardVelocityController


class TeleoperationDemo:
    """Teleoperation demo for bimanual manipulation with data collection."""

    def __init__(self, joint_type='revolute', render_mode='human', save_data=False, data_dir='data/demos'):
        """
        Initialize demo.
        """
        self.joint_type = joint_type
        self.render_mode = render_mode
        self.save_data = save_data
        
        if self.save_data:
            print(f"Data collection enabled. Saving to {data_dir}")
            self.collector = DataCollector(data_dir, filename_prefix="teleop", demo_type="teleoperation")
        else:
            self.collector = None

        print(f"Creating BiArt environment with {joint_type} joint...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode=render_mode,
            joint_type=joint_type
        )

        joint_type_map = {
            'revolute': JointType.REVOLUTE,
            'prismatic': JointType.PRISMATIC,
            'fixed': JointType.FIXED
        }
        self.joint_enum = joint_type_map[joint_type]

        gains = PDGains(
            kp_linear=1500.0,
            kd_linear=1000.0,
            kp_angular=750.0,
            kd_angular=500.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        joint_speed = 0.5
        if self.joint_enum == JointType.PRISMATIC:
            joint_speed *= 5.0

        self.keyboard = KeyboardVelocityController(linear_speed=20.0, angular_speed=0.5, joint_speed=joint_speed)
        
        self.controlled_ee_idx = 0
        self.desired_poses = None

        self.show_help = True
        self.font = None
        self.font_small = None

        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count = 0
        self.saved_files = []

        self.reset_environment()

    def reset_environment(self):
        """Reset environment and controllers."""
        print(f"\nResetting environment (Episode {self.episode_count + 1})...")

        obs, info = self.env.reset()
        self.desired_poses = obs['ee_poses'].copy()
        self.controller.reset()

        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        print("Environment reset complete!")

    def get_current_state(self, obs):
        """Get current state from observation."""
        return {
            'ee_poses': obs['ee_poses'],
            'ee_velocities': obs['ee_velocities'],
            'external_wrenches': obs['external_wrenches'],
            'joint_state': self.env.object_manager.get_joint_state(),
            'link_poses': obs['link_poses']
        }

    def draw_desired_frames(self, screen):
        """Draw desired pose frames for both grippers."""
        if self.desired_poses is None:
            return
        
        colors = [(65, 105, 225), (220, 20, 60)]
        
        for i, pose in enumerate(self.desired_poses):
            x, y, theta = pose
            
            if not (0 <= x <= 512 and 0 <= y <= 512):
                continue
            
            arrow_length = 25
            arrow_width = 8
            
            jaw_angle = theta + np.pi / 2
            cos_j, sin_j = np.cos(jaw_angle), np.sin(jaw_angle)
            
            tip_x = x + arrow_length * cos_j
            tip_y = y + arrow_length * sin_j
            
            base_left_x = x - arrow_width * sin_j
            base_left_y = y + arrow_width * cos_j
            base_right_x = x + arrow_width * sin_j
            base_right_y = y - arrow_width * cos_j
            
            arrow_points = [
                (int(tip_x), int(tip_y)),
                (int(base_left_x), int(base_left_y)),
                (int(base_right_x), int(base_right_y))
            ]
            
            arrow_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
            color_with_alpha = (*colors[i], 120)
            pygame.draw.polygon(arrow_surface, color_with_alpha, arrow_points)
            screen.blit(arrow_surface, (0, 0))
            
            pygame.draw.polygon(screen, colors[i], arrow_points, 2)
            pygame.draw.circle(screen, colors[i], (int(x), int(y)), 5, 2)

    def draw_info_overlay(self, screen, state):
        """Draw information overlay on the screen."""
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 14, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 12)

        vel_cmd = self.keyboard.get_velocity_array()
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

        y_offset = draw_text(f"=== Teleoperation Demo ({self.joint_type.upper()}) ===", y_offset, self.font, (0, 0, 128))
        
        if self.save_data:
            y_offset = draw_text("[DATA COLLECTION ENABLED]", y_offset, color=(200, 0, 0))
        
        y_offset += 5
        y_offset = draw_text(f"Episode: {self.episode_count} | Step: {self.step_count}", y_offset)
        y_offset = draw_text(f"Controlled EE: {self.controlled_ee_idx} (1/2 to switch)", y_offset, color=(128, 0, 0))
        y_offset = draw_text(f"Velocity: [{vel_cmd[0]:5.1f}, {vel_cmd[1]:5.1f}, {vel_cmd[2]:4.2f}]", y_offset, color=(0, 100, 0))
        
        if self.show_help:
            y_offset += 10
            y_offset = draw_text("Controls: Arrows=Move, Q/W=Rotate, A/S=Joint, R=Reset, ESC=Exit", y_offset, color=(80, 80, 80))

    def run(self):
        """Run the interactive demo."""
        print("\n" + "="*60)
        print("Starting Teleoperation Demo")
        if self.save_data:
            print("DATA COLLECTION ENABLED - Press R to save episode")
        print("="*60)

        pygame.init()
        running = True
        clock = pygame.time.Clock()

        try:
            while running:
                pygame.event.pump()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            if self.collector and len(self.collector.images) > 0:
                                filepath = self.collector.save_episode(self.episode_count, self.joint_type)
                                if filepath:
                                    self.saved_files.append(filepath)
                            self.reset_environment()
                        elif event.key == pygame.K_h:
                            self.show_help = not self.show_help
                        elif event.key == pygame.K_1:
                            self.controlled_ee_idx = 0
                            print("Switched to EE 0 (LEFT)")
                        elif event.key == pygame.K_2:
                            self.controlled_ee_idx = 1
                            print("Switched to EE 1 (RIGHT)")
                        elif event.key == pygame.K_SPACE:
                            self.keyboard.velocity = np.zeros(3)
                            self.keyboard.joint_velocity = 0.0

                keys = pygame.key.get_pressed()
                velocity_cmd = self.keyboard.process_keys(keys)
                joint_velocity_cmd = self.keyboard.get_joint_velocity()

                obs = self.env.get_obs()
                state = self.get_current_state(obs)
                
                # Capture image for data collection
                if self.collector:
                    img_obs = self.env._render_frame(visualize=False)
                    obs['joint_state'] = self.env.object_manager.get_joint_state()

                dt = 1.0 / self.env.control_hz
                other_ee_idx = 1 - self.controlled_ee_idx
                
                controlled_velocity = velocity_cmd
                other_velocity = compute_constrained_velocity(
                    self.env,
                    self.controlled_ee_idx,
                    controlled_velocity,
                    joint_velocity_cmd
                )
                
                self.desired_poses[self.controlled_ee_idx, 0] += controlled_velocity[0] * dt
                self.desired_poses[self.controlled_ee_idx, 1] += controlled_velocity[1] * dt
                self.desired_poses[self.controlled_ee_idx, 2] += controlled_velocity[2] * dt
                
                self.desired_poses[other_ee_idx, 0] += other_velocity[0] * dt
                self.desired_poses[other_ee_idx, 1] += other_velocity[1] * dt
                self.desired_poses[other_ee_idx, 2] += other_velocity[2] * dt

                self.desired_poses[:, :2] = np.clip(self.desired_poses[:, :2], 20.0, 492.0)

                desired_velocities = np.zeros((2, 3))
                desired_velocities[self.controlled_ee_idx] = controlled_velocity
                desired_velocities[other_ee_idx] = other_velocity

                wrenches = self.controller.compute_wrenches(
                    state['ee_poses'],
                    self.desired_poses,
                    desired_velocities,
                    current_velocities=state['ee_velocities']
                )

                action = np.concatenate([wrenches[0], wrenches[1]])
                
                # Collect data
                if self.collector:
                    self.collector.add_step(obs, self.desired_poses.copy(), img_obs)
                
                obs, reward, terminated, truncated, info = self.env.step(action)

                self.step_count += 1
                self.total_reward += reward

                # Render
                if self.render_mode == 'human':
                    if self.env.window is None:
                        self.env.window = pygame.display.set_mode((512, 512))
                        pygame.display.set_caption("SWIVL Teleoperation Demo")
                    if self.env.clock is None:
                        self.env.clock = pygame.time.Clock()

                    screen_surface = self.env._draw()
                    self.draw_desired_frames(screen_surface)
                    self.env.window.blit(screen_surface, screen_surface.get_rect())
                    
                    state = self.get_current_state(obs)
                    self.draw_info_overlay(self.env.window, state)
                    pygame.display.flip()

                if self.step_count % 100 == 0:
                    print(f"[Step {self.step_count:4d}] EE{self.controlled_ee_idx}")

                clock.tick(self.env.metadata['render_fps'])

                if terminated or truncated:
                    print(f"\nEpisode finished!")
                    if self.collector and len(self.collector.images) > 0:
                        filepath = self.collector.save_episode(self.episode_count, self.joint_type)
                        if filepath:
                            self.saved_files.append(filepath)
                    self.reset_environment()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            
        finally:
            self.env.close()
            pygame.quit()
            
            if self.collector and len(self.collector.images) > 0:
                filepath = self.collector.save_episode(self.episode_count, self.joint_type)
                if filepath:
                    self.saved_files.append(filepath)

        print("\n" + "="*60)
        print("Demo Statistics")
        print("="*60)
        print(f"Total episodes: {self.episode_count}")
        if self.saved_files:
            print(f"Files saved: {len(self.saved_files)}")
            for f in self.saved_files:
                print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description='Teleoperation demo with optional data collection')
    parser.add_argument('joint_type', nargs='?', default='revolute',
                        choices=['revolute', 'prismatic', 'fixed'])
    parser.add_argument('--save-data', action='store_true', help='Enable data collection')
    parser.add_argument('--data-dir', default='data/demos', help='Directory to save data')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TELEOPERATION DEMO" + (" WITH DATA COLLECTION" if args.save_data else ""))
    print("="*60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print("="*60)

    demo = TeleoperationDemo(
        joint_type=args.joint_type,
        render_mode='human',
        save_data=args.save_data,
        data_dir=args.data_dir
    )
    demo.run()


if __name__ == "__main__":
    main()




