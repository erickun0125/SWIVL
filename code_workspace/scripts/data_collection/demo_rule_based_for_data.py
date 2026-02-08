"""
Rule-based Demo for Bimanual Manipulation with Data Collection

Automatically moves object from initial pose to goal pose using kinematic constraints,
while collecting demonstration data for imitation learning.

Usage:
    python demo_rule_based_for_data.py [joint_type] [--num-episodes N] [--data-dir DIR]

Arguments:
    joint_type: 'revolute', 'prismatic', or 'fixed' (default: revolute)
    --num-episodes: Number of episodes to collect (default: 10)
    --data-dir: Directory to save data (default: data/demos)
"""

import argparse
import numpy as np
import pygame
import h5py
import cv2
import os
from datetime import datetime
from typing import Dict, Optional

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains
from src.se2_math import normalize_angle
from src.bimanual_utils import compute_constrained_velocity
from src.data_utils import DataCollector

# Import AutoVelocityController from demo_base (canonical location)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from demos.demo_base import AutoVelocityController


class RuleBasedDataDemo:
    """Rule-based demo for bimanual manipulation with data collection."""

    def __init__(self, joint_type='revolute', render_mode='human', data_dir='data/demos'):
        """
        Initialize demo.
        """
        self.joint_type = joint_type
        self.render_mode = render_mode
        
        print(f"Data collection enabled. Saving to {data_dir}")
        self.collector = DataCollector(data_dir, filename_prefix="rule_based", demo_type="rule_based")

        print(f"Creating BiArt environment with {joint_type} joint...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode=render_mode,
            joint_type=joint_type
        )

        # Map joint type string to enum
        joint_type_map = {
            'revolute': JointType.REVOLUTE,
            'prismatic': JointType.PRISMATIC,
            'fixed': JointType.FIXED
        }
        self.joint_enum = joint_type_map[joint_type]

        # PD controller
        gains = PDGains(
            kp_linear=1500.0,
            kd_linear=1500.0,
            kp_angular=800.0,
            kd_angular=1000.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Create automatic velocity controller
        joint_speed = 0.75
        if self.joint_enum == JointType.PRISMATIC:
            joint_speed *= 5.0
        self.auto_controller = AutoVelocityController(
            linear_speed=21.0,
            angular_speed=0.5,
            joint_speed=joint_speed
        )
        
        self.controlled_ee_idx = 0
        self.desired_poses = None
        self.goal_ee_poses = None
        self.goal_joint_state = 0.0

        # Statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count = 0
        self.saved_files = []

    def reset_environment(self):
        """Reset environment and controllers."""
        print(f"\nResetting environment (Episode {self.episode_count + 1})...")

        # Reset environment
        obs, info = self.env.reset()

        # Initialize desired poses to current EE poses
        self.desired_poses = obs['ee_poses'].copy()
        
        # Get goal EE poses from goal_manager
        self.goal_ee_poses = self.env.goal_manager.get_goal_ee_poses()
        self.goal_joint_state = self.env.goal_manager.get_goal_joint_state()

        # Reset controller
        self.controller.reset()

        # Reset statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        print("Environment reset complete!")

    def get_current_state(self, obs):
        """Extract state from observation."""
        current_ee_poses = obs['ee_poses']
        current_velocities = obs['ee_velocities']
        external_wrenches = obs['external_wrenches']
        link_poses = obs['link_poses']
        joint_state = self.env.object_manager.get_joint_state()

        return {
            'ee_poses': current_ee_poses,
            'ee_velocities': current_velocities,
            'external_wrenches': external_wrenches,
            'joint_state': joint_state,
            'link_poses': link_poses
        }

    def run(self, num_episodes=10, max_steps_per_episode=500):
        """Run the rule-based demo and collect data."""
        print("\n" + "="*60)
        print("Starting Rule-based Data Collection Demo")
        print("="*60)
        print(f"Collecting {num_episodes} episodes...")

        pygame.init()
        clock = pygame.time.Clock()

        # Initialize first episode
        self.reset_environment()

        episodes_collected = 0
        running = True

        while running and episodes_collected < num_episodes:
            pygame.event.pump()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            obs = self.env.get_obs()
            state = self.get_current_state(obs)
            current_q = state['joint_state']
            
            # Capture image for data collection
            img_obs = self.env._render_frame(visualize=False)
            obs['joint_state'] = current_q
            
            # Compute velocity toward goal (automatic control)
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
                controlled_velocity[0] * dt, controlled_velocity[1] * dt, controlled_velocity[2] * dt
            ])
            self.desired_poses[other_ee_idx] += np.array([
                other_velocity[0] * dt, other_velocity[1] * dt, other_velocity[2] * dt
            ])
            self.desired_poses[:, :2] = np.clip(self.desired_poses[:, :2], 20.0, 492.0)

            desired_velocities = np.zeros((2, 3))
            desired_velocities[self.controlled_ee_idx] = controlled_velocity
            desired_velocities[other_ee_idx] = other_velocity

            wrenches = self.controller.compute_wrenches(
                state['ee_poses'], self.desired_poses, desired_velocities,
                current_velocities=state['ee_velocities']
            )

            action = np.concatenate([wrenches[0], wrenches[1]])
            
            # Collect data before step
            self.collector.add_step(obs, self.desired_poses.copy(), img_obs)
            
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update statistics
            self.step_count += 1
            self.total_reward += reward

            # Render
            if self.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption("SWIVL Rule-based Data Collection")
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()

                screen_surface = self.env._draw()
                self.env.window.blit(screen_surface, screen_surface.get_rect())
                pygame.display.flip()

            # Print periodic status
            if self.step_count % 100 == 0:
                err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
                err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
                print(f"[Ep {episodes_collected+1}/{num_episodes}] "
                      f"Step {self.step_count}: L_err={err_left:.1f}px R_err={err_right:.1f}px")

            clock.tick(self.env.metadata['render_fps'])

            # Handle episode termination
            is_timeout = self.step_count >= max_steps_per_episode
            episode_done = terminated or truncated or is_timeout
            
            if episode_done:
                is_success = info.get('is_success', False)
                print(f"\nEpisode attempt finished!")
                print(f"  Total steps: {self.step_count}")
                print(f"  Success: {is_success}")
                print(f"  Timeout: {is_timeout}")
                
                # Only save successful episodes (skip timeout failures)
                if is_timeout and not is_success:
                    print(f"  ⚠️ Timeout without success - discarding episode data")
                    self.collector.reset_buffer()
                else:
                    # Save episode data
                    episodes_collected += 1
                    filepath = self.collector.save_episode(episodes_collected, self.joint_type)
                    if filepath:
                        self.saved_files.append(filepath)
                        print(f"  ✓ Episode {episodes_collected} saved")
                
                if episodes_collected < num_episodes:
                    self.reset_environment()

        # Cleanup
        self.env.close()
        pygame.quit()
        
        print("\n" + "="*60)
        print("Data Collection Complete")
        print("="*60)
        print(f"Episodes collected: {episodes_collected}")
        print(f"Files saved: {len(self.saved_files)}")
        for f in self.saved_files:
            print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description='Rule-based demo with data collection')
    parser.add_argument('joint_type', nargs='?', default='revolute',
                        choices=['revolute', 'prismatic', 'fixed'])
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to collect')
    parser.add_argument('--data-dir', default='data/demos',
                        help='Directory to save data')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode (timeout)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("RULE-BASED DATA COLLECTION DEMO")
    print("="*60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Timeout: {args.max_steps} steps")
    print("(Episodes that timeout without success will be discarded)")
    print("="*60)

    demo = RuleBasedDataDemo(
        joint_type=args.joint_type, 
        render_mode='human',
        data_dir=args.data_dir
    )
    demo.run(num_episodes=args.num_episodes, max_steps_per_episode=args.max_steps)


if __name__ == "__main__":
    main()


