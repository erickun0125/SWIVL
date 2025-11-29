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


class AutoVelocityController:
    """
    Automatic velocity controller that generates velocity commands toward a goal.
    """
    
    def __init__(self, linear_speed=20.0, angular_speed=0.5, joint_speed=0.5):
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
        self.velocity = np.zeros(3)
        self.joint_velocity = 0.0
    
    def compute_velocity_toward_goal(self, current_pose, goal_pose, gain=0.5):
        """
        Compute velocity command to move from current pose toward goal pose.
        
        Args:
            current_pose: Current [x, y, theta] of controlled EE
            goal_pose: Goal [x, y, theta] of controlled EE
            gain: Proportional gain for velocity computation
        
        Returns:
            velocity: [vx, vy, omega] velocity command
        """
        # Position error
        pos_error = goal_pose[:2] - current_pose[:2]
        pos_dist = np.linalg.norm(pos_error)
        
        # Compute linear velocity (proportional with saturation and minimum speed)
        if pos_dist > 0.5:
            vel_mag = min(max(pos_dist * gain, self.linear_speed * 0.5), self.linear_speed)
            linear_vel = (pos_error / pos_dist) * vel_mag
        else:
            linear_vel = np.zeros(2)
        
        # Compute angular velocity (proportional with saturation)
        ang_error = normalize_angle(goal_pose[2] - current_pose[2])
        angular_vel = np.clip(ang_error * gain, -self.angular_speed, self.angular_speed)
        
        self.velocity = np.array([linear_vel[0], linear_vel[1], angular_vel])
        return self.velocity
    
    def compute_joint_velocity_toward_goal(self, current_q, goal_q, gain=0.5):
        """
        Compute joint velocity command to move toward goal joint state.
        """
        q_error = goal_q - current_q
        self.joint_velocity = np.clip(q_error * gain, -self.joint_speed, self.joint_speed)
        return self.joint_velocity
    
    def get_velocity_array(self):
        """Return current EE velocity as array."""
        return self.velocity.copy()
    
    def get_joint_velocity(self):
        """Return current joint velocity."""
        return self.joint_velocity


class DataCollector:
    """Collects and saves demonstration data."""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reset_buffer()
        
    def reset_buffer(self):
        """Reset data buffer."""
        self.images = []
        self.ee_poses = []
        self.ee_velocities = []
        self.external_wrenches = []
        self.joint_states = []
        self.link_poses = []
        self.actions = []  # Desired poses
        
    def add_step(self, obs: Dict, action: np.ndarray, image: np.ndarray):
        """
        Add a step to the buffer.
        
        Args:
            obs: Observation dictionary
            action: Action taken (desired poses)
            image: Rendered image (H, W, 3)
        """
        self.images.append(image)
        self.ee_poses.append(obs['ee_poses'])
        self.ee_velocities.append(obs['ee_velocities'])
        self.external_wrenches.append(obs['external_wrenches'])
        self.link_poses.append(obs['link_poses'])
        
        # Handle joint state (scalar or array)
        joint_state = obs.get('joint_state', 0.0)
        self.joint_states.append(joint_state)
        
        self.actions.append(action)
        
    def save_episode(self, episode_idx: int, joint_type: str = 'revolute'):
        """Save buffered data to HDF5."""
        if len(self.images) == 0:
            print("No data to save.")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rule_based_{joint_type}_{timestamp}_ep{episode_idx}.h5"
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert lists to arrays
        images = np.array(self.images, dtype=np.uint8)
        ee_poses = np.array(self.ee_poses, dtype=np.float32)
        ee_velocities = np.array(self.ee_velocities, dtype=np.float32)
        external_wrenches = np.array(self.external_wrenches, dtype=np.float32)
        link_poses = np.array(self.link_poses, dtype=np.float32)
        joint_states = np.array(self.joint_states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        
        print(f"Saving episode {episode_idx} with {len(images)} steps to {filepath}...")
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            obs_group = f.create_group('obs')
            action_group = f.create_group('action')
            
            # Save observations
            obs_group.create_dataset('images', data=images, compression='gzip')
            obs_group.create_dataset('ee_poses', data=ee_poses)
            obs_group.create_dataset('ee_velocities', data=ee_velocities)
            obs_group.create_dataset('external_wrenches', data=external_wrenches)
            obs_group.create_dataset('link_poses', data=link_poses)
            obs_group.create_dataset('joint_states', data=joint_states)
            
            # Save actions
            action_group.create_dataset('desired_poses', data=actions)
            
            # Save metadata
            f.attrs['num_steps'] = len(images)
            f.attrs['joint_type'] = joint_type
            f.attrs['demo_type'] = 'rule_based'
            
        print("Save complete.")
        self.reset_buffer()
        return filepath


class RuleBasedDataDemo:
    """Rule-based demo for bimanual manipulation with data collection."""

    def __init__(self, joint_type='revolute', render_mode='human', data_dir='data/demos'):
        """
        Initialize demo.
        """
        self.joint_type = joint_type
        self.render_mode = render_mode
        
        print(f"Data collection enabled. Saving to {data_dir}")
        self.collector = DataCollector(data_dir)

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
        joint_speed = 0.5
        if self.joint_enum == JointType.PRISMATIC:
            joint_speed *= 5.0
        self.auto_controller = AutoVelocityController(
            linear_speed=20.0,
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
            episode_done = terminated or truncated or self.step_count >= max_steps_per_episode
            
            if episode_done:
                print(f"\nEpisode {episodes_collected + 1} finished!")
                print(f"  Total steps: {self.step_count}")
                print(f"  Success: {info.get('is_success', False)}")
                
                # Save episode data
                filepath = self.collector.save_episode(episodes_collected + 1, self.joint_type)
                if filepath:
                    self.saved_files.append(filepath)
                
                episodes_collected += 1
                
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
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("RULE-BASED DATA COLLECTION DEMO")
    print("="*60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Data Dir: {args.data_dir}")
    print("="*60)

    demo = RuleBasedDataDemo(
        joint_type=args.joint_type, 
        render_mode='human',
        data_dir=args.data_dir
    )
    demo.run(num_episodes=args.num_episodes, max_steps_per_episode=args.max_steps)


if __name__ == "__main__":
    main()

