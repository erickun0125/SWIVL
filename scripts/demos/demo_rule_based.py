"""
Rule-based Demo for Bimanual Manipulation
Automatically moves object from initial pose to desired pose using kinematic constraints.

Structure is IDENTICAL to demo_teleoperation.py, but with automatic velocity generation
instead of keyboard input.
"""

import argparse
import numpy as np
import pygame

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains
from src.se2_math import normalize_angle

# Handle import for both direct execution and module import
try:
    from scripts.demos.bimanual_utils import compute_constrained_velocity
except ModuleNotFoundError:
    from bimanual_utils import compute_constrained_velocity


class AutoVelocityController:
    """
    Automatic velocity controller that generates velocity commands toward a goal.
    Replaces KeyboardVelocityController from teleoperation demo.
    """
    
    def __init__(self, linear_speed=10.0, angular_speed=0.5, joint_speed=0.5):
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
        self.velocity = np.zeros(3)  # [vx, vy, omega] for controlled EE
        self.joint_velocity = 0.0    # Joint velocity
    
    def compute_velocity_toward_goal(self, current_pose, goal_pose, gain=0.3):
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
        
        # Compute linear velocity (proportional with saturation)
        if pos_dist > 0.5:
            vel_mag = min(pos_dist * gain, self.linear_speed)
            linear_vel = (pos_error / pos_dist) * vel_mag
        else:
            linear_vel = np.zeros(2)
        
        # Compute angular velocity (proportional with saturation)
        ang_error = normalize_angle(goal_pose[2] - current_pose[2])
        angular_vel = np.clip(ang_error * gain, -self.angular_speed, self.angular_speed)
        
        self.velocity = np.array([linear_vel[0], linear_vel[1], angular_vel])
        return self.velocity
    
    def compute_joint_velocity_toward_goal(self, current_q, goal_q, gain=0.3):
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


class RuleBasedDemo:
    """
    Rule-based demo for bimanual manipulation.
    Structure is IDENTICAL to TeleoperationDemo, but with automatic velocity generation.
    """

    def __init__(self, joint_type='revolute', render_mode='human'):
        """
        Initialize demo.
        """
        self.joint_type = joint_type
        self.render_mode = render_mode

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
            kd_linear=1000.0,
            kp_angular=750.0,
            kd_angular=500.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Create automatic velocity controller (replaces keyboard controller)
        # Increase these values for faster movement:
        #   linear_speed: max linear velocity (pixels/s)
        #   angular_speed: max angular velocity (rad/s)
        #   joint_speed: max joint velocity
        joint_speed = 1.0  # Increased from 0.5
        if self.joint_enum == JointType.PRISMATIC:
            joint_speed *= 5.0
        self.auto_controller = AutoVelocityController(
            linear_speed=30.0,   # Increased from 10.0 (3x faster)
            angular_speed=1.5,   # Increased from 0.5 (3x faster)
            joint_speed=joint_speed
        )
        
        self.controlled_ee_idx = 0
        self.desired_poses = None
        
        # Goal will be retrieved from env.goal_manager
        self.goal_ee_poses = None

        # Display settings
        self.font = None
        self.font_small = None

        # Statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count = 0

        # Initialize environment
        self.reset_environment()

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

    def draw_info_overlay(self, screen, state):
        """Draw information overlay."""
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

        y_offset = draw_text(f"=== Rule-based Demo ({self.joint_type.upper()}) ===", y_offset, self.font, (0, 0, 128))
        y_offset += 5
        y_offset = draw_text(f"Step: {self.step_count} | Reward: {self.total_reward:.2f}", y_offset)
        
        # Show tracking error
        if hasattr(self, 'goal_ee_poses'):
            err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
            err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
            y_offset = draw_text(f"Goal Error: L={err_left:.1f}px R={err_right:.1f}px", y_offset, color=(200, 0, 0) if err_left > 50 else (0, 100, 0))
        
        y_offset += 5
        
        # EE info
        for i in range(2):
            pose = state['ee_poses'][i]
            y_offset = draw_text(f"EE {i}: x={pose[0]:.1f} y={pose[1]:.1f} θ={pose[2]:.2f}", y_offset)
        
        # Controls
        y_offset += 10
        y_offset = draw_text("R: Reset | ESC: Exit", y_offset, color=(100, 100, 100))

    def run(self):
        """Run the rule-based demo."""
        print("\n" + "="*60)
        print("Starting Rule-based Demo")
        print("="*60)
        print("Controls: R = Reset, ESC = Exit")
        print("="*60)

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
            
            # Compute velocity toward goal (automatic control)
            # Higher gain = faster acceleration & deceleration near goal
            controlled_velocity = self.auto_controller.compute_velocity_toward_goal(
                state['ee_poses'][self.controlled_ee_idx],
                self.goal_ee_poses[self.controlled_ee_idx],
                gain=0.5  # Increased from 0.2
            )
            joint_velocity_cmd = self.auto_controller.compute_joint_velocity_toward_goal(
                current_q,
                self.goal_joint_state,
                gain=0.5  # Increased from 0.2
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
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update statistics
            self.step_count += 1
            self.total_reward += reward

            # Render
            if self.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption("SWIVL Rule-based Demo")
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()

                screen_surface = self.env._draw()
                self.env.window.blit(screen_surface, screen_surface.get_rect())
                
                # Draw info overlay
                state = self.get_current_state(obs)
                self.draw_info_overlay(self.env.window, state)
                
                pygame.display.flip()

            # Print periodic status
            if self.step_count % 100 == 0:
                err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
                err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
                print(f"[Step {self.step_count:4d}] Goal Error: L={err_left:.1f}px R={err_right:.1f}px | q={current_q:.2f}")

            clock.tick(self.env.metadata['render_fps'])

            # Handle episode termination
            if terminated or truncated:
                print(f"\nEpisode finished!")
                self.reset_environment()

        # Cleanup
        self.env.close()
        pygame.quit()
        print("\nDemo completed!")


def main():
    parser = argparse.ArgumentParser(description='Rule-based demo for bimanual manipulation')
    parser.add_argument('joint_type', nargs='?', default='revolute',
                        choices=['revolute', 'prismatic', 'fixed'])
    args = parser.parse_args()

    print("\n" + "="*60)
    print("RULE-BASED BIMANUAL MANIPULATION DEMO")
    print("="*60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print("="*60)

    demo = RuleBasedDemo(joint_type=args.joint_type, render_mode='human')
    demo.run()


if __name__ == "__main__":
    main()
