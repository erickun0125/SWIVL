"""
Unified Interactive Demo for Bimanual Manipulation System

This demo integrates all components:
- Linkage object management (R, P, Fixed joints)
- PD controller for pose tracking
- Keyboard teleoperation
- Real-time visualization with wrench display

Usage:
    python demo_teleoperation.py [joint_type]

Arguments:
    joint_type: 'revolute', 'prismatic', or 'fixed' (default: revolute)

Keyboard Controls:
    Arrow Keys: Move controlled end-effector (linear velocity)
    Q/W: Rotate counterclockwise/clockwise (angular velocity)
    1/2: Switch controlled end-effector (0 or 1)
    Space: Reset velocity to zero
    R: Reset environment
    H: Toggle help display
    ESC: Exit

Display Information:
    - Current controlled EE and its velocity command
    - Pose of each end-effector
    - External wrench on each end-effector (real-time)
    - Joint configuration
    - Reward and success status
"""

import argparse
import numpy as np
import pygame
import sys

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains


class KeyboardVelocityController:
    """Simple keyboard velocity controller for teleoperation."""
    
    def __init__(self, linear_speed=30.0, angular_speed=1.0):
        """
        Initialize keyboard velocity controller.
        
        Args:
            linear_speed: Linear velocity magnitude
            angular_speed: Angular velocity magnitude
        """
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.velocity = np.zeros(3)  # [vx, vy, omega]
    
    def process_keys(self, keys):
        """Process keyboard state and return velocity command."""
        self.velocity = np.zeros(3)
        
        # Linear velocity
        if keys[pygame.K_UP]:
            self.velocity[1] += self.linear_speed
        if keys[pygame.K_DOWN]:
            self.velocity[1] -= self.linear_speed
        if keys[pygame.K_LEFT]:
            self.velocity[0] -= self.linear_speed
        if keys[pygame.K_RIGHT]:
            self.velocity[0] += self.linear_speed
        
        # Angular velocity
        if keys[pygame.K_q]:
            self.velocity[2] += self.angular_speed
        if keys[pygame.K_w]:
            self.velocity[2] -= self.angular_speed
        
        return self.velocity
    
    def get_velocity_array(self):
        """Return current velocity as array."""
        return self.velocity.copy()


class TeleoperationDemo:
    """Unified demo for bimanual manipulation with teleoperation."""

    def __init__(self, joint_type='revolute', render_mode='human'):
        """
        Initialize demo.

        Args:
            joint_type: Type of articulated joint ('revolute', 'prismatic', 'fixed')
            render_mode: Rendering mode
        """
        self.joint_type = joint_type
        self.render_mode = render_mode

        # Create environment
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

        # Create PD controller with tuned gains
        gains = PDGains(
            kp_linear=0.0,
            kd_linear=1000.0,
            kp_angular=0.0,
            kd_angular=100.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Create keyboard controller
        self.keyboard = KeyboardVelocityController(linear_speed=30.0, angular_speed=1.0)
        
        # Control which EE
        self.controlled_ee_idx = 0
        
        # Desired poses (will be initialized in reset)
        self.desired_poses = None

        # Display settings
        self.show_help = True
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

        # Reset controller
        self.controller.reset()

        # Reset statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        print("Environment reset complete!")

    def get_current_state(self, obs):
        """Get current state of end-effectors and object from observation."""
        # Get EE poses and velocities from observation
        current_ee_poses = obs['ee_poses']
        current_velocities = obs['ee_twists']  # [vx, vy, omega] per gripper
        external_wrenches = obs['external_wrenches']  # [tau, fx, fy] per gripper
        link_poses = obs['link_poses']

        # Get joint configuration from object manager
        joint_state = self.env.object_manager.get_joint_state()

        return {
            'ee_poses': current_ee_poses,
            'ee_velocities': current_velocities,
            'external_wrenches': external_wrenches,
            'joint_state': joint_state,
            'link_poses': link_poses
        }

    def draw_info_overlay(self, screen, state):
        """
        Draw information overlay on the screen.

        Args:
            screen: Pygame surface to draw on
            state: Current state dictionary
        """
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 14, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 12)

        vel_cmd = self.keyboard.get_velocity_array()

        # Prepare text lines
        y_offset = 10
        line_height = 18
        x_margin = 10

        def draw_text(text, y, font=None, color=(0, 0, 0), bg_color=(255, 255, 255, 200)):
            """Helper to draw text with background."""
            if font is None:
                font = self.font_small

            # Create text surface
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (x_margin, y)

            # Draw semi-transparent background
            bg_surface = pygame.Surface((text_rect.width + 10, text_rect.height + 4))
            bg_surface.set_alpha(200)
            bg_surface.fill(bg_color[:3])
            screen.blit(bg_surface, (x_margin - 5, y - 2))

            # Draw text
            screen.blit(text_surface, text_rect)

            return y + line_height

        # Title
        y_offset = draw_text(
            f"=== Bimanual Manipulation Demo ({self.joint_type.upper()}) ===",
            y_offset,
            self.font,
            color=(0, 0, 128)
        )
        y_offset += 5

        # Episode info
        y_offset = draw_text(
            f"Episode: {self.episode_count} | Step: {self.step_count} | Total Reward: {self.total_reward:.2f}",
            y_offset
        )
        y_offset += 5

        # Controlled EE and velocity command
        y_offset = draw_text(
            f"Controlled EE: {self.controlled_ee_idx} (Press 1/2 to switch)",
            y_offset,
            color=(128, 0, 0)
        )
        y_offset = draw_text(
            f"Velocity Cmd: vx={vel_cmd[0]:6.1f} vy={vel_cmd[1]:6.1f} omega={vel_cmd[2]:5.2f}",
            y_offset,
            color=(0, 100, 0)
        )
        y_offset += 5

        # End-effector information
        for i in range(2):
            ee_name = f"EE {i} ({'LEFT' if i == 0 else 'RIGHT'})"
            marker = ">" if i == self.controlled_ee_idx else " "

            # Pose
            pose = state['ee_poses'][i]
            y_offset = draw_text(
                f"{marker} {ee_name} Pose: x={pose[0]:6.1f} y={pose[1]:6.1f} theta={pose[2]:5.2f}",
                y_offset,
                color=(0, 0, 200) if i == self.controlled_ee_idx else (0, 0, 0)
            )

            # External wrench [tau, fx, fy] in MR convention
            wrench = state['external_wrenches'][i]
            wrench_mag = np.linalg.norm(wrench[1:3])  # Force magnitude (fx, fy)
            y_offset = draw_text(
                f"  Wrench: tau={wrench[0]:6.1f} fx={wrench[1]:6.1f} fy={wrench[2]:5.1f} |F|={wrench_mag:5.1f}",
                y_offset,
                color=(200, 0, 0) if wrench_mag > 50 else (100, 100, 100)
            )
            y_offset += 3

        # Object information
        y_offset += 2
        y_offset = draw_text("--- Object State ---", y_offset, color=(128, 0, 128))

        # Joint configuration
        joint_state = state['joint_state']
        if self.joint_enum == JointType.REVOLUTE:
            y_offset = draw_text(
                f"Joint Angle: {joint_state:5.2f} rad ({np.degrees(joint_state):6.1f} deg)",
                y_offset
            )
        elif self.joint_enum == JointType.PRISMATIC:
            y_offset = draw_text(f"Joint Position: {joint_state:6.2f}", y_offset)
        else:  # FIXED
            y_offset = draw_text("Joint: FIXED (no DOF)", y_offset)

        # Link poses
        for i, pose in enumerate(state['link_poses']):
            y_offset = draw_text(
                f"Link {i}: x={pose[0]:6.1f} y={pose[1]:6.1f} theta={pose[2]:5.2f}",
                y_offset
            )

        # Help text
        if self.show_help:
            y_offset += 10
            y_offset = draw_text("--- Controls ---", y_offset, color=(0, 100, 0))
            help_lines = [
                "Arrow Keys: Move EE",
                "Q/W: Rotate CCW/CW",
                "1/2: Switch EE",
                "Space: Stop",
                "R: Reset",
                "H: Toggle Help",
                "ESC: Exit"
            ]
            for line in help_lines:
                y_offset = draw_text(line, y_offset, color=(0, 80, 0))

    def run(self):
        """Run the interactive demo."""
        print("\n" + "="*60)
        print("Starting Interactive Demo")
        print("="*60)
        print("\nKeyboard controls:")
        print("  Arrow keys: Move controlled end-effector")
        print("  Q/W: Rotate counterclockwise/clockwise")
        print("  1/2: Switch controlled end-effector")
        print("  Space: Reset velocity")
        print("  R: Reset environment")
        print("  H: Toggle help display")
        print("  ESC: Exit")
        print("\nDemo running...")

        # Initialize pygame for human rendering
        pygame.init()
        
        # Main loop
        running = True
        clock = pygame.time.Clock()

        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
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
                        print("Velocity reset to zero")

            # Get keyboard state
            keys = pygame.key.get_pressed()
            velocity_cmd = self.keyboard.process_keys(keys)

            # Get current observation
            obs = self.env.get_obs()
            state = self.get_current_state(obs)

            # Update desired pose for controlled EE based on velocity
            dt = 1.0 / self.env.control_hz
            self.desired_poses[self.controlled_ee_idx, 0] += velocity_cmd[0] * dt
            self.desired_poses[self.controlled_ee_idx, 1] += velocity_cmd[1] * dt
            self.desired_poses[self.controlled_ee_idx, 2] += velocity_cmd[2] * dt

            # Clip desired poses to workspace
            self.desired_poses[:, :2] = np.clip(self.desired_poses[:, :2], 20.0, 492.0)

            # Compute desired velocities for PD controller
            desired_velocities = np.zeros((2, 3))  # [vx, vy, omega] for each gripper
            desired_velocities[self.controlled_ee_idx] = velocity_cmd

            # Compute control wrenches using PD controller
            wrenches = self.controller.compute_wrenches(
                state['ee_poses'],
                self.desired_poses,
                desired_velocities,
                current_velocities=state['ee_velocities']
            )

            # Step environment with wrench action
            # Action format: [left_tau, left_fx, left_fy, right_tau, right_fx, right_fy]
            action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update statistics
            self.step_count += 1
            self.total_reward += reward

            # Render
            if self.render_mode == 'human':
                # Initialize pygame window if needed
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption("SWIVL Bimanual Manipulation Demo")
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()

                # Draw base scene
                screen_surface = self.env._draw()

                # Blit to window
                self.env.window.blit(screen_surface, screen_surface.get_rect())

                # Draw info overlay on top
                state = self.get_current_state(obs)
                self.draw_info_overlay(self.env.window, state)

                # Update display
                pygame.event.pump()
                pygame.display.update()

            # Print periodic status
            if self.step_count % 50 == 0:
                wrench_mag = np.linalg.norm(state['external_wrenches'][self.controlled_ee_idx][1:3])
                print(f"[Step {self.step_count:4d}] EE{self.controlled_ee_idx} | "
                      f"Vel:[{velocity_cmd[0]:5.1f},{velocity_cmd[1]:5.1f},{velocity_cmd[2]:4.2f}] | "
                      f"Wrench:|F|={wrench_mag:5.1f} | "
                      f"Reward:{reward:6.3f} | "
                      f"Success:{info.get('is_success', False)}")

            # Control frame rate
            clock.tick(self.env.metadata['render_fps'])

            # Handle episode termination
            if terminated or truncated:
                print(f"\nEpisode finished!")
                print(f"  Total steps: {self.step_count}")
                print(f"  Total reward: {self.total_reward:.2f}")
                print(f"  Success: {info.get('is_success', False)}")
                self.reset_environment()

        # Cleanup
        self.env.close()
        pygame.quit()

        print("\n" + "="*60)
        print("Demo Statistics")
        print("="*60)
        print(f"Total episodes: {self.episode_count}")
        print(f"Final total reward: {self.total_reward:.2f}")
        print("\nDemo completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interactive demo for bimanual manipulation with teleoperation'
    )
    parser.add_argument(
        'joint_type',
        nargs='?',
        default='revolute',
        choices=['revolute', 'prismatic', 'fixed'],
        help='Type of articulated joint (default: revolute)'
    )
    parser.add_argument(
        '--render-mode',
        default='human',
        choices=['human', 'rgb_array'],
        help='Rendering mode (default: human)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("BIMANUAL MANIPULATION TELEOPERATION DEMO")
    print("="*60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print(f"Render Mode: {args.render_mode}")
    print("="*60)

    # Create and run demo
    demo = TeleoperationDemo(
        joint_type=args.joint_type,
        render_mode=args.render_mode
    )
    demo.run()


if __name__ == "__main__":
    main()
