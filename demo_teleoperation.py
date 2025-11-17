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

from gym_biart.envs.biart import BiArtEnv
from gym_biart.envs.linkage_manager import LinkageObject, JointType, create_two_link_object
from gym_biart.envs.pd_controller import MultiGripperController, PDGains
from gym_biart.envs.keyboard_planner import MultiEEPlanner


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

        # Create linkage object
        self.linkage = create_two_link_object(self.joint_enum)

        # Create PD controller with tuned gains
        gains = PDGains(
            kp_linear=30.0,
            kd_linear=8.0,
            kp_angular=15.0,
            kd_angular=3.0
        )
        self.controller = MultiGripperController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Create keyboard planner
        self.planner = MultiEEPlanner(
            num_end_effectors=2,
            linkage_object=self.linkage,
            control_dt=1.0 / self.env.control_hz
        )

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

        # Extract initial poses from observation
        # obs format: [left_gripper(3), right_gripper(3), link1(3), link2(3), ext_wrench(6)]
        initial_ee_poses = np.array([
            obs[0:3],   # Left gripper
            obs[3:6],   # Right gripper
        ])

        # Set linkage bodies
        self.linkage.set_link_body(0, self.env.link1)
        self.linkage.set_link_body(1, self.env.link2)

        # Initialize planner
        self.planner.initialize_from_current_state(initial_ee_poses)

        # Reset controller
        self.controller.reset()

        # Reset statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        print("Environment reset complete!")

    def get_current_state(self):
        """Get current state of end-effectors and object."""
        # Get current EE poses
        current_ee_poses = np.array([
            [self.env.left_gripper.position.x,
             self.env.left_gripper.position.y,
             self.env.left_gripper.angle],
            [self.env.right_gripper.position.x,
             self.env.right_gripper.position.y,
             self.env.right_gripper.angle],
        ])

        # Get current velocities
        current_velocities = np.array([
            [self.env.left_gripper.velocity.x,
             self.env.left_gripper.velocity.y,
             self.env.left_gripper.angular_velocity],
            [self.env.right_gripper.velocity.x,
             self.env.right_gripper.velocity.y,
             self.env.right_gripper.angular_velocity],
        ])

        # Get external wrenches
        external_wrenches = np.array([
            self.env.external_wrench_left,
            self.env.external_wrench_right
        ])

        # Get object state
        self.linkage.update_joint_states()
        joint_config = self.linkage.get_configuration()
        link_poses = self.linkage.get_all_link_poses()

        return {
            'ee_poses': current_ee_poses,
            'ee_velocities': current_velocities,
            'external_wrenches': external_wrenches,
            'joint_config': joint_config,
            'link_poses': link_poses
        }

    def draw_info_overlay(self, screen):
        """
        Draw information overlay on the screen.

        Args:
            screen: Pygame surface to draw on
        """
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 14, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 12)

        # Get current state
        state = self.get_current_state()
        controlled_idx = self.planner.get_controlled_ee_index()
        vel_cmd = self.planner.keyboard.get_velocity_array()

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
            f"Controlled EE: {controlled_idx} (Press 1/2 to switch)",
            y_offset,
            color=(128, 0, 0)
        )
        y_offset = draw_text(
            f"Velocity Cmd: vx={vel_cmd[0]:6.1f} vy={vel_cmd[1]:6.1f} ω={vel_cmd[2]:5.2f}",
            y_offset,
            color=(0, 100, 0)
        )
        y_offset += 5

        # End-effector information
        for i in range(2):
            ee_name = f"EE {i} ({'LEFT' if i == 0 else 'RIGHT'})"
            marker = "►" if i == controlled_idx else " "

            # Pose
            pose = state['ee_poses'][i]
            y_offset = draw_text(
                f"{marker} {ee_name} Pose: x={pose[0]:6.1f} y={pose[1]:6.1f} θ={pose[2]:5.2f}",
                y_offset,
                color=(0, 0, 200) if i == controlled_idx else (0, 0, 0)
            )

            # External wrench
            wrench = state['external_wrenches'][i]
            wrench_mag = np.linalg.norm(wrench[:2])
            y_offset = draw_text(
                f"  Wrench: fx={wrench[0]:6.1f} fy={wrench[1]:6.1f} τ={wrench[2]:5.1f} |F|={wrench_mag:5.1f}",
                y_offset,
                color=(200, 0, 0) if wrench_mag > 50 else (100, 100, 100)
            )
            y_offset += 3

        # Object information
        y_offset += 2
        y_offset = draw_text("--- Object State ---", y_offset, color=(128, 0, 128))

        # Joint configuration
        joint_config = state['joint_config']
        if len(joint_config) > 0:
            if self.joint_enum == JointType.REVOLUTE:
                y_offset = draw_text(
                    f"Joint Angle: {joint_config[0]:5.2f} rad ({np.degrees(joint_config[0]):6.1f}°)",
                    y_offset
                )
            elif self.joint_enum == JointType.PRISMATIC:
                y_offset = draw_text(f"Joint Position: {joint_config[0]:6.2f}", y_offset)
            else:  # FIXED
                y_offset = draw_text("Joint: FIXED (no DOF)", y_offset)

        # Link poses
        for i, pose in enumerate(state['link_poses']):
            y_offset = draw_text(
                f"Link {i}: x={pose[0]:6.1f} y={pose[1]:6.1f} θ={pose[2]:5.2f}",
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

        # Main loop
        running = True
        clock = pygame.time.Clock()

        while running:
            # Get pygame events
            events = pygame.event.get()

            # Check for special commands
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_environment()
                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help

            # Get current state
            state = self.get_current_state()

            # Update planner with keyboard input
            desired_poses, actions = self.planner.update(events, state['ee_poses'])

            # Check for quit
            if actions['quit']:
                print("\nExiting demo...")
                running = False
                break

            # Compute control wrenches using PD controller
            wrenches = self.controller.compute_wrenches(
                state['ee_poses'],
                desired_poses,
                state['ee_velocities']
            )

            # Construct action for environment
            # Action format: [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
            action = np.concatenate([wrenches[0], wrenches[1]])

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update statistics
            self.step_count += 1
            self.total_reward += reward

            # Render base environment
            if self.render_mode == 'human':
                # Initialize pygame window if needed
                if self.env.window is None:
                    pygame.init()
                    pygame.display.init()
                    self.env.window = pygame.display.set_mode((512, 512))
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()

                # Draw base scene
                screen_surface = self.env._draw()

                # Blit to window
                self.env.window.blit(screen_surface, screen_surface.get_rect())

                # Draw info overlay on top
                self.draw_info_overlay(self.env.window)

                # Update display
                pygame.event.pump()
                pygame.display.update()

            # Print periodic status
            if self.step_count % 50 == 0:
                controlled_idx = self.planner.get_controlled_ee_index()
                vel_cmd = self.planner.keyboard.get_velocity_array()
                wrench_mag = np.linalg.norm(state['external_wrenches'][controlled_idx][:2])
                print(f"[Step {self.step_count:4d}] EE{controlled_idx} | "
                      f"Vel:[{vel_cmd[0]:5.1f},{vel_cmd[1]:5.1f},{vel_cmd[2]:4.2f}] | "
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
