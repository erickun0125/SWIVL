"""
Static Pose Holding Demo

This demo initializes the environment and holds the initial pose using PD control.
Useful for:
- Verifying gripper grasping works correctly
- Checking frame alignment
- Visualizing the setup without movement
- Debugging stability issues

Usage:
    python demo_static_hold.py [joint_type]

Arguments:
    joint_type: 'revolute', 'prismatic', or 'fixed' (default: revolute)

Controls:
    R: Reset environment
    ESC: Exit
"""

import argparse
import numpy as np
import pygame
import sys
import os

# Prevent pygame window creation error
os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'

from gym_biart.envs.biart import BiArtEnv
from gym_biart.envs.linkage_manager import create_two_link_object, JointType
from gym_biart.envs.pd_controller import MultiGripperController, PDGains


class StaticHoldDemo:
    """Demo for holding initial pose with visualization."""

    def __init__(self, joint_type='revolute'):
        """Initialize demo."""
        self.joint_type = joint_type

        # Create environment
        print(f"Creating BiArt environment with {joint_type} joint...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode='human',
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

        # Create PD controller with soft gains for stable holding
        gains = PDGains(
            kp_linear=15.0,   # Softer gains for stability
            kd_linear=5.0,
            kp_angular=8.0,
            kd_angular=2.0
        )
        self.controller = MultiGripperController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Display settings
        self.font = None
        self.font_small = None

        # Statistics
        self.step_count = 0
        self.episode_count = 0

        # Desired poses (will be set to initial poses)
        self.desired_poses = None

        # Initialize
        self.reset_environment()

    def reset_environment(self):
        """Reset environment and set desired poses to initial poses."""
        print(f"\nResetting environment (Episode {self.episode_count + 1})...")

        # Reset environment
        obs, info = self.env.reset()

        # Extract initial poses
        initial_ee_poses = np.array([
            obs[0:3],   # Left gripper
            obs[3:6],   # Right gripper
        ])

        # Set linkage bodies
        self.linkage.set_link_body(0, self.env.link1)
        self.linkage.set_link_body(1, self.env.link2)

        # Set desired poses to initial poses (hold position)
        self.desired_poses = initial_ee_poses.copy()

        # Reset controller
        self.controller.reset()

        # Reset statistics
        self.step_count = 0
        self.episode_count += 1

        print("Environment reset complete!")
        print(f"Holding initial poses:")
        print(f"  Left EE:  x={self.desired_poses[0,0]:.1f}, y={self.desired_poses[0,1]:.1f}, θ={np.degrees(self.desired_poses[0,2]):.1f}°")
        print(f"  Right EE: x={self.desired_poses[1,0]:.1f}, y={self.desired_poses[1,1]:.1f}, θ={np.degrees(self.desired_poses[1,2]):.1f}°")

    def get_current_state(self):
        """Get current state of end-effectors and object."""
        current_ee_poses = np.array([
            [self.env.left_gripper.position.x,
             self.env.left_gripper.position.y,
             self.env.left_gripper.angle],
            [self.env.right_gripper.position.x,
             self.env.right_gripper.position.y,
             self.env.right_gripper.angle],
        ])

        current_velocities = np.array([
            [self.env.left_gripper.velocity.x,
             self.env.left_gripper.velocity.y,
             self.env.left_gripper.angular_velocity],
            [self.env.right_gripper.velocity.x,
             self.env.right_gripper.velocity.y,
             self.env.right_gripper.angular_velocity],
        ])

        external_wrenches = np.array([
            self.env.external_wrench_left,
            self.env.external_wrench_right
        ])

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
        """Draw information overlay."""
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 14, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 12)

        state = self.get_current_state()

        y_offset = 10
        line_height = 18
        x_margin = 10

        def draw_text(text, y, font=None, color=(0, 0, 0), bg_color=(255, 255, 255, 200)):
            if font is None:
                font = self.font_small

            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (x_margin, y)

            bg_surface = pygame.Surface((text_rect.width + 10, text_rect.height + 4))
            bg_surface.set_alpha(200)
            bg_surface.fill(bg_color[:3])
            screen.blit(bg_surface, (x_margin - 5, y - 2))

            screen.blit(text_surface, text_rect)
            return y + line_height

        # Title
        y_offset = draw_text(
            f"=== Static Pose Holding Demo ({self.joint_type.upper()}) ===",
            y_offset,
            self.font,
            color=(0, 0, 128)
        )
        y_offset += 5

        # Episode info
        y_offset = draw_text(
            f"Episode: {self.episode_count} | Step: {self.step_count}",
            y_offset
        )
        y_offset += 5

        # Mode
        y_offset = draw_text(
            "Mode: HOLDING INITIAL POSE",
            y_offset,
            color=(0, 128, 0)
        )
        y_offset += 5

        # End-effector information
        for i in range(2):
            ee_name = f"EE {i} ({'LEFT' if i == 0 else 'RIGHT'})"

            # Current pose
            pose = state['ee_poses'][i]
            y_offset = draw_text(
                f"{ee_name} Pose: x={pose[0]:6.1f} y={pose[1]:6.1f} θ={pose[2]:5.2f}",
                y_offset,
                color=(0, 0, 200)
            )

            # Desired pose
            desired = self.desired_poses[i]
            y_offset = draw_text(
                f"  Desired:     x={desired[0]:6.1f} y={desired[1]:6.1f} θ={desired[2]:5.2f}",
                y_offset,
                color=(0, 100, 0)
            )

            # Error
            error = pose - desired
            error_norm = np.linalg.norm(error[:2])
            y_offset = draw_text(
                f"  Error: |pos|={error_norm:5.2f} Δθ={error[2]:5.2f}",
                y_offset,
                color=(200, 0, 0) if error_norm > 5.0 else (100, 100, 100)
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

        joint_config = state['joint_config']
        if len(joint_config) > 0:
            if self.joint_enum == JointType.REVOLUTE:
                y_offset = draw_text(
                    f"Joint Angle: {joint_config[0]:5.2f} rad ({np.degrees(joint_config[0]):6.1f}°)",
                    y_offset
                )
            elif self.joint_enum == JointType.PRISMATIC:
                y_offset = draw_text(f"Joint Position: {joint_config[0]:6.2f}", y_offset)
            else:
                y_offset = draw_text("Joint: FIXED (no DOF)", y_offset)

        for i, pose in enumerate(state['link_poses']):
            y_offset = draw_text(
                f"Link {i}: x={pose[0]:6.1f} y={pose[1]:6.1f} θ={pose[2]:5.2f}",
                y_offset
            )

        # Controls
        y_offset += 10
        y_offset = draw_text("--- Controls ---", y_offset, color=(0, 100, 0))
        y_offset = draw_text("R: Reset", y_offset, color=(0, 80, 0))
        y_offset = draw_text("ESC: Exit", y_offset, color=(0, 80, 0))

    def run(self):
        """Run the demo."""
        print("\n" + "="*60)
        print("Static Pose Holding Demo")
        print("="*60)
        print("\nThis demo holds the initial pose using PD control.")
        print("Observe the stability and grasping behavior.")
        print("\nControls:")
        print("  R: Reset environment")
        print("  ESC: Exit")
        print("\nDemo running...")

        clock = pygame.time.Clock()
        running = True

        while running:
            # Process events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_environment()

            # Get current state
            state = self.get_current_state()

            # Compute control wrenches to hold desired poses
            wrenches = self.controller.compute_wrenches(
                state['ee_poses'],
                self.desired_poses,
                state['ee_velocities']
            )

            # Apply action
            action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update statistics
            self.step_count += 1

            # Render
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

            # Draw info overlay
            self.draw_info_overlay(self.env.window)

            # Update display
            pygame.event.pump()
            pygame.display.update()

            # Print periodic status
            if self.step_count % 100 == 0:
                error_norm = np.linalg.norm(state['ee_poses'][0][:2] - self.desired_poses[0][:2])
                wrench_mag = np.linalg.norm(state['external_wrenches'][0][:2])
                print(f"[Step {self.step_count:4d}] Pos Error: {error_norm:5.2f} | "
                      f"Wrench: {wrench_mag:5.1f}N | "
                      f"Stable: {error_norm < 2.0}")

            # Control frame rate
            clock.tick(self.env.metadata['render_fps'])

        # Cleanup
        self.env.close()
        pygame.quit()

        print("\n" + "="*60)
        print("Demo completed successfully!")
        print(f"Total steps: {self.step_count}")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Static pose holding demo for bimanual manipulation'
    )
    parser.add_argument(
        'joint_type',
        nargs='?',
        default='revolute',
        choices=['revolute', 'prismatic', 'fixed'],
        help='Type of articulated joint (default: revolute)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("STATIC POSE HOLDING DEMO")
    print("="*60)
    print(f"Joint Type: {args.joint_type.upper()}")
    print("="*60)

    demo = StaticHoldDemo(joint_type=args.joint_type)
    demo.run()


if __name__ == "__main__":
    main()
