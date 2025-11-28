"""
Playground Demo for Independent Bimanual Control

A free-form testing environment where both end-effectors can be controlled
independently without any object constraints. Useful for testing and debugging
controllers in the most flexible setup.

Usage:
    python scripts/demos/demo_playground.py

Keyboard Controls:
    === Left End-Effector (EE0) ===
    WASD: Move left EE (linear velocity)
    Q/E: Rotate left EE (angular velocity)
    
    === Right End-Effector (EE1) ===
    Arrow Keys: Move right EE (linear velocity)
    Z/C: Rotate right EE (angular velocity)
    
    === General ===
    Space: Reset all velocities to zero
    R: Reset environment
    H: Toggle help display
    ESC: Exit

Display Information:
    - Velocity commands for both EEs
    - Pose of each end-effector
    - External wrench on each end-effector
"""

import numpy as np
import pygame
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs.biart import BiArtEnv
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains


class IndependentKeyboardController:
    """Independent keyboard controller for two end-effectors with velocity accumulation."""
    
    def __init__(self, linear_accel=2.0, angular_accel=0.05, damping=0.95):
        """
        Initialize independent keyboard controller with acceleration-based control.
        
        Args:
            linear_accel: Linear acceleration magnitude per frame
            angular_accel: Angular acceleration magnitude per frame
            damping: Velocity damping factor (0-1), higher = less damping
        """
        self.linear_accel = linear_accel
        self.angular_accel = angular_accel
        self.damping = damping
        
        # Current velocities (persistent across frames)
        self.velocity_left = np.zeros(3)  # [vx, vy, omega] for EE0
        self.velocity_right = np.zeros(3)  # [vx, vy, omega] for EE1
        
        # Maximum velocities
        self.max_linear_vel = 50.0
        self.max_angular_vel = 2.0
    
    def process_keys(self, keys):
        """
        Process keyboard state and update velocities with accumulation.
        
        Velocities accumulate when keys are pressed and decay when not pressed.
        """
        # Track which axes have input
        left_input = np.zeros(3, dtype=bool)
        right_input = np.zeros(3, dtype=bool)
        
        # Left EE (EE0) - WASD + Q/E
        if keys[pygame.K_w]:
            self.velocity_left[1] += self.linear_accel
            left_input[1] = True
        if keys[pygame.K_s]:
            self.velocity_left[1] -= self.linear_accel
            left_input[1] = True
        if keys[pygame.K_a]:
            self.velocity_left[0] -= self.linear_accel
            left_input[0] = True
        if keys[pygame.K_d]:
            self.velocity_left[0] += self.linear_accel
            left_input[0] = True
        if keys[pygame.K_q]:
            self.velocity_left[2] += self.angular_accel
            left_input[2] = True
        if keys[pygame.K_e]:
            self.velocity_left[2] -= self.angular_accel
            left_input[2] = True
        
        # Right EE (EE1) - Arrow Keys + Z/C
        if keys[pygame.K_UP]:
            self.velocity_right[1] += self.linear_accel
            right_input[1] = True
        if keys[pygame.K_DOWN]:
            self.velocity_right[1] -= self.linear_accel
            right_input[1] = True
        if keys[pygame.K_LEFT]:
            self.velocity_right[0] -= self.linear_accel
            right_input[0] = True
        if keys[pygame.K_RIGHT]:
            self.velocity_right[0] += self.linear_accel
            right_input[0] = True
        if keys[pygame.K_z]:
            self.velocity_right[2] += self.angular_accel
            right_input[2] = True
        if keys[pygame.K_c]:
            self.velocity_right[2] -= self.angular_accel
            right_input[2] = True
        
        # Apply damping to axes without input
        for i in range(3):
            if not left_input[i]:
                self.velocity_left[i] *= self.damping
            if not right_input[i]:
                self.velocity_right[i] *= self.damping
        
        # Clamp velocities to maximum
        # Linear velocities (x, y)
        linear_mag_left = np.linalg.norm(self.velocity_left[:2])
        if linear_mag_left > self.max_linear_vel:
            self.velocity_left[:2] *= self.max_linear_vel / linear_mag_left
        
        linear_mag_right = np.linalg.norm(self.velocity_right[:2])
        if linear_mag_right > self.max_linear_vel:
            self.velocity_right[:2] *= self.max_linear_vel / linear_mag_right
        
        # Angular velocities (omega)
        self.velocity_left[2] = np.clip(self.velocity_left[2], 
                                         -self.max_angular_vel, 
                                         self.max_angular_vel)
        self.velocity_right[2] = np.clip(self.velocity_right[2], 
                                          -self.max_angular_vel, 
                                          self.max_angular_vel)
        
        return self.velocity_left.copy(), self.velocity_right.copy()
    
    def get_velocity_array(self):
        """Return current velocities as (2, 3) array."""
        return np.array([self.velocity_left, self.velocity_right])
    
    def reset(self):
        """Reset all velocities to zero."""
        self.velocity_left = np.zeros(3)
        self.velocity_right = np.zeros(3)


class PlaygroundDemo:
    """Playground demo for independent bimanual control."""

    def __init__(self, render_mode='human'):
        """
        Initialize demo.

        Args:
            render_mode: Rendering mode
        """
        self.render_mode = render_mode

        # Create environment (with fixed joint for simplest case)
        print(f"Creating playground environment...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode=render_mode,
            joint_type='none',  # Simplest joint type
        )

        # Create PD controller with tuned gains
        gains = PDGains(
            kp_linear=0.0,
            kd_linear=1000.0,
            kp_angular=0.0,
            kd_angular=100.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Create independent keyboard controller with acceleration-based control
        self.keyboard = IndependentKeyboardController(
            linear_accel=2.0,      # Linear acceleration per frame
            angular_accel=0.05,    # Angular acceleration per frame  
            damping=0.95           # Velocity damping (0.95 = 5% decay per frame)
        )
        
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
        """Get current state of end-effectors from observation."""
        current_ee_poses = obs['ee_poses']
        current_velocities = obs['ee_twists']
        external_wrenches = obs['external_wrenches']
        link_poses = obs['link_poses']

        return {
            'ee_poses': current_ee_poses,
            'ee_velocities': current_velocities,
            'external_wrenches': external_wrenches,
            'link_poses': link_poses
        }

    def draw_info_overlay(self, screen, state, vel_left, vel_right):
        """
        Draw information overlay on the screen.

        Args:
            screen: Pygame surface to draw on
            state: Current state dictionary
            vel_left: Velocity command for left EE
            vel_right: Velocity command for right EE
        """
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 14, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 12)

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
            f"=== Playground Demo (Independent Control) ===",
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

        # Velocity commands
        y_offset = draw_text("--- Velocity Commands ---", y_offset, color=(0, 100, 0))
        y_offset = draw_text(
            f"Left  (WASD+Q/E): vx={vel_left[0]:6.1f} vy={vel_left[1]:6.1f} ω={vel_left[2]:5.2f}",
            y_offset,
            color=(128, 0, 0)
        )
        y_offset = draw_text(
            f"Right (Arrows+Z/C): vx={vel_right[0]:6.1f} vy={vel_right[1]:6.1f} ω={vel_right[2]:5.2f}",
            y_offset,
            color=(0, 0, 128)
        )
        y_offset += 5

        # End-effector information
        for i in range(2):
            ee_name = f"EE {i} ({'LEFT' if i == 0 else 'RIGHT'})"

            # Pose
            pose = state['ee_poses'][i]
            y_offset = draw_text(
                f"{ee_name} Pose: x={pose[0]:6.1f} y={pose[1]:6.1f} θ={pose[2]:5.2f}",
                y_offset,
                color=(128, 0, 0) if i == 0 else (0, 0, 128)
            )

            # External wrench [tau, fx, fy] in MR convention
            wrench = state['external_wrenches'][i]
            wrench_mag = np.linalg.norm(wrench[1:3])
            y_offset = draw_text(
                f"  Wrench: τ={wrench[0]:6.1f} fx={wrench[1]:6.1f} fy={wrench[2]:5.1f} |F|={wrench_mag:5.1f}",
                y_offset,
                color=(200, 0, 0) if wrench_mag > 50 else (100, 100, 100)
            )
            y_offset += 3

        # Help text
        if self.show_help:
            y_offset += 10
            y_offset = draw_text("--- Controls ---", y_offset, color=(0, 100, 0))
            help_lines = [
                "Left EE: WASD (Move), Q/E (Rotate)",
                "Right EE: Arrows (Move), Z/C (Rotate)",
                "Space: Stop All",
                "R: Reset",
                "H: Toggle Help",
                "ESC: Exit"
            ]
            for line in help_lines:
                y_offset = draw_text(line, y_offset, color=(0, 80, 0))

    def run(self):
        """Run the playground demo."""
        print("\n" + "="*60)
        print("Starting Playground Demo")
        print("="*60)
        print("\nIndependent bimanual control:")
        print("  Left EE:  WASD (move) + Q/E (rotate)")
        print("  Right EE: Arrows (move) + Z/C (rotate)")
        print("  Space: Stop all")
        print("  R: Reset")
        print("  H: Toggle help")
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
                    elif event.key == pygame.K_SPACE:
                        self.keyboard.reset()
                        print("All velocities reset to zero")

            # Get keyboard state
            keys = pygame.key.get_pressed()
            vel_left, vel_right = self.keyboard.process_keys(keys)

            # Get current observation
            obs = self.env.get_obs()
            state = self.get_current_state(obs)

            # Update desired poses for both EEs based on their independent velocities
            dt = 1.0 / self.env.control_hz
            
            # Left EE (EE0)
            self.desired_poses[0, 0] += vel_left[0] * dt
            self.desired_poses[0, 1] += vel_left[1] * dt
            self.desired_poses[0, 2] += vel_left[2] * dt
            
            # Right EE (EE1)
            self.desired_poses[1, 0] += vel_right[0] * dt
            self.desired_poses[1, 1] += vel_right[1] * dt
            self.desired_poses[1, 2] += vel_right[2] * dt

            # Clip desired poses to workspace
            self.desired_poses[:, :2] = np.clip(self.desired_poses[:, :2], 20.0, 492.0)

            # Compute desired velocities for PD controller
            desired_velocities = np.array([vel_left, vel_right])

            # Compute control wrenches using PD controller
            wrenches = self.controller.compute_wrenches(
                state['ee_poses'],
                self.desired_poses,
                desired_velocities,
                current_velocities=state['ee_velocities']
            )

            # Step environment with wrench action
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
                    pygame.display.set_caption("SWIVL Playground Demo")
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()

                # Draw base scene
                screen_surface = self.env._draw()

                # Blit to window
                self.env.window.blit(screen_surface, screen_surface.get_rect())

                # Draw info overlay on top
                state = self.get_current_state(obs)
                self.draw_info_overlay(self.env.window, state, vel_left, vel_right)

                # Update display
                pygame.event.pump()
                pygame.display.update()

            # Print periodic status
            if self.step_count % 100 == 0:
                wrench_left_mag = np.linalg.norm(state['external_wrenches'][0][1:3])
                wrench_right_mag = np.linalg.norm(state['external_wrenches'][1][1:3])
                print(f"[Step {self.step_count:4d}] "
                      f"L:|F|={wrench_left_mag:5.1f} R:|F|={wrench_right_mag:5.1f} | "
                      f"Reward:{reward:6.3f}")

            # Control frame rate
            clock.tick(self.env.metadata['render_fps'])

            # Handle episode termination
            if terminated or truncated:
                print(f"\nEpisode finished!")
                print(f"  Total steps: {self.step_count}")
                print(f"  Total reward: {self.total_reward:.2f}")
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
    print("\n" + "="*60)
    print("PLAYGROUND DEMO - INDEPENDENT BIMANUAL CONTROL")
    print("="*60)
    print("Render Mode: human")
    print("="*60)

    # Create and run demo
    demo = PlaygroundDemo(render_mode='human')
    demo.run()


if __name__ == "__main__":
    main()
