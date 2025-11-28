"""
Unified Interactive Demo for Bimanual Manipulation System

This demo integrates all components:
- Linkage object management (R, P, Fixed joints)
- PD controller for pose tracking
- Keyboard teleoperation with kinematic constraints
- Real-time visualization with wrench display

Usage:
    python demo_teleoperation.py [joint_type]

Arguments:
    joint_type: 'revolute', 'prismatic', or 'fixed' (default: revolute)

Keyboard Controls:
    Arrow Keys: Move controlled end-effector (linear velocity)
    Q/W: Rotate counterclockwise/clockwise (angular velocity)
    A/S: Object joint velocity (negative/positive)
    1/2: Switch controlled end-effector (0 or 1)
    Space: Reset velocity to zero
    R: Reset environment
    H: Toggle help display
    ESC: Exit

Kinematic Constraint:
    When you control one EE and specify joint velocity,
    the other EE automatically follows the kinematic constraint
    of the articulated object. This allows coordinated bimanual
    manipulation while respecting object kinematics.

Display Information:
    - Current controlled EE and its velocity command
    - Joint velocity command
    - Pose of each end-effector (current and desired)
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
    """Keyboard velocity controller with EE and joint velocity control."""
    
    def __init__(self, linear_speed=30.0, angular_speed=1.0, joint_speed=1.0):
        """
        Initialize keyboard velocity controller.
        
        Args:
            linear_speed: Linear velocity magnitude (pixels/s)
            angular_speed: Angular velocity magnitude (rad/s)
            joint_speed: Joint velocity magnitude (rad/s for revolute, pixels/s for prismatic)
        """
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.joint_speed = joint_speed
        self.velocity = np.zeros(3)  # [vx, vy, omega] for controlled EE
        self.joint_velocity = 0.0    # Joint velocity
    
    def process_keys(self, keys):
        """Process keyboard state and return velocity command."""
        self.velocity = np.zeros(3)
        self.joint_velocity = 0.0
        
        # Linear velocity (Pygame Y-axis points downward, so invert Y)
        if keys[pygame.K_UP]:
            self.velocity[1] -= self.linear_speed   # Up arrow → move up on screen
        if keys[pygame.K_DOWN]:
            self.velocity[1] += self.linear_speed   # Down arrow → move down on screen
        if keys[pygame.K_LEFT]:
            self.velocity[0] -= self.linear_speed   # Left arrow → move left
        if keys[pygame.K_RIGHT]:
            self.velocity[0] += self.linear_speed   # Right arrow → move right
        
        # Angular velocity (Q = CCW, W = CW in screen coordinates)
        if keys[pygame.K_q]:
            self.velocity[2] -= self.angular_speed  # Q → CCW
        if keys[pygame.K_w]:
            self.velocity[2] += self.angular_speed  # W → CW
        
        # Joint velocity (A = negative, S = positive)
        if keys[pygame.K_a]:
            self.joint_velocity -= self.joint_speed  # A → negative joint velocity
        if keys[pygame.K_s]:
            self.joint_velocity += self.joint_speed  # S → positive joint velocity
        
        return self.velocity
    
    def get_velocity_array(self):
        """Return current EE velocity as array."""
        return self.velocity.copy()
    
    def get_joint_velocity(self):
        """Return current joint velocity."""
        return self.joint_velocity


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
            kp_linear=1000.0,
            kd_linear=500.0,
            kp_angular=100.0,
            kd_angular=10.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Create keyboard controller
        self.keyboard = KeyboardVelocityController(linear_speed=90.0, angular_speed=3.0, joint_speed=3.0)
        
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

    def compute_constrained_velocity(
        self,
        controlled_ee_idx: int,
        controlled_velocity: np.ndarray,
        joint_velocity: float
    ) -> np.ndarray:
        """
        Compute the other EE's velocity based on kinematic constraints.
        
        When one EE is controlled and joint velocity is specified,
        the other EE must follow the kinematic constraint of the articulated object.
        
        Args:
            controlled_ee_idx: 0 (left) or 1 (right)
            controlled_velocity: [vx, vy, omega] of controlled EE in world frame
            joint_velocity: Joint velocity (rad/s for revolute, pixels/s for prismatic)
        
        Returns:
            other_velocity: [vx, vy, omega] of other EE in world frame
        """
        joint_type = self.env.object_manager.joint_type
        link_poses = self.env.object_manager.get_link_poses()
        grasping_poses = self.env.object_manager.get_grasping_poses()
        cfg = self.env.object_config
        
        link1_pose = link_poses[0]
        link2_pose = link_poses[1]
        left_ee_pose = grasping_poses["left"]
        right_ee_pose = grasping_poses["right"]
        
        # Joint position (link1's right end = link2's left end)
        joint_pos = link1_pose[:2] + (cfg.link_length / 2) * np.array([
            np.cos(link1_pose[2]),
            np.sin(link1_pose[2])
        ])
        
        if joint_type == JointType.FIXED:
            # Fixed joint: both EEs move together with same velocity
            return controlled_velocity.copy()
        
        elif joint_type == JointType.REVOLUTE:
            # Revolute joint: rotation around joint center
            if controlled_ee_idx == 0:  # Left EE (on link1) is controlled
                # Left EE velocity determines link1 motion
                # Right EE (on link2) = link1 motion + joint rotation effect
                
                # Link1 angular velocity = left EE angular velocity
                omega_link1 = controlled_velocity[2]
                
                # Left EE offset from link1 center
                left_offset = left_ee_pose[:2] - link1_pose[:2]
                
                # Link1 center velocity from left EE velocity
                # V_link1 = V_left_ee - omega × offset
                v_link1 = controlled_velocity[:2] - omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                # Joint velocity from link1 center
                joint_offset = joint_pos - link1_pose[:2]
                v_joint = v_link1 + omega_link1 * np.array([-joint_offset[1], joint_offset[0]])
                
                # Link2 angular velocity = link1 angular + joint velocity
                omega_link2 = omega_link1 + joint_velocity
                
                # Link2 center from joint
                link2_offset = link2_pose[:2] - joint_pos
                v_link2 = v_joint + omega_link2 * np.array([-link2_offset[1], link2_offset[0]])
                
                # Right EE from link2 center
                right_offset = right_ee_pose[:2] - link2_pose[:2]
                v_right = v_link2 + omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                return np.array([v_right[0], v_right[1], omega_link2])
                
            else:  # Right EE (on link2) is controlled
                # Right EE velocity determines link2 motion (including joint effect)
                # Left EE (on link1) = link2 motion - joint rotation effect
                
                omega_link2 = controlled_velocity[2]
                
                # Right EE offset from link2 center
                right_offset = right_ee_pose[:2] - link2_pose[:2]
                
                # Link2 center velocity from right EE velocity
                v_link2 = controlled_velocity[:2] - omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                # Link2 center from joint
                link2_offset = link2_pose[:2] - joint_pos
                
                # Joint velocity (world frame)
                v_joint = v_link2 - omega_link2 * np.array([-link2_offset[1], link2_offset[0]])
                
                # Link1 angular velocity = link2 angular - joint velocity
                omega_link1 = omega_link2 - joint_velocity
                
                # Joint from link1 center
                joint_offset = joint_pos - link1_pose[:2]
                v_link1 = v_joint - omega_link1 * np.array([-joint_offset[1], joint_offset[0]])
                
                # Left EE from link1 center
                left_offset = left_ee_pose[:2] - link1_pose[:2]
                v_left = v_link1 + omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                return np.array([v_left[0], v_left[1], omega_link1])
        
        elif joint_type == JointType.PRISMATIC:
            # Prismatic joint: sliding along link1's x-axis (no relative rotation)
            # Both links have same orientation
            slide_dir = np.array([np.cos(link1_pose[2]), np.sin(link1_pose[2])])
            
            if controlled_ee_idx == 0:  # Left EE controlled
                # Link1 motion = left EE motion
                omega_link1 = controlled_velocity[2]
                
                left_offset = left_ee_pose[:2] - link1_pose[:2]
                v_link1 = controlled_velocity[:2] - omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                # Link2 has same rotation (prismatic constraint)
                omega_link2 = omega_link1
                
                # Link2 velocity = Link1 velocity + joint sliding effect
                v_link2 = v_link1 + joint_velocity * slide_dir
                
                # Right EE from link2
                right_offset = right_ee_pose[:2] - link2_pose[:2]
                v_right = v_link2 + omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                return np.array([v_right[0], v_right[1], omega_link2])
                
            else:  # Right EE controlled
                omega_link2 = controlled_velocity[2]
                
                right_offset = right_ee_pose[:2] - link2_pose[:2]
                v_link2 = controlled_velocity[:2] - omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                # Link1 has same rotation
                omega_link1 = omega_link2
                
                # Link1 velocity = Link2 velocity - joint sliding effect
                v_link1 = v_link2 - joint_velocity * slide_dir
                
                # Left EE from link1
                left_offset = left_ee_pose[:2] - link1_pose[:2]
                v_left = v_link1 + omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                return np.array([v_left[0], v_left[1], omega_link1])
        
        # Fallback
        return controlled_velocity.copy()

    def draw_desired_frames(self, screen):
        """
        Draw desired pose frames for both grippers.
        
        Arrow points in jaw direction (+y in gripper body frame = theta + π/2).
        
        Args:
            screen: Pygame surface to draw on
        """
        if self.desired_poses is None:
            return
        
        # Colors for each gripper
        colors = [
            (65, 105, 225),   # RoyalBlue for left
            (220, 20, 60)     # Crimson for right
        ]
        
        for i, pose in enumerate(self.desired_poses):
            x, y, theta = pose
            
            # Skip if outside visible area
            if not (0 <= x <= 512 and 0 <= y <= 512):
                continue
            
            # Frame size
            arrow_length = 25
            arrow_width = 8
            
            # Arrow points in jaw direction (+y in body frame = theta + π/2)
            jaw_angle = theta + np.pi / 2
            cos_j, sin_j = np.cos(jaw_angle), np.sin(jaw_angle)
            
            # Arrow tip (jaw direction)
            tip_x = x + arrow_length * cos_j
            tip_y = y + arrow_length * sin_j
            
            # Arrow base corners (perpendicular to jaw direction)
            base_left_x = x - arrow_width * sin_j
            base_left_y = y + arrow_width * cos_j
            base_right_x = x + arrow_width * sin_j
            base_right_y = y - arrow_width * cos_j
            
            # Draw filled arrow (desired pose indicator)
            arrow_points = [
                (int(tip_x), int(tip_y)),
                (int(base_left_x), int(base_left_y)),
                (int(base_right_x), int(base_right_y))
            ]
            
            # Create semi-transparent surface for the arrow
            arrow_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
            color_with_alpha = (*colors[i], 120)  # Add alpha
            pygame.draw.polygon(arrow_surface, color_with_alpha, arrow_points)
            screen.blit(arrow_surface, (0, 0))
            
            # Draw arrow outline
            pygame.draw.polygon(screen, colors[i], arrow_points, 2)
            
            # Draw small circle at origin
            pygame.draw.circle(screen, colors[i], (int(x), int(y)), 5, 2)
            
            # Draw "D" label for "Desired"
            if self.font_small is not None:
                label = self.font_small.render(f"D{i}", True, colors[i])
                screen.blit(label, (int(x) + 8, int(y) - 15))

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
        joint_vel = self.keyboard.get_joint_velocity()
        y_offset = draw_text(
            f"Controlled EE: {self.controlled_ee_idx} (Press 1/2 to switch)",
            y_offset,
            color=(128, 0, 0)
        )
        y_offset = draw_text(
            f"EE Velocity: vx={vel_cmd[0]:6.1f} vy={vel_cmd[1]:6.1f} omega={vel_cmd[2]:5.2f}",
            y_offset,
            color=(0, 100, 0)
        )
        y_offset = draw_text(
            f"Joint Velocity: {joint_vel:6.2f} (A/S keys)",
            y_offset,
            color=(150, 50, 0) if joint_vel != 0 else (100, 100, 100)
        )
        y_offset += 5

        # End-effector information
        for i in range(2):
            ee_name = f"EE {i} ({'LEFT' if i == 0 else 'RIGHT'})"
            marker = ">" if i == self.controlled_ee_idx else " "

            # Current Pose
            pose = state['ee_poses'][i]
            y_offset = draw_text(
                f"{marker} {ee_name} Pose: x={pose[0]:6.1f} y={pose[1]:6.1f} theta={pose[2]:5.2f}",
                y_offset,
                color=(0, 0, 200) if i == self.controlled_ee_idx else (0, 0, 0)
            )
            
            # Desired Pose
            if self.desired_poses is not None:
                des_pose = self.desired_poses[i]
                y_offset = draw_text(
                    f"  Desired: x={des_pose[0]:6.1f} y={des_pose[1]:6.1f} theta={des_pose[2]:5.2f}",
                    y_offset,
                    color=(100, 0, 150) if i == self.controlled_ee_idx else (80, 80, 80)
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
                "A/S: Joint velocity -/+",
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
        print("  A/S: Joint velocity (negative/positive)")
        print("  1/2: Switch controlled end-effector")
        print("  Space: Reset velocity")
        print("  R: Reset environment")
        print("  H: Toggle help display")
        print("  ESC: Exit")
        print("\nNote: Other EE follows kinematic constraints automatically!")
        print("\nDemo running...")

        # Initialize pygame for human rendering
        pygame.init()
        
        # Main loop
        running = True
        clock = pygame.time.Clock()

        while running:
            # Pump events first to update keyboard state
            pygame.event.pump()
            
            # Process events for discrete actions
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
                        self.keyboard.joint_velocity = 0.0
                        print("Velocity reset to zero")

            # Get current keyboard state (continuous control)
            keys = pygame.key.get_pressed()
            velocity_cmd = self.keyboard.process_keys(keys)
            joint_velocity_cmd = self.keyboard.get_joint_velocity()

            # Get current observation
            obs = self.env.get_obs()
            state = self.get_current_state(obs)

            # Compute velocities for both EEs using kinematic constraints
            dt = 1.0 / self.env.control_hz
            other_ee_idx = 1 - self.controlled_ee_idx
            
            # Controlled EE velocity from keyboard
            controlled_velocity = velocity_cmd
            
            # Other EE velocity from kinematic constraint
            other_velocity = self.compute_constrained_velocity(
                self.controlled_ee_idx,
                controlled_velocity,
                joint_velocity_cmd
            )
            
            # Update desired poses for both EEs
            self.desired_poses[self.controlled_ee_idx, 0] += controlled_velocity[0] * dt
            self.desired_poses[self.controlled_ee_idx, 1] += controlled_velocity[1] * dt
            self.desired_poses[self.controlled_ee_idx, 2] += controlled_velocity[2] * dt
            
            self.desired_poses[other_ee_idx, 0] += other_velocity[0] * dt
            self.desired_poses[other_ee_idx, 1] += other_velocity[1] * dt
            self.desired_poses[other_ee_idx, 2] += other_velocity[2] * dt

            # Clip desired poses to workspace
            self.desired_poses[:, :2] = np.clip(self.desired_poses[:, :2], 20.0, 492.0)

            # Compute desired velocities for PD controller
            desired_velocities = np.zeros((2, 3))  # [vx, vy, omega] for each gripper
            desired_velocities[self.controlled_ee_idx] = controlled_velocity
            desired_velocities[other_ee_idx] = other_velocity

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

                # Draw desired pose frames on the surface
                self.draw_desired_frames(screen_surface)

                # Blit to window
                self.env.window.blit(screen_surface, screen_surface.get_rect())

                # Draw info overlay on top
                state = self.get_current_state(obs)
                self.draw_info_overlay(self.env.window, state)

                # Update display
                pygame.display.flip()

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
