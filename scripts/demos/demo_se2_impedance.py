"""
SE(2) Impedance Control Demo for Bimanual Manipulation
Demonstrates the use of SE2ImpedanceController for compliant manipulation.
"""

import argparse
import numpy as np
import pygame

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.se2_impedance_controller import MultiGripperSE2ImpedanceController
from src.se2_dynamics import SE2RobotParams
from src.se2_math import normalize_angle
from src.bimanual_utils import compute_constrained_velocity


class AutoVelocityController:
    """
    Automatic velocity controller that generates velocity commands toward a goal.
    Same as in demo_rule_based.py.
    """
    
    def __init__(self, linear_speed=10.0, angular_speed=0.5, joint_speed=0.5):
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.joint_speed = joint_speed
        self.velocity = np.zeros(3)  # [vx, vy, omega] for controlled EE
        self.joint_velocity = 0.0    # Joint velocity
    
    def compute_velocity_toward_goal(self, current_pose, goal_pose, gain=0.3):
        # Position error
        pos_error = goal_pose[:2] - current_pose[:2]
        pos_dist = np.linalg.norm(pos_error)
        
        # Compute linear velocity (proportional with saturation)
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
    
    def compute_joint_velocity_toward_goal(self, current_q, goal_q, gain=0.3):
        q_error = goal_q - current_q
        self.joint_velocity = np.clip(q_error * gain, -self.joint_speed, self.joint_speed)
        return self.joint_velocity
    
    def get_velocity_array(self):
        return self.velocity.copy()
    
    def get_joint_velocity(self):
        return self.joint_velocity


class SE2ImpedanceDemo:
    """
    Demo for SE(2) Impedance Control.
    """

    def __init__(self, joint_type='revolute', render_mode='human'):
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

        # Robot Parameters (Calculated from GripperConfig)
        # Mass = 1.2 kg (base + fixed jaw + moving jaw)
        # Inertia = 97.6 kg*pixel^2 (about base center)
        self.robot_params = SE2RobotParams(
            mass=1.2,
            inertia=97.6
        )

        # Impedance Controller
        # Tuning for stable tracking
        self.controller = MultiGripperSE2ImpedanceController(
            num_grippers=2,
            robot_params=self.robot_params,
            M_d=np.diag([97.6, 1.2, 1.2]),     # Match robot inertia
            D_d=np.diag([50.0, 10.0, 10.0]),   # Damping (tuned)
            K_d=np.diag([200.0, 50.0, 50.0]),  # Stiffness (tuned)
            model_matching=True,
            max_force=100.0,
            max_torque=500.0  # Increased torque limit for higher inertia
        )

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
        print(f"\nResetting environment (Episode {self.episode_count + 1})...")

        obs, info = self.env.reset()

        self.desired_poses = obs['ee_poses'].copy()
        self.goal_ee_poses = self.env.goal_manager.get_goal_ee_poses()
        self.goal_joint_state = self.env.goal_manager.get_goal_joint_state()

        self.controller.reset()

        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        print("Environment reset complete!")

    def get_current_state(self, obs):
        return {
            'ee_poses': obs['ee_poses'],
            'ee_velocities': obs['ee_velocities'],
            'external_wrenches': obs['external_wrenches'],
            'joint_state': self.env.object_manager.get_joint_state(),
            'link_poses': obs['link_poses']
        }

    def point_velocity_to_body_twist(self, pose, point_velocity):
        """
        Convert world frame point velocity [vx, vy, omega] to body twist [omega, vx_b, vy_b].
        """
        vx_world, vy_world, omega = point_velocity
        theta = pose[2]
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        vx_body = cos_theta * vx_world + sin_theta * vy_world
        vy_body = -sin_theta * vx_world + cos_theta * vy_world
        
        return np.array([omega, vx_body, vy_body])

    def draw_info_overlay(self, screen, state):
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

        y_offset = draw_text(f"=== SE(2) Impedance Demo ({self.joint_type.upper()}) ===", y_offset, self.font, (0, 0, 128))
        y_offset += 5
        y_offset = draw_text(f"Step: {self.step_count} | Reward: {self.total_reward:.2f}", y_offset)
        
        if hasattr(self, 'goal_ee_poses'):
            err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
            err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
            y_offset = draw_text(f"Goal Error: L={err_left:.1f}px R={err_right:.1f}px", y_offset, color=(200, 0, 0) if err_left > 50 else (0, 100, 0))
        
        y_offset += 5
        for i in range(2):
            pose = state['ee_poses'][i]
            y_offset = draw_text(f"EE {i}: x={pose[0]:.1f} y={pose[1]:.1f} θ={pose[2]:.2f}", y_offset)
            
            # Show external wrench
            wrench = state['external_wrenches'][i]
            y_offset = draw_text(f"  F_ext: τ={wrench[0]:.1f} fx={wrench[1]:.1f} fy={wrench[2]:.1f}", y_offset, color=(100, 0, 100))

        y_offset += 10
        y_offset = draw_text("R: Reset | ESC: Exit", y_offset, color=(100, 100, 100))

    def run(self):
        print("\n" + "="*60)
        print("Starting SE(2) Impedance Demo")
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
            
            # Compute desired velocity (world frame point velocity)
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

            # Prepare inputs for impedance controller
            current_poses = state['ee_poses']
            
            # Convert velocities to body twists
            current_twists = []
            desired_twists = []
            
            desired_velocities_world = np.zeros((2, 3))
            desired_velocities_world[self.controlled_ee_idx] = controlled_velocity
            desired_velocities_world[other_ee_idx] = other_velocity
            
            for i in range(2):
                # Current twist
                current_twists.append(
                    self.point_velocity_to_body_twist(current_poses[i], state['ee_velocities'][i])
                )
                # Desired twist (using desired pose orientation for transform, or current? usually current for control)
                # But here we are generating desired velocity in world frame.
                # Let's use current pose for transform to be consistent with how we apply it.
                desired_twists.append(
                    self.point_velocity_to_body_twist(current_poses[i], desired_velocities_world[i])
                )
            
            current_twists = np.array(current_twists)
            desired_twists = np.array(desired_twists)
            
            # Compute wrenches
            wrenches = self.controller.compute_wrenches(
                current_poses,
                self.desired_poses,
                current_twists,
                desired_twists,
                desired_accels=None, # Assume zero accel for now
                external_wrenches=state['external_wrenches']
            )

            action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.step_count += 1
            self.total_reward += reward

            if self.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption("SWIVL SE(2) Impedance Demo")
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()

                screen_surface = self.env._draw()
                self.env.window.blit(screen_surface, screen_surface.get_rect())
                self.draw_info_overlay(self.env.window, state)
                pygame.display.flip()

            if self.step_count % 100 == 0:
                err_left = np.linalg.norm(state['ee_poses'][0, :2] - self.goal_ee_poses[0, :2])
                err_right = np.linalg.norm(state['ee_poses'][1, :2] - self.goal_ee_poses[1, :2])
                print(f"[Step {self.step_count:4d}] Goal Error: L={err_left:.1f}px R={err_right:.1f}px")

            clock.tick(self.env.metadata['render_fps'])

            if terminated or truncated:
                print(f"\nEpisode finished!")
                self.reset_environment()

        self.env.close()
        pygame.quit()
        print("\nDemo completed!")


def main():
    parser = argparse.ArgumentParser(description='SE(2) Impedance Demo')
    parser.add_argument('joint_type', nargs='?', default='revolute',
                        choices=['revolute', 'prismatic', 'fixed'])
    args = parser.parse_args()

    demo = SE2ImpedanceDemo(joint_type=args.joint_type, render_mode='human')
    demo.run()


if __name__ == "__main__":
    main()
