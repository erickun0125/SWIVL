"""
Rule-based Demo for Bimanual Manipulation
Automatically moves object from initial pose to desired pose using kinematic constraints.
"""

import argparse
import numpy as np
import pygame
import sys
import time

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains
from src.se2_math import SE2Pose, normalize_angle, se2_inverse


class RuleBasedDemo:
    """Demo with rule-based trajectory generation and constrained control."""

    def __init__(self, joint_type='revolute', render_mode='human'):
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

        # Create PD controller
        gains = PDGains(
            kp_linear=1500.0,
            kd_linear=1000.0,
            kp_angular=750.0,
            kd_angular=500.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)

        # Demo settings
        self.controlled_ee_idx = 0  # Master arm (Left)
        self.duration = 5.0  # Trajectory duration (seconds)
        self.hold_time = 2.0  # Time to hold at goal
        
        # Desired State (Goal) - Bottom Right
        # Link 1 Pose: [x, y, theta]
        self.goal_link1_pose = np.array([380.0, 380.0, np.pi/4])
        
        # Goal Joint State
        if self.joint_enum == JointType.REVOLUTE:
            self.goal_joint_state = 0.5  # radians
        elif self.joint_enum == JointType.PRISMATIC:
            self.goal_joint_state = 20.0  # pixels
        else:
            self.goal_joint_state = 0.0

        # Statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count = 0

        # Initialize
        self.reset_environment()

    def reset_environment(self):
        """Reset environment and trajectories."""
        print(f"\nResetting environment (Episode {self.episode_count + 1})...")

        # Reset environment (random initial pose)
        obs, _ = self.env.reset()
        
        # --- Kinematic Calibration ---
        # Update Grasping Frames based on ACTUAL grasped pose (Relative Pose)
        # This prevents kinematic inconsistency between 'ideal' grasp and 'actual' grasp
        link_poses = obs['link_poses']
        ee_poses = obs['ee_poses']
        
        # Left EE (Master) -> Link 1
        T_link1 = SE2Pose.from_array(link_poses[0]).to_matrix()
        T_ee_left = SE2Pose.from_array(ee_poses[0]).to_matrix()
        # T_local = inv(T_link) @ T_ee
        T_local_left = se2_inverse(T_link1) @ T_ee_left
        
        # Right EE (Slave) -> Link 2
        T_link2 = SE2Pose.from_array(link_poses[1]).to_matrix()
        T_ee_right = SE2Pose.from_array(ee_poses[1]).to_matrix()
        T_local_right = se2_inverse(T_link2) @ T_ee_right
        
        # Update ObjectManager's grasping frames in-place
        frames = self.env.object_manager.object.grasping_frames
        frames["left"].local_pose = SE2Pose.from_matrix(T_local_left).to_array()
        frames["right"].local_pose = SE2Pose.from_matrix(T_local_right).to_array()
        
        print("Calibrated grasping frames to match actual initial grasp.")
        
        # Capture Initial State
        self.start_time = time.time()
        self.start_link_poses = obs['link_poses'].copy()
        self.start_joint_state = self.env.object_manager.get_joint_state()
        
        # Calculate Start Link 1 Pose
        self.start_link1_pose = self.start_link_poses[0]
        
        # Calculate Goal Link Poses and EE Poses
        self.goal_link_poses = self._compute_all_link_poses(
            self.goal_link1_pose, self.goal_joint_state
        )
        self.goal_ee_poses = self._compute_ee_poses(self.goal_link_poses)
        
        # Initialize Trajectory Current State
        self.current_desired_ee_poses = obs['ee_poses'].copy()
        
        self.controller.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

    def _compute_all_link_poses(self, link1_pose, joint_state):
        """
        Compute both link poses from link1 pose and joint state.
        
        Matches the kinematic structure defined in object_manager.py:
        - Revolute: joint at link1's right end, link2 rotates around it
        - Prismatic: joint_state = position of link2's left end in link1 frame
        - Fixed: link2 rigidly attached at link1's right end
        """
        cfg = self.env.object_config
        L = cfg.link_length
        
        if self.joint_enum == JointType.REVOLUTE:
            # Link2 orientation = link1 orientation + joint angle
            link2_theta = link1_pose[2] + joint_state
            
            # Joint position is at link1's right end (L/2 from center along link axis)
            cos1, sin1 = np.cos(link1_pose[2]), np.sin(link1_pose[2])
            joint_x = link1_pose[0] + (L / 2) * cos1
            joint_y = link1_pose[1] + (L / 2) * sin1
            
            # Link2 center is L/2 from joint along link2's axis
            cos2, sin2 = np.cos(link2_theta), np.sin(link2_theta)
            link2_x = joint_x + (L / 2) * cos2
            link2_y = joint_y + (L / 2) * sin2
            link2_pose = np.array([link2_x, link2_y, link2_theta])
            
        elif self.joint_enum == JointType.PRISMATIC:
            # Prismatic: both links stay parallel (same orientation)
            # joint_state = x-position of link2's left end in link1's frame
            # 
            # In object_manager.py get_joint_state():
            #   link2_left = Vec2d(-L/2, 0)  # link2's left end in link2 frame
            #   pos_in_link1 = link1.world_to_local(link2.local_to_world(link2_left))
            #   return pos_in_link1.x
            #
            # So link2_left_in_link1 = (joint_state, 0)
            # Link2 center in link1 frame = (joint_state + L/2, 0)
            # 
            # Transform to world:
            cos1, sin1 = np.cos(link1_pose[2]), np.sin(link1_pose[2])
            offset_in_link1 = joint_state + L / 2  # distance from link1 center to link2 center
            link2_x = link1_pose[0] + offset_in_link1 * cos1
            link2_y = link1_pose[1] + offset_in_link1 * sin1
            link2_pose = np.array([link2_x, link2_y, link1_pose[2]])
            
        else:  # FIXED
            # Link2 rigidly attached at link1's right end
            # Link2 center is L from link1 center (L/2 + L/2)
            cos1, sin1 = np.cos(link1_pose[2]), np.sin(link1_pose[2])
            link2_x = link1_pose[0] + L * cos1
            link2_y = link1_pose[1] + L * sin1
            link2_pose = np.array([link2_x, link2_y, link1_pose[2]])
            
        return np.array([link1_pose, link2_pose])

    def _compute_ee_poses(self, link_poses):
        """Compute desired EE poses from link poses using grasping frames."""
        frames = self.env.object_manager.object.grasping_frames
        
        ee_poses = np.zeros((2, 3))
        
        # Left EE -> Link 1
        f_left = frames["left"]
        T_link1 = SE2Pose.from_array(link_poses[0]).to_matrix()
        T_grasp_local = SE2Pose.from_array(f_left.local_pose).to_matrix()
        T_grasp_world = T_link1 @ T_grasp_local
        ee_poses[0] = SE2Pose.from_matrix(T_grasp_world).to_array()
        
        # Right EE -> Link 2
        f_right = frames["right"]
        T_link2 = SE2Pose.from_array(link_poses[1]).to_matrix()
        T_grasp_local = SE2Pose.from_array(f_right.local_pose).to_matrix()
        T_grasp_world = T_link2 @ T_grasp_local
        ee_poses[1] = SE2Pose.from_matrix(T_grasp_world).to_array()
        
        return ee_poses

    def get_trajectory_state(self, t):
        """Get desired state at time t."""
        if t >= self.duration:
            return self.goal_link1_pose, self.goal_joint_state, np.zeros(3), 0.0
            
        # Linear interpolation with smooth step (cosine)
        alpha = (1 - np.cos(np.pi * t / self.duration)) / 2
        
        # Interpolate Link 1 Pose
        pos_start = self.start_link1_pose[:2]
        pos_goal = self.goal_link1_pose[:2]
        curr_pos = pos_start + (pos_goal - pos_start) * alpha
        
        # Interpolate Angle (shortest path)
        theta_start = self.start_link1_pose[2]
        theta_goal = self.goal_link1_pose[2]
        diff = normalize_angle(theta_goal - theta_start)
        curr_theta = theta_start + diff * alpha
        
        curr_link1_pose = np.array([curr_pos[0], curr_pos[1], curr_theta])
        
        # Interpolate Joint
        curr_joint = self.start_joint_state + (self.goal_joint_state - self.start_joint_state) * alpha
        
        # Compute Velocities
        # Derivative of alpha: d(alpha)/dt = (pi/T) * sin(pi*t/T) / 2
        alpha_dot = (np.pi / self.duration) * np.sin(np.pi * t / self.duration) / 2
        
        vel_pos = (pos_goal - pos_start) * alpha_dot
        vel_theta = diff * alpha_dot
        
        vel_link1 = np.array([vel_pos[0], vel_pos[1], vel_theta])
        vel_joint = (self.goal_joint_state - self.start_joint_state) * alpha_dot
        
        return curr_link1_pose, curr_joint, vel_link1, vel_joint

    def compute_constrained_velocity(self, controlled_ee_idx, controlled_velocity, joint_velocity, 
                                       link1_pose, link2_pose, joint_pos):
        """
        Compute the other EE's velocity based on kinematic constraints.
        
        When one EE (Master) is controlled with a specified velocity and joint velocity,
        the other EE (Slave) must follow the kinematic constraint of the articulated object.
        
        Args:
            controlled_ee_idx: 0 (left/master) or 1 (right/slave)
            controlled_velocity: [vx, vy, omega] of controlled EE in world frame
            joint_velocity: Joint velocity (rad/s for revolute, pixels/s for prismatic)
            link1_pose: Current link1 pose [x, y, theta]
            link2_pose: Current link2 pose [x, y, theta]
            joint_pos: Joint position in world frame [x, y]
        
        Returns:
            other_velocity: [vx, vy, omega] of other EE in world frame
        """
        joint_type = self.joint_enum
        
        # Get grasp offsets from calibrated grasping frames
        grasp_frames = self.env.object_manager.object.grasping_frames
        left_local = grasp_frames["left"].local_pose  # [x, y, theta] in link1 frame
        right_local = grasp_frames["right"].local_pose  # [x, y, theta] in link2 frame
        
        # Convert local grasp position to offset vector in world frame
        # Left EE offset from Link1 center (in world frame)
        cos1, sin1 = np.cos(link1_pose[2]), np.sin(link1_pose[2])
        left_offset = np.array([
            cos1 * left_local[0] - sin1 * left_local[1],
            sin1 * left_local[0] + cos1 * left_local[1]
        ])
        
        # Right EE offset from Link2 center (in world frame)
        cos2, sin2 = np.cos(link2_pose[2]), np.sin(link2_pose[2])
        right_offset = np.array([
            cos2 * right_local[0] - sin2 * right_local[1],
            sin2 * right_local[0] + cos2 * right_local[1]
        ])

        if joint_type == JointType.FIXED:
            # Fixed joint: both links move together, same velocity for both EEs
            return controlled_velocity.copy()
            
        elif joint_type == JointType.REVOLUTE:
            # Revolute: rotation around joint center
            # Joint is at the connection point between links
            
            if controlled_ee_idx == 0:  # Left EE (Master) controls Link1
                omega_link1 = controlled_velocity[2]
                
                # Link1 center velocity: V_link1 = V_ee - ω × r_ee
                # where r_ee is offset from link center to EE
                v_link1 = controlled_velocity[:2] - omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                # Joint point velocity: V_joint = V_link1 + ω × r_joint
                joint_offset = joint_pos - link1_pose[:2]
                v_joint = v_link1 + omega_link1 * np.array([-joint_offset[1], joint_offset[0]])
                
                # Link2 angular velocity = Link1 angular velocity + joint velocity
                omega_link2 = omega_link1 + joint_velocity
                
                # Link2 center velocity from joint
                link2_from_joint = link2_pose[:2] - joint_pos
                v_link2 = v_joint + omega_link2 * np.array([-link2_from_joint[1], link2_from_joint[0]])
                
                # Right EE velocity
                v_right = v_link2 + omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                return np.array([v_right[0], v_right[1], omega_link2])
                
            else:  # Right EE (Master) controls Link2
                omega_link2 = controlled_velocity[2]
                
                # Link2 center velocity
                v_link2 = controlled_velocity[:2] - omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                # Joint point velocity from Link2
                link2_from_joint = link2_pose[:2] - joint_pos
                v_joint = v_link2 - omega_link2 * np.array([-link2_from_joint[1], link2_from_joint[0]])
                
                # Link1 angular velocity = Link2 angular velocity - joint velocity
                omega_link1 = omega_link2 - joint_velocity
                
                # Link1 center velocity from joint
                joint_offset = joint_pos - link1_pose[:2]
                v_link1 = v_joint - omega_link1 * np.array([-joint_offset[1], joint_offset[0]])
                
                # Left EE velocity
                v_left = v_link1 + omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                return np.array([v_left[0], v_left[1], omega_link1])
                
        elif joint_type == JointType.PRISMATIC:
            # Prismatic: sliding along link1's x-axis, both links stay parallel
            slide_dir = np.array([cos1, sin1])
            perp_dir = np.array([-sin1, cos1])
            
            # Center distance = joint_state + L/2 (from _compute_all_link_poses)
            q = self.current_q_for_kinematics
            center_dist = q + self.env.object_config.link_length / 2
            
            if controlled_ee_idx == 0:  # Left EE (Master) controls Link1
                omega_link1 = controlled_velocity[2]
                v_link1 = controlled_velocity[:2] - omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                # Both links have same angular velocity (prismatic constraint)
                omega_link2 = omega_link1
                
                # Link2 velocity includes sliding and rotation coupling
                # V_link2 = V_link1 + q_dot * slide_dir + ω * center_dist * perp_dir
                v_link2 = v_link1 + joint_velocity * slide_dir + omega_link1 * center_dist * perp_dir
                
                # Right EE velocity
                v_right = v_link2 + omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                return np.array([v_right[0], v_right[1], omega_link2])
                
            else:  # Right EE (Master) controls Link2
                omega_link2 = controlled_velocity[2]
                v_link2 = controlled_velocity[:2] - omega_link2 * np.array([-right_offset[1], right_offset[0]])
                
                # Both links have same angular velocity
                omega_link1 = omega_link2
                
                # Link1 velocity (inverse of the above)
                # V_link1 = V_link2 - q_dot * slide_dir - ω * center_dist * perp_dir
                v_link1 = v_link2 - joint_velocity * slide_dir - omega_link1 * center_dist * perp_dir
                
                # Left EE velocity
                v_left = v_link1 + omega_link1 * np.array([-left_offset[1], left_offset[0]])
                
                return np.array([v_left[0], v_left[1], omega_link1])
        
        # Fallback (should not reach here)
        return controlled_velocity.copy()

    def draw_desired_object(self, screen):
        """Draw outline of object at goal pose."""
        # Colors
        color = (0, 255, 0, 100) # Transparent Green
        
        cfg = self.env.object_config
        w, h = cfg.link_length, cfg.link_width
        
        # Local vertices
        local_verts = np.array([
            [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]
        ])
        
        for pose in self.goal_link_poses:
            x, y, theta = pose
            
            # Transform vertices
            R = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta), np.cos(theta)]])
            t = np.array([x, y])
            
            world_verts = (R @ local_verts.T).T + t
            
            # Draw polygon
            pygame.draw.polygon(screen, color, world_verts, 2)
            
            # Draw center
            pygame.draw.circle(screen, color[:3], (int(x), int(y)), 3)

    def draw_desired_ee_frames(self, screen, desired_poses):
        """
        Draw desired pose frames for both grippers.
        Red/Blue arrows indicating position and orientation.
        """
        if desired_poses is None:
            return
        
        # Colors for each gripper
        colors = [
            (65, 105, 225),   # RoyalBlue for left
            (220, 20, 60)     # Crimson for right
        ]
        
        for i, pose in enumerate(desired_poses):
            x, y, theta = pose
            
            # Skip if outside visible area
            if not (0 <= x <= 512 and 0 <= y <= 512):
                continue
            
            # Frame size
            arrow_length = 30
            arrow_width = 10
            
            # Arrow points in jaw direction (+y in body frame = theta + π/2)
            jaw_angle = theta + np.pi / 2
            cos_j, sin_j = np.cos(jaw_angle), np.sin(jaw_angle)
            
            # Arrow tip (jaw direction)
            tip_x = x + arrow_length * cos_j
            tip_y = y + arrow_length * sin_j
            
            # Arrow base corners
            base_left_x = x - arrow_width * sin_j
            base_left_y = y + arrow_width * cos_j
            base_right_x = x + arrow_width * sin_j
            base_right_y = y - arrow_width * cos_j
            
            arrow_points = [
                (int(tip_x), int(tip_y)),
                (int(base_left_x), int(base_left_y)),
                (int(base_right_x), int(base_right_y))
            ]
            
            # Draw ghost arrow (semi-transparent)
            arrow_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
            color_with_alpha = (*colors[i], 120)
            pygame.draw.polygon(arrow_surface, color_with_alpha, arrow_points)
            screen.blit(arrow_surface, (0, 0))
            
            # Draw outline
            pygame.draw.polygon(screen, colors[i], arrow_points, 2)
            
            # Draw origin
            pygame.draw.circle(screen, colors[i], (int(x), int(y)), 4)

    def draw_info_overlay(self, screen, t, curr_ee_poses, current_ee_poses, curr_link_poses):
        """Draw debug information overlay."""
        font = pygame.font.SysFont('monospace', 12)
        
        y = 10
        line_height = 16
        
        def draw_text(text, color=(0, 0, 0)):
            nonlocal y
            surface = font.render(text, True, color)
            # Background
            bg = pygame.Surface((surface.get_width() + 4, surface.get_height() + 2))
            bg.fill((255, 255, 255))
            bg.set_alpha(200)
            screen.blit(bg, (8, y - 1))
            screen.blit(surface, (10, y))
            y += line_height
        
        # Time and progress
        progress = min(t / self.duration, 1.0) * 100
        draw_text(f"Time: {t:.2f}s / {self.duration:.1f}s ({progress:.1f}%)", (0, 0, 128))
        
        # Joint state
        actual_q = self.env.object_manager.get_joint_state()
        target_q = self.current_q_for_kinematics
        if self.joint_enum == JointType.REVOLUTE:
            draw_text(f"Joint: {np.degrees(actual_q):.1f}° / {np.degrees(target_q):.1f}° (target)")
        else:
            draw_text(f"Joint: {actual_q:.1f} / {target_q:.1f} (target)")
        
        y += 5
        
        # EE tracking errors
        for i in range(2):
            name = "Left" if i == 0 else "Right"
            pos_err = np.linalg.norm(current_ee_poses[i, :2] - curr_ee_poses[i, :2])
            ang_err = np.abs(normalize_angle(current_ee_poses[i, 2] - curr_ee_poses[i, 2]))
            color = (0, 128, 0) if pos_err < 10 else (200, 0, 0)
            draw_text(f"{name} EE: pos_err={pos_err:.1f}px, ang_err={np.degrees(ang_err):.1f}°", color)
        
        y += 5
        
        # Controls
        draw_text("R: Reset | ESC: Exit", (100, 100, 100))

    def run(self):
        print("\nStarting Rule-based Demo...")
        print("Controls: R = Reset, ESC = Exit")
        print(f"Joint type: {self.joint_type}")
        print(f"Trajectory duration: {self.duration}s, hold time: {self.hold_time}s")
        
        pygame.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        
        running = True
        step_count = 0
        
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_environment()
                        step_count = 0

            # Time
            t = time.time() - self.start_time
            
            # 1. Get Trajectory Point
            curr_link1, curr_q, vel_link1, vel_q = self.get_trajectory_state(t)
            self.current_q_for_kinematics = curr_q  # Store for constraint computation
            
            # 2. Compute Desired EE Poses from current interpolated link poses
            curr_link_poses = self._compute_all_link_poses(curr_link1, curr_q)
            curr_ee_poses = self._compute_ee_poses(curr_link_poses)
            
            # 3. Compute Velocities
            # Master Arm (Left, idx 0) Velocity
            # V_master = V_link1 + w1 x r_master
            grasp_frames = self.env.object_manager.object.grasping_frames
            left_local = grasp_frames["left"].local_pose
            
            # Offset in world frame
            theta = curr_link1[2]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            left_offset = np.array([
                cos_t * left_local[0] - sin_t * left_local[1],
                sin_t * left_local[0] + cos_t * left_local[1]
            ])
            
            omega = vel_link1[2]
            v_master_linear = vel_link1[:2] + omega * np.array([-left_offset[1], left_offset[0]])
            v_master = np.array([v_master_linear[0], v_master_linear[1], omega])
            
            # 4. Compute Slave Arm (Right, idx 1) Velocity using Constraints
            cfg = self.env.object_config
            joint_pos = curr_link1[:2] + (cfg.link_length / 2) * np.array([cos_t, sin_t])
            
            v_slave = self.compute_constrained_velocity(
                self.controlled_ee_idx, v_master, vel_q, 
                curr_link_poses[0], curr_link_poses[1], joint_pos
            )
            
            desired_velocities = np.array([v_master, v_slave])
            
            # 5. PD Control
            obs = self.env.get_obs()
            current_ee_poses = obs['ee_poses']
            current_velocities = obs['ee_velocities']  # [vx, vy, omega] point velocities
            
            wrenches = self.controller.compute_wrenches(
                current_ee_poses,
                curr_ee_poses,
                desired_velocities,
                current_velocities
            )
            
            action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1
            
            # Logging (every 50 steps)
            if step_count % 50 == 0:
                pos_err_left = np.linalg.norm(current_ee_poses[0, :2] - curr_ee_poses[0, :2])
                pos_err_right = np.linalg.norm(current_ee_poses[1, :2] - curr_ee_poses[1, :2])
                print(f"[Step {step_count:4d}] t={t:.2f}s | "
                      f"L_err={pos_err_left:.1f}px R_err={pos_err_right:.1f}px | "
                      f"q={self.env.object_manager.get_joint_state():.2f}")
            
            # Render
            if self.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption("SWIVL Rule-based Demo")
                
                screen = self.env._draw()
                
                # Draw Goal Object
                self.draw_desired_object(screen)
                
                # Draw Desired EE frames (Command visualization)
                self.draw_desired_ee_frames(screen, curr_ee_poses)
                
                # Draw info overlay
                self.draw_info_overlay(screen, t, curr_ee_poses, current_ee_poses, curr_link_poses)
                
                self.env.window.blit(screen, (0, 0))
                pygame.display.flip()
                
            clock.tick(self.env.control_hz)  # Match environment control rate
            
            # Reset when trajectory completed
            if t > self.duration + self.hold_time:
                print(f"\nTrajectory completed. Final errors:")
                pos_err_left = np.linalg.norm(current_ee_poses[0, :2] - self.goal_ee_poses[0, :2])
                pos_err_right = np.linalg.norm(current_ee_poses[1, :2] - self.goal_ee_poses[1, :2])
                print(f"  Left EE: {pos_err_left:.1f}px, Right EE: {pos_err_right:.1f}px")
                self.reset_environment()
                step_count = 0
        
        # Cleanup
        self.env.close()
        pygame.quit()
        print("\nDemo finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('joint_type', nargs='?', default='revolute', choices=['revolute', 'prismatic', 'fixed'])
    args = parser.parse_args()
    
    demo = RuleBasedDemo(joint_type=args.joint_type)
    demo.run()

