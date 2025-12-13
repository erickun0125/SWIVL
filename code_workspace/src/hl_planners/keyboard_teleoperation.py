"""
Keyboard Teleoperation Planner

This planner allows users to control one end-effector via keyboard input,
while automatically computing the other end-effector's motion to maintain
grasp on the articulated object.

Key features:
- Direct velocity control for one EE
- Object joint velocity control
- Automatic computation of other EE velocity based on object constraints
- Outputs desired velocity (not pose) to low-level controller
"""

import numpy as np
import pygame
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from src.se2_math import normalize_angle, integrate_velocity

# Note: We use simple rotation for velocity frame conversion, NOT the deprecated
# world_to_body_velocity which incorrectly includes p × ω terms.
# For point velocity: v_body = R^T @ v_world
# from src.envs.grasping_frames import GraspingFrameManager # Removed: Not available

class KinematicConstraintSolver:
    """Helper to compute required twists for both grippers."""

    def __init__(self, joint_type: str, link_length: float):
        self.joint_type = joint_type.lower()
        self.link_length = link_length

    @staticmethod
    def _omega_cross_r(omega: float, r: np.ndarray) -> np.ndarray:
        return omega * np.array([-r[1], r[0]])

    def _point_velocity(
        self,
        link_twist: np.ndarray,
        link_pose: np.ndarray,
        point: np.ndarray
    ) -> np.ndarray:
        """Velocity of a point rigidly attached to the link (spatial frame)."""
        r = point - link_pose[:2]
        return link_twist[:2] + self._omega_cross_r(link_twist[2], r)

    def _compute_joint_position(self, link1_pose: np.ndarray) -> np.ndarray:
        """Joint position in world frame (link1 right end)."""
        theta = link1_pose[2]
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        offset = rot @ np.array([self.link_length / 2.0, 0.0])
        return link1_pose[:2] + offset

    def compute_gripper_twists(
        self,
        controlled_idx: int,
        primary_twist: np.ndarray,
        joint_velocity: float,
        ee_poses: np.ndarray,
        link_poses: np.ndarray
    ) -> np.ndarray:
        """
        Compute spatial twists [vx, vy, omega] for both grippers.
        """
        gripper_twists = np.zeros((2, 3))
        gripper_twists[controlled_idx] = primary_twist

        joint_pos = self._compute_joint_position(link_poses[0])
        link_twists = [None, None]
        ctrl_link_idx = controlled_idx
        other_link_idx = 1 - controlled_idx

        # Convert controlled gripper twist to link twist
        link_twists[ctrl_link_idx] = self._link_twist_from_gripper(
            primary_twist,
            link_poses[ctrl_link_idx],
            ee_poses[ctrl_link_idx]
        )

        # Determine joint/angular relationships
        dq = joint_velocity
        if self.joint_type != "revolute":
            dq = joint_velocity  # interpreted as linear speed for prismatic

        link_twists = self._solve_other_link(
            link_twists,
            ctrl_link_idx,
            other_link_idx,
            dq,
            joint_pos,
            link_poses
        )

        # Convert link twists back to gripper twists
        for idx in range(2):
            if link_twists[idx] is None:
                continue
            gripper_twists[idx] = self._gripper_spatial_from_link(
                link_twists[idx],
                link_poses[idx],
                ee_poses[idx]
            )
        return gripper_twists

    def _link_twist_from_gripper(
        self,
        gripper_spatial_twist: np.ndarray,
        link_pose: np.ndarray,
        gripper_pose: np.ndarray
    ) -> np.ndarray:
        """Convert gripper spatial twist [vx, vy, omega] to link spatial twist."""
        omega = gripper_spatial_twist[2]
        v_point = gripper_spatial_twist[:2]
        r = gripper_pose[:2] - link_pose[:2]
        v_link = v_point - self._omega_cross_r(omega, r)
        return np.array([v_link[0], v_link[1], omega])

    def _gripper_spatial_from_link(
        self,
        link_twist: np.ndarray,
        link_pose: np.ndarray,
        gripper_pose: np.ndarray
    ) -> np.ndarray:
        """Convert link spatial twist to gripper spatial twist [vx, vy, omega]."""
        r = gripper_pose[:2] - link_pose[:2]
        omega = link_twist[2]
        v_point = link_twist[:2] + self._omega_cross_r(omega, r)
        return np.array([v_point[0], v_point[1], omega])

    def _solve_other_link(
        self,
        link_twists: List[Optional[np.ndarray]],
        ctrl_link_idx: int,
        other_link_idx: int,
        joint_velocity: float,
        joint_pos: np.ndarray,
        link_poses: np.ndarray
    ) -> List[Optional[np.ndarray]]:
        """Solve for follower link twist using joint constraints."""
        ctrl_twist = link_twists[ctrl_link_idx]
        if ctrl_twist is None:
            return link_twists

        # Velocity at joint from controlled link
        v_joint_ctrl = self._point_velocity(
            ctrl_twist,
            link_poses[ctrl_link_idx],
            joint_pos
        )

        # Determine follower angular velocity
        ctrl_omega = ctrl_twist[2]
        if self.joint_type == "revolute":
            if ctrl_link_idx == 0:
                omega_other = ctrl_omega + joint_velocity
            else:
                omega_other = ctrl_omega - joint_velocity
            v_joint_other = v_joint_ctrl
        elif self.joint_type == "prismatic":
            omega_other = ctrl_omega
            axis = np.array([np.cos(link_poses[0, 2]), np.sin(link_poses[0, 2])])
            shift = joint_velocity * axis
            if ctrl_link_idx == 0:
                v_joint_other = v_joint_ctrl + shift
            else:
                v_joint_other = v_joint_ctrl - shift
        else:  # fixed
            omega_other = ctrl_omega
            v_joint_other = v_joint_ctrl

        r_other = joint_pos - link_poses[other_link_idx][:2]
        v_other = v_joint_other - self._omega_cross_r(omega_other, r_other)
        link_twists[other_link_idx] = np.array([v_other[0], v_other[1], omega_other])
        return link_twists


@dataclass
class TeleopConfig:
    """Configuration for teleoperation."""
    linear_speed: float = 30.0  # pixels/s
    angular_speed: float = 1.0  # rad/s
    joint_speed: float = 0.5    # rad/s for revolute, pixels/s for prismatic
    control_dt: float = 0.1     # 10 Hz control
    controlled_gripper: str = "left"  # Which gripper is directly controlled


class KeyboardTeleoperationPlanner:
    """
    Keyboard teleoperation planner.

    Controls one EE with keyboard, automatically computes other EE motion.
    """

    def __init__(
        self,
        config: Optional[TeleopConfig] = None,
        joint_type: str = "revolute",
        link_length: float = 40.0
    ):
        """
        Initialize keyboard teleoperation planner.

        Args:
            config: Teleoperation configuration
            joint_type: Type of articulated object joint
            link_length: Length of links
        """
        self.config = config if config is not None else TeleopConfig()
        self.joint_type = joint_type
        
        # Use internal solver to enforce grasp constraints
        self.constraint_solver = KinematicConstraintSolver(joint_type, link_length)

        # Current desired velocities
        self.controlled_ee_velocity = np.zeros(3)  # [vx, vy, omega]
        self.joint_velocity = 0.0
        self.other_ee_velocity = np.zeros(3)

        # For integration
        self.prev_desired_poses = None

    def reset(self, initial_ee_poses: np.ndarray, initial_link_poses: np.ndarray):
        """
        Reset planner with initial state.

        Args:
            initial_ee_poses: Initial EE poses (2, 3)
            initial_link_poses: Initial link poses (2, 3)
        """
        self.prev_desired_poses = initial_ee_poses.copy()
        self.controlled_ee_velocity = np.zeros(3)
        self.joint_velocity = 0.0
        self.other_ee_velocity = np.zeros(3)

    def process_keyboard_input(self, keys: Dict[int, bool]) -> Tuple[np.ndarray, float]:
        """
        Process keyboard input and compute commanded velocities.

        Args:
            keys: Dictionary of key states {key_code: is_pressed}

        Returns:
            Tuple of (controlled_ee_velocity, joint_velocity)
        """
        # Linear velocity in body frame
        vx_body = 0.0
        vy_body = 0.0

        # Up/Down: forward/backward (x-axis in body frame)
        if keys.get(pygame.K_UP, False):
            vx_body += self.config.linear_speed
        if keys.get(pygame.K_DOWN, False):
            vx_body -= self.config.linear_speed

        # Left/Right: lateral (y-axis in body frame)
        if keys.get(pygame.K_LEFT, False):
            vy_body += self.config.linear_speed
        if keys.get(pygame.K_RIGHT, False):
            vy_body -= self.config.linear_speed

        # Q/W: rotate counter-clockwise/clockwise
        omega = 0.0
        if keys.get(pygame.K_q, False):
            omega += self.config.angular_speed
        if keys.get(pygame.K_w, False):
            omega -= self.config.angular_speed

        # A/D: joint motion (counter-clockwise/clockwise for revolute, extend/retract for prismatic)
        joint_vel = 0.0
        if keys.get(pygame.K_a, False):
            joint_vel += self.config.joint_speed
        if keys.get(pygame.K_d, False):
            joint_vel -= self.config.joint_speed

        # Commanded velocity is in body frame, need to convert to world frame
        # But we'll do this conversion using current pose later
        controlled_velocity = np.array([vx_body, vy_body, omega])

        return controlled_velocity, joint_vel

    def get_action(
        self,
        keyboard_events: Dict[int, bool],
        current_ee_poses: np.ndarray,
        current_link_poses: np.ndarray,
        current_ee_velocities: Optional[np.ndarray] = None,
        current_link_velocities: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get desired poses and velocities for both EEs.

        Args:
            keyboard_events: Keyboard input state
            current_ee_poses: Current EE poses (2, 3)
            current_link_poses: Current link poses (2, 3)
            current_ee_velocities: Current EE velocities (2, 3)
            current_link_velocities: Current link velocities (2, 3)

        Returns:
            Dictionary with:
                - 'desired_poses': (2, 3) array
                - 'desired_velocities': (2, 3) array
                - 'desired_accelerations': (2, 3) array (all zeros)
        """
        # Process keyboard input for controlled EE velocity (body frame)
        controlled_vel_body, joint_vel = self.process_keyboard_input(keyboard_events)

        # Get controlled EE index
        controlled_idx = 0 if self.config.controlled_gripper == "left" else 1
        other_idx = 1 - controlled_idx

        # Transform controlled velocity from body frame to world frame
        controlled_pose = current_ee_poses[controlled_idx]
        theta = controlled_pose[2]

        controlled_vel_world = np.array([
            np.cos(theta) * controlled_vel_body[0] - np.sin(theta) * controlled_vel_body[1],
            np.sin(theta) * controlled_vel_body[0] + np.cos(theta) * controlled_vel_body[1],
            controlled_vel_body[2]
        ])

        spatial_twists = self.constraint_solver.compute_gripper_twists(
            controlled_idx=controlled_idx,
            primary_twist=controlled_vel_world,
            joint_velocity=joint_vel,
            ee_poses=current_ee_poses,
            link_poses=current_link_poses
        )

        desired_body_twists = np.zeros_like(spatial_twists)

        # Integrate velocities to get desired poses
        # If this is first call, use current poses as baseline
        if self.prev_desired_poses is None:
            self.prev_desired_poses = current_ee_poses.copy()

        desired_poses = np.zeros((2, 3))
        for i in range(2):
            spatial = spatial_twists[i]  # [vx, vy, omega] in world frame
            pose = current_ee_poses[i]
            theta = pose[2]
            
            # Convert world frame velocity to body frame using simple rotation
            # v_body = R^T @ v_world (NO p × ω term for point velocity!)
            vx_world, vy_world, omega = spatial
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            vx_body = cos_theta * vx_world + sin_theta * vy_world
            vy_body = -sin_theta * vx_world + cos_theta * vy_world
            
            # MR convention: [omega, vx, vy]
            body_twist = np.array([omega, vx_body, vy_body])
            desired_body_twists[i] = body_twist
            
            desired_poses[i] = integrate_velocity(
                self.prev_desired_poses[i],
                body_twist,
                self.config.control_dt
            )

        # Store for next iteration
        self.prev_desired_poses = desired_poses.copy()

        # Desired accelerations are zero
        desired_accelerations = np.zeros((2, 3))

        return {
            'desired_poses': desired_poses,
            'desired_velocities': desired_body_twists.copy(),
            'desired_body_twists': desired_body_twists,
            'desired_spatial_twists': spatial_twists,
            'desired_accelerations': desired_accelerations
        }
