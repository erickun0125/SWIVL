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
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.se2_math import normalize_angle, integrate_velocity
# from src.envs.grasping_frames import GraspingFrameManager # Removed: Not available

class KinematicConstraintSolver:
    """Helper to compute required velocity for the secondary gripper."""
    def __init__(self, joint_type: str, link_length: float):
        self.joint_type = joint_type
        self.link_length = link_length

    def compute_desired_gripper_velocity(
        self,
        controlled_gripper: str,
        controlled_velocity: np.ndarray,
        joint_velocity: float,
        link1_pose: np.ndarray,
        link2_pose: np.ndarray,
        link1_velocity: np.ndarray,
        link2_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity for the follower gripper to maintain grasp constraint.
        
        Simplified logic: Assuming grippers are attached to links.
        Velocity of follower EE = Velocity of follower Link at grasp point.
        """
        # 1. Determine which link is controlled and which is follower
        if controlled_gripper == "left":
            # Left EE -> Link 1 (usually)
            follower_link_pose = link2_pose
            follower_link_idx = 1
        else:
            # Right EE -> Link 2
            follower_link_pose = link1_pose
            follower_link_idx = 0
            
        # TODO: Implement full SE(2) Jacobian-based inverse kinematics here.
        # For now, returning zero velocity for safety if logic is complex.
        # In a real implementation, we would use the Jacobians J_left, J_right
        # and the object Jacobian J_obj to solve:
        # V_follower = J_follower * V_joint_space
        
        return np.zeros(3)


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
        
        # Use internal solver instead of missing manager
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

        # Compute other EE velocity based on object constraints
        if current_link_velocities is None:
            current_link_velocities = np.zeros((2, 3))

        other_vel_world = self.constraint_solver.compute_desired_gripper_velocity(
            controlled_gripper=self.config.controlled_gripper,
            controlled_velocity=controlled_vel_world,
            joint_velocity=joint_vel,
            link1_pose=current_link_poses[0],
            link2_pose=current_link_poses[1],
            link1_velocity=current_link_velocities[0],
            link2_velocity=current_link_velocities[1]
        )

        # Build desired velocities array
        desired_velocities = np.zeros((2, 3))
        desired_velocities[controlled_idx] = controlled_vel_world
        desired_velocities[other_idx] = other_vel_world

        # Integrate velocities to get desired poses
        # If this is first call, use current poses as baseline
        if self.prev_desired_poses is None:
            self.prev_desired_poses = current_ee_poses.copy()

        desired_poses = np.zeros((2, 3))
        for i in range(2):
            # Integrate from previous desired pose
            desired_poses[i] = integrate_velocity(
                self.prev_desired_poses[i],
                desired_velocities[i],
                self.config.control_dt
            )

        # Store for next iteration
        self.prev_desired_poses = desired_poses.copy()

        # Desired accelerations are zero
        desired_accelerations = np.zeros((2, 3))

        return {
            'desired_poses': desired_poses,
            'desired_velocities': desired_velocities,
            'desired_accelerations': desired_accelerations
        }
