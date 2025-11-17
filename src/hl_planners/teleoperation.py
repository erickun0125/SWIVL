"""
High-Level Keyboard Input Planner

Provides teleoperation interface for bimanual manipulation:
- Maps keyboard inputs to velocity commands
- Controls one end-effector directly via keyboard
- Automatically computes desired poses for other end-effectors to maintain grasp
- Integrates with linkage object configuration

Keyboard mapping:
- Arrow keys: Linear velocity (up/down/left/right)
- Q/W: Angular velocity (counterclockwise/clockwise)
- 1/2: Switch controlled end-effector
- Space: Reset velocity to zero
- ESC: Exit
"""

import numpy as np
import pygame
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .linkage_manager import LinkageObject


@dataclass
class VelocityCommand:
    """Velocity command in body frame."""
    linear_x: float = 0.0   # Forward/backward velocity
    linear_y: float = 0.0   # Left/right velocity
    angular: float = 0.0     # Angular velocity


class KeyboardInputManager:
    """
    Manages keyboard input and converts to velocity commands.

    Keyboard mapping:
    - Arrow Up: Forward (+x in body frame)
    - Arrow Down: Backward (-x in body frame)
    - Arrow Left: Left (+y in body frame)
    - Arrow Right: Right (-y in body frame)
    - Q: Rotate counterclockwise (+angular)
    - W: Rotate clockwise (-angular)
    """

    def __init__(
        self,
        max_linear_vel: float = 50.0,
        max_angular_vel: float = 1.0,
        vel_increment: float = 10.0,
        ang_increment: float = 0.2
    ):
        """
        Initialize keyboard input manager.

        Args:
            max_linear_vel: Maximum linear velocity magnitude
            max_angular_vel: Maximum angular velocity magnitude
            vel_increment: Linear velocity increment per key press
            ang_increment: Angular velocity increment per key press
        """
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.vel_increment = vel_increment
        self.ang_increment = ang_increment

        # Current velocity command
        self.velocity_cmd = VelocityCommand()

        # Key state tracking
        self.keys_pressed = set()

    def process_events(self, events: List[pygame.event.Event]) -> Dict[str, bool]:
        """
        Process pygame events and update velocity command.

        Args:
            events: List of pygame events

        Returns:
            Dictionary with action flags:
                - 'quit': Whether to quit
                - 'reset': Whether to reset velocity
                - 'switch_ee': Whether to switch controlled end-effector
        """
        actions = {
            'quit': False,
            'reset': False,
            'switch_ee': False,
            'ee_index': None
        }

        for event in events:
            if event.type == pygame.QUIT:
                actions['quit'] = True

            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)

                # Handle special actions
                if event.key == pygame.K_ESCAPE:
                    actions['quit'] = True
                elif event.key == pygame.K_SPACE:
                    actions['reset'] = True
                    self.reset_velocity()
                elif event.key == pygame.K_1:
                    actions['switch_ee'] = True
                    actions['ee_index'] = 0
                elif event.key == pygame.K_2:
                    actions['switch_ee'] = True
                    actions['ee_index'] = 1

            elif event.type == pygame.KEYUP:
                if event.key in self.keys_pressed:
                    self.keys_pressed.remove(event.key)

        # Update velocity based on currently pressed keys
        self._update_velocity_from_keys()

        return actions

    def _update_velocity_from_keys(self):
        """Update velocity command based on currently pressed keys."""
        # Linear velocity
        vx = 0.0
        vy = 0.0

        if pygame.K_UP in self.keys_pressed:
            vx += self.vel_increment
        if pygame.K_DOWN in self.keys_pressed:
            vx -= self.vel_increment
        if pygame.K_LEFT in self.keys_pressed:
            vy += self.vel_increment
        if pygame.K_RIGHT in self.keys_pressed:
            vy -= self.vel_increment

        # Clamp to max velocity
        vel_magnitude = np.sqrt(vx**2 + vy**2)
        if vel_magnitude > self.max_linear_vel:
            scale = self.max_linear_vel / vel_magnitude
            vx *= scale
            vy *= scale

        self.velocity_cmd.linear_x = vx
        self.velocity_cmd.linear_y = vy

        # Angular velocity
        omega = 0.0
        if pygame.K_q in self.keys_pressed:
            omega += self.ang_increment
        if pygame.K_w in self.keys_pressed:
            omega -= self.ang_increment

        # Clamp to max angular velocity
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)
        self.velocity_cmd.angular = omega

    def get_velocity_command(self) -> VelocityCommand:
        """Get current velocity command."""
        return self.velocity_cmd

    def reset_velocity(self):
        """Reset velocity command to zero."""
        self.velocity_cmd = VelocityCommand()

    def get_velocity_array(self) -> np.ndarray:
        """Get velocity command as numpy array [vx, vy, omega]."""
        return np.array([
            self.velocity_cmd.linear_x,
            self.velocity_cmd.linear_y,
            self.velocity_cmd.angular
        ])


class MultiEEPlanner:
    """
    High-level planner for multi-end-effector manipulation.

    Controls one end-effector via keyboard input and automatically
    computes desired poses for other end-effectors to maintain grasp
    on the articulated object.
    """

    def __init__(
        self,
        num_end_effectors: int,
        linkage_object: LinkageObject,
        control_dt: float = 0.1
    ):
        """
        Initialize multi-EE planner.

        Args:
            num_end_effectors: Number of end-effectors (grippers)
            linkage_object: LinkageObject being manipulated
            control_dt: Control timestep for velocity integration
        """
        self.num_ee = num_end_effectors
        self.linkage = linkage_object
        self.control_dt = control_dt

        # Current controlled end-effector index
        self.controlled_ee_idx = 0

        # Current desired poses for all end-effectors
        self.desired_poses = np.zeros((num_end_effectors, 3))

        # Keyboard input manager
        self.keyboard = KeyboardInputManager()

    def initialize_from_current_state(self, current_ee_poses: np.ndarray):
        """
        Initialize desired poses from current end-effector poses.

        Args:
            current_ee_poses: Array of shape (num_ee, 3) with current poses
        """
        if current_ee_poses.shape[0] != self.num_ee:
            raise ValueError(f"Expected {self.num_ee} poses, got {current_ee_poses.shape[0]}")

        self.desired_poses = current_ee_poses.copy()

    def update(
        self,
        events: List[pygame.event.Event],
        current_ee_poses: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Update planner with keyboard input and compute desired poses.

        Args:
            events: List of pygame events
            current_ee_poses: Current end-effector poses (num_ee, 3)

        Returns:
            Tuple of:
                - Desired poses for all end-effectors (num_ee, 3)
                - Action flags from keyboard input
        """
        # Process keyboard input
        actions = self.keyboard.process_events(events)

        # Handle end-effector switching
        if actions['switch_ee'] and actions['ee_index'] is not None:
            self.controlled_ee_idx = actions['ee_index']
            print(f"Switched to controlling EE {self.controlled_ee_idx}")

        # Get velocity command
        vel_cmd = self.keyboard.get_velocity_array()

        # Update desired pose for controlled end-effector
        self._update_controlled_ee_pose(vel_cmd, current_ee_poses[self.controlled_ee_idx])

        # Compute desired poses for other end-effectors to maintain grasp
        self._compute_grasp_maintaining_poses(current_ee_poses)

        return self.desired_poses, actions

    def _update_controlled_ee_pose(
        self,
        velocity_command: np.ndarray,
        current_pose: np.ndarray
    ):
        """
        Update desired pose for controlled end-effector from velocity command.

        IMPORTANT: Velocity is integrated from DESIRED pose, not current pose!
        This ensures that when velocity is zero, desired pose stays fixed.

        Args:
            velocity_command: Velocity command [vx, vy, omega] in body frame
            current_pose: Current pose [x, y, theta] of controlled EE (for reference frame)
        """
        # Get PREVIOUS desired pose (not current pose!)
        prev_desired = self.desired_poses[self.controlled_ee_idx]
        x_des, y_des, theta_des = prev_desired

        vx_body, vy_body, omega = velocity_command

        # Transform body frame velocity to world frame
        # Use desired orientation, not current orientation
        cos_theta = np.cos(theta_des)
        sin_theta = np.sin(theta_des)

        vx_world = cos_theta * vx_body - sin_theta * vy_body
        vy_world = sin_theta * vx_body + cos_theta * vy_body

        # Integrate from DESIRED pose
        x_new = x_des + vx_world * self.control_dt
        y_new = y_des + vy_world * self.control_dt
        theta_new = theta_des + omega * self.control_dt

        # Normalize angle
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        # Update desired pose
        self.desired_poses[self.controlled_ee_idx] = np.array([
            x_new, y_new, theta_new
        ])

    def _compute_grasp_maintaining_poses(self, current_ee_poses: np.ndarray):
        """
        Compute desired poses for non-controlled end-effectors.

        These poses are automatically determined to maintain grasp on the
        linkage object based on its current configuration.

        Args:
            current_ee_poses: Current poses of all end-effectors
        """
        # Update linkage state
        self.linkage.update_joint_states()

        # Get suggested grasp poses from linkage
        grasp_poses = self.linkage.compute_grasp_poses(num_grippers=self.num_ee)

        # Update desired poses for non-controlled end-effectors
        for i in range(self.num_ee):
            if i != self.controlled_ee_idx and i < len(grasp_poses):
                # Use suggested grasp pose
                self.desired_poses[i] = grasp_poses[i]

    def set_controlled_ee(self, ee_idx: int):
        """Set which end-effector is controlled by keyboard."""
        if 0 <= ee_idx < self.num_ee:
            self.controlled_ee_idx = ee_idx
        else:
            raise ValueError(f"EE index {ee_idx} out of range")

    def get_desired_poses(self) -> np.ndarray:
        """Get desired poses for all end-effectors."""
        return self.desired_poses.copy()

    def get_controlled_ee_index(self) -> int:
        """Get index of currently controlled end-effector."""
        return self.controlled_ee_idx

    def reset(self):
        """Reset planner state."""
        self.keyboard.reset_velocity()


class CoordinatedMotionPlanner:
    """
    Advanced planner for coordinated bimanual motion.

    Supports different coordination modes:
    - Independent: Each EE controlled separately
    - Coordinated: EEs move together to manipulate object
    - Leader-Follower: One EE leads, others follow while maintaining grasp
    """

    def __init__(
        self,
        num_end_effectors: int,
        linkage_object: LinkageObject,
        control_dt: float = 0.1
    ):
        """Initialize coordinated motion planner."""
        self.num_ee = num_end_effectors
        self.linkage = linkage_object
        self.control_dt = control_dt

        # Coordination mode
        self.mode = "leader_follower"  # "independent", "coordinated", "leader_follower"

        # Leader EE for leader-follower mode
        self.leader_ee_idx = 0

        # Desired poses
        self.desired_poses = np.zeros((num_end_effectors, 3))

        # Keyboard input
        self.keyboard = KeyboardInputManager()

    def update_leader_follower(
        self,
        events: List[pygame.event.Event],
        current_ee_poses: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Update in leader-follower mode.

        Leader EE is controlled by keyboard, followers maintain grasp.

        Args:
            events: Pygame events
            current_ee_poses: Current EE poses

        Returns:
            Desired poses and action flags
        """
        # Process keyboard
        actions = self.keyboard.process_events(events)

        # Get velocity command for leader
        vel_cmd = self.keyboard.get_velocity_array()

        # Update leader pose
        leader_pose = current_ee_poses[self.leader_ee_idx]
        x, y, theta = leader_pose
        vx_body, vy_body, omega = vel_cmd

        # Transform to world frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        vx_world = cos_theta * vx_body - sin_theta * vy_body
        vy_world = sin_theta * vx_body + cos_theta * vy_body

        # Integrate
        self.desired_poses[self.leader_ee_idx] = np.array([
            x + vx_world * self.control_dt,
            y + vy_world * self.control_dt,
            theta + omega * self.control_dt
        ])

        # Compute follower poses to maintain grasp
        self._compute_follower_poses(current_ee_poses)

        return self.desired_poses, actions

    def _compute_follower_poses(self, current_ee_poses: np.ndarray):
        """
        Compute desired poses for follower end-effectors.

        Followers maintain their grasp on the object while the leader moves.
        """
        # Update linkage state
        self.linkage.update_joint_states()

        # Get grasp poses
        grasp_poses = self.linkage.compute_grasp_poses(num_grippers=self.num_ee)

        # Update follower poses
        for i in range(self.num_ee):
            if i != self.leader_ee_idx and i < len(grasp_poses):
                self.desired_poses[i] = grasp_poses[i]

    def update_coordinated(
        self,
        events: List[pygame.event.Event],
        current_ee_poses: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Update in coordinated mode.

        Both EEs move together to manipulate the object as a whole.
        Object centroid follows keyboard command.

        Args:
            events: Pygame events
            current_ee_poses: Current EE poses

        Returns:
            Desired poses and action flags
        """
        # Process keyboard
        actions = self.keyboard.process_events(events)

        # Get velocity command
        vel_cmd = self.keyboard.get_velocity_array()

        # Compute object centroid
        centroid = self.linkage.compute_object_centroid()

        # Move object centroid according to velocity command
        # For simplicity, use average orientation of EEs
        avg_theta = np.mean([pose[2] for pose in current_ee_poses])

        cos_theta = np.cos(avg_theta)
        sin_theta = np.sin(avg_theta)
        vx_body, vy_body, omega = vel_cmd

        vx_world = cos_theta * vx_body - sin_theta * vy_body
        vy_world = sin_theta * vx_body + cos_theta * vy_body

        # New centroid position
        new_centroid = centroid + np.array([vx_world, vy_world]) * self.control_dt

        # Compute displacement
        displacement = new_centroid - centroid

        # Move all EE desired poses by the same displacement
        for i in range(self.num_ee):
            self.desired_poses[i, :2] = current_ee_poses[i, :2] + displacement
            self.desired_poses[i, 2] = current_ee_poses[i, 2] + omega * self.control_dt

        return self.desired_poses, actions


def create_keyboard_teleoperation_system(
    num_end_effectors: int,
    linkage_object: LinkageObject,
    mode: str = "leader_follower"
) -> MultiEEPlanner:
    """
    Factory function to create keyboard teleoperation system.

    Args:
        num_end_effectors: Number of end-effectors
        linkage_object: LinkageObject being manipulated
        mode: Control mode ("leader_follower" or "coordinated")

    Returns:
        Configured planner instance
    """
    if mode == "leader_follower":
        planner = MultiEEPlanner(num_end_effectors, linkage_object)
    elif mode == "coordinated":
        planner = CoordinatedMotionPlanner(num_end_effectors, linkage_object)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return planner
