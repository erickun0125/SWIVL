"""
Goal Manager for Bimanual Manipulation Environment

Manages goal configuration for articulated objects, including:
- Goal link poses
- Goal joint states  
- Goal EE poses (computed from link poses)
- Random goal generation
- Goal visualization
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pygame

from src.envs.object_manager import JointType


@dataclass
class GoalConfig:
    """Configuration for goal state of articulated object."""
    
    # Whether to generate random goals each episode
    random_goals: bool = False
    
    # Goal pose for Link 1 (base link): [x, y, theta]
    # Default is centered and slightly rotated for visibility
    link1_pose: np.ndarray = field(default_factory=lambda: np.array([300.0, 300.0, np.pi/6]))
    
    # Goal joint state (radians for revolute, pixels for prismatic)
    joint_state: float = -2.0
    
    # Workspace bounds for random goal generation
    workspace_min: Tuple[float, float] = (180.0, 180.0)
    workspace_max: Tuple[float, float] = (332.0, 332.0)
    
    # Joint state bounds for random generation
    joint_min: float = -np.pi/3
    joint_max: float = np.pi/3


class GoalManager:
    """
    Manages goal configuration for bimanual manipulation.
    
    Computes consistent goal poses for links and end-effectors based on
    object kinematics and grasping frame configurations.
    """
    
    def __init__(
        self,
        joint_type: JointType,
        link_length: float,
        workspace_size: int = 512,
        config: Optional[GoalConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize goal manager.
        
        Args:
            joint_type: Type of joint (REVOLUTE, PRISMATIC, FIXED)
            link_length: Length of each link (pixels)
            workspace_size: Size of workspace (pixels, square)
            config: Goal configuration (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.joint_type = joint_type
        self.link_length = link_length
        self.workspace_size = workspace_size
        self.config = config if config is not None else GoalConfig()
        
        # Random number generator
        self.np_random = np.random.default_rng(seed)
        
        # Adjust joint bounds based on joint type (only if using default config)
        if config is None:
            if joint_type == JointType.PRISMATIC:
                self.config.joint_min = -15.0  # pixels
                self.config.joint_max = 15.0
                self.config.joint_state = 5.0  # Default for prismatic (conservative)
            elif joint_type == JointType.FIXED:
                self.config.joint_state = 0.0
                self.config.joint_min = 0.0
                self.config.joint_max = 0.0
        
        # Computed goal poses (set on reset)
        self.goal_link1_pose: np.ndarray = self.config.link1_pose.copy()
        self.goal_joint_state: float = self.config.joint_state
        self.goal_link_poses: Optional[np.ndarray] = None
        self.goal_ee_poses: Optional[np.ndarray] = None
        
        # Grasping frames (set when object manager provides them)
        self._grasping_frames: Optional[Dict] = None
    
    def set_grasping_frames(self, grasping_frames: Dict):
        """
        Set grasping frame configuration from object manager.
        
        Args:
            grasping_frames: Dictionary with 'left' and 'right' GraspingFrame objects
        """
        self._grasping_frames = grasping_frames
    
    def reset(self, grasping_frames: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
        """
        Reset goal configuration.
        
        If config.random_goals is True, generates new random goal.
        Otherwise uses default configuration.
        
        Args:
            grasping_frames: Optional grasping frames to update
            
        Returns:
            Tuple of (goal_link1_pose, goal_joint_state)
        """
        if grasping_frames is not None:
            self._grasping_frames = grasping_frames
        
        if self.config.random_goals:
            self._generate_random_goal()
        else:
            self.goal_link1_pose = self.config.link1_pose.copy()
            self.goal_joint_state = self.config.joint_state
        
        # Compute derived poses
        self._compute_goal_poses()
        
        return self.goal_link1_pose.copy(), self.goal_joint_state
    
    def _generate_random_goal(self):
        """Generate random goal within workspace bounds."""
        # Random Link1 position
        x = self.np_random.uniform(self.config.workspace_min[0], self.config.workspace_max[0])
        y = self.np_random.uniform(self.config.workspace_min[1], self.config.workspace_max[1])
        theta = self.np_random.uniform(-np.pi, np.pi)
        
        self.goal_link1_pose = np.array([x, y, theta])
        
        # Random joint state
        if self.joint_type != JointType.FIXED:
            self.goal_joint_state = self.np_random.uniform(
                self.config.joint_min, self.config.joint_max
            )
        else:
            self.goal_joint_state = 0.0
    
    def _compute_goal_poses(self):
        """Compute goal link poses and EE poses from link1 pose and joint state."""
        L = self.link_length
        link1 = self.goal_link1_pose
        
        # Compute Link2 pose based on joint type
        if self.joint_type == JointType.REVOLUTE:
            link2_theta = link1[2] + self.goal_joint_state
            cos1, sin1 = np.cos(link1[2]), np.sin(link1[2])
            joint_x = link1[0] + (L / 2) * cos1
            joint_y = link1[1] + (L / 2) * sin1
            cos2, sin2 = np.cos(link2_theta), np.sin(link2_theta)
            link2_x = joint_x + (L / 2) * cos2
            link2_y = joint_y + (L / 2) * sin2
            link2 = np.array([link2_x, link2_y, link2_theta])
            
        elif self.joint_type == JointType.PRISMATIC:
            cos1, sin1 = np.cos(link1[2]), np.sin(link1[2])
            # Link2 center offset from Link1 center
            offset = self.goal_joint_state + L / 2
            link2_x = link1[0] + offset * cos1
            link2_y = link1[1] + offset * sin1
            link2 = np.array([link2_x, link2_y, link1[2]])
            
        else:  # FIXED
            cos1, sin1 = np.cos(link1[2]), np.sin(link1[2])
            link2_x = link1[0] + L * cos1
            link2_y = link1[1] + L * sin1
            link2 = np.array([link2_x, link2_y, link1[2]])
        
        self.goal_link_poses = np.array([link1, link2])
        
        # Compute EE poses if grasping frames are available
        if self._grasping_frames is not None:
            self._compute_goal_ee_poses(link1, link2)
    
    def _compute_goal_ee_poses(self, link1_pose: np.ndarray, link2_pose: np.ndarray):
        """Compute goal EE poses from link poses and grasping frames."""
        frames = self._grasping_frames
        
        # Left EE -> Link 1
        left_local = frames["left"].local_pose
        cos1, sin1 = np.cos(link1_pose[2]), np.sin(link1_pose[2])
        left_x = link1_pose[0] + cos1 * left_local[0] - sin1 * left_local[1]
        left_y = link1_pose[1] + sin1 * left_local[0] + cos1 * left_local[1]
        left_theta = link1_pose[2] + left_local[2]
        
        # Right EE -> Link 2
        right_local = frames["right"].local_pose
        cos2, sin2 = np.cos(link2_pose[2]), np.sin(link2_pose[2])
        right_x = link2_pose[0] + cos2 * right_local[0] - sin2 * right_local[1]
        right_y = link2_pose[1] + sin2 * right_local[0] + cos2 * right_local[1]
        right_theta = link2_pose[2] + right_local[2]
        
        self.goal_ee_poses = np.array([
            [left_x, left_y, left_theta],
            [right_x, right_y, right_theta]
        ])
    
    def get_goal_link_poses(self) -> np.ndarray:
        """Get goal poses for both links."""
        if self.goal_link_poses is None:
            self._compute_goal_poses()
        return self.goal_link_poses.copy()
    
    def get_goal_ee_poses(self) -> Optional[np.ndarray]:
        """Get goal poses for both EEs (if grasping frames are set)."""
        if self.goal_ee_poses is None and self._grasping_frames is not None:
            self._compute_goal_poses()
        return self.goal_ee_poses.copy() if self.goal_ee_poses is not None else None
    
    def get_goal_joint_state(self) -> float:
        """Get goal joint state."""
        return self.goal_joint_state
    
    def draw_goal(self, screen: pygame.Surface, draw_links: bool = True, draw_ee: bool = True):
        """
        Draw goal visualization on pygame surface.
        
        Args:
            screen: Pygame surface to draw on
            draw_links: Whether to draw goal link shapes
            draw_ee: Whether to draw goal EE markers
        """
        if self.goal_link_poses is None:
            return
        
        L = self.link_length
        ws = self.workspace_size
        
        # Semi-transparent green for goal
        goal_color = (0, 200, 100)
        goal_alpha = 80
        
        if draw_links:
            # Create transparent surface
            overlay = pygame.Surface((ws, ws), pygame.SRCALPHA)
            
            for link_pose in self.goal_link_poses:
                x, y, theta = link_pose
                
                # Link rectangle corners
                half_w, half_h = L / 2, 7
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                
                corners = [
                    (x + cos_t * half_w - sin_t * half_h, y + sin_t * half_w + cos_t * half_h),
                    (x - cos_t * half_w - sin_t * half_h, y - sin_t * half_w + cos_t * half_h),
                    (x - cos_t * half_w + sin_t * half_h, y - sin_t * half_w - cos_t * half_h),
                    (x + cos_t * half_w + sin_t * half_h, y + sin_t * half_w - cos_t * half_h),
                ]
                corners = [(int(cx), int(cy)) for cx, cy in corners]
                
                pygame.draw.polygon(overlay, (*goal_color, goal_alpha), corners)
                pygame.draw.polygon(overlay, goal_color, corners, 2)
            
            screen.blit(overlay, (0, 0))
        
        if draw_ee and self.goal_ee_poses is not None:
            # Draw EE goal markers (crosses)
            for i, pose in enumerate(self.goal_ee_poses):
                x, y, theta = pose
                if not (0 <= x <= ws and 0 <= y <= ws):
                    continue
                
                # Cross marker
                size = 12
                pygame.draw.line(screen, goal_color, 
                               (int(x - size), int(y)), (int(x + size), int(y)), 2)
                pygame.draw.line(screen, goal_color,
                               (int(x), int(y - size)), (int(x), int(y + size)), 2)
                
                # Orientation indicator
                jaw_angle = theta + np.pi / 2
                end_x = x + 18 * np.cos(jaw_angle)
                end_y = y + 18 * np.sin(jaw_angle)
                pygame.draw.line(screen, goal_color,
                               (int(x), int(y)), (int(end_x), int(end_y)), 2)

