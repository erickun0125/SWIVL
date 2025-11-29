"""
SWIVL Hierarchical Policy Evaluation Script

Evaluates the complete SWIVL hierarchical control pipeline:
1. High-Level Policy (Layer 2): ACT/Diffusion/Flow Matching → desired poses
2. Low-Level RL Policy (Layer 3): PPO → impedance modulation variables
3. Screw-Decomposed Impedance Controller (Layer 4): → control wrenches

This script loads trained HL and LL policies and runs them together with
the screw-decomposed impedance controller.

Usage:
    python scripts/evaluation/eval_hierarchical_policy.py \
        --hl_policy act \
        --hl_checkpoint checkpoints/act_best.pth \
        --ll_checkpoint checkpoints/impedance_policy.zip \
        --num_episodes 10

Controls:
    SPACE - Pause/Resume
    R     - Reset episode
    I     - Toggle impedance info display
    ESC   - Exit
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import pygame
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs.biart import BiArtEnv
from src.trajectory_generator import CubicSplineTrajectory, MinimumJerkTrajectory
from src.ll_controllers.se2_screw_decomposed_impedance import (
    MultiGripperSE2ScrewDecomposedImpedanceController,
    ScrewDecomposedImpedanceParams
)
from src.se2_dynamics import SE2RobotParams
from scripts.training.train_hl_policy import create_policy, LinearNormalizer


@dataclass
class EvalMetrics:
    """Evaluation metrics for an episode."""
    episode_reward: float = 0.0
    episode_length: int = 0
    tracking_error: float = 0.0
    fighting_force: float = 0.0
    avg_d_parallel: float = 0.0
    avg_d_perp: float = 0.0
    avg_k_p: float = 0.0
    avg_alpha: float = 0.0
    success: bool = False


class HierarchicalPolicyEvaluator:
    """
    Evaluates the complete SWIVL hierarchical control pipeline.
    
    Components:
    - HL Policy: Generates action chunks (desired poses/trajectories)
    - LL Policy: Outputs impedance modulation variables
    - Controller: Computes control wrenches using impedance parameters
    """
    
    def __init__(
        self,
        hl_policy_type: str,
        hl_checkpoint_path: str,
        ll_checkpoint_path: str,
        hl_config_path: str = 'scripts/configs/hl_policy_config.yaml',
        ll_config_path: str = 'scripts/configs/rl_config.yaml',
        device: str = 'auto',
        render_mode: str = 'human',
        execution_horizon: int = 10
    ):
        """
        Initialize hierarchical policy evaluator.
        
        Args:
            hl_policy_type: Type of HL policy ('act', 'diffusion', 'flow_matching')
            hl_checkpoint_path: Path to HL policy checkpoint
            ll_checkpoint_path: Path to LL policy checkpoint (SB3 PPO)
            hl_config_path: Path to HL policy config
            ll_config_path: Path to LL config for environment/controller params
            device: Computation device
            render_mode: Rendering mode
            execution_horizon: Steps to execute before re-planning
        """
        self.device = torch.device(
            device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"Using device: {self.device}")
        
        self.execution_horizon = execution_horizon
        
        # Load configurations
        self.hl_config = self._load_config(hl_config_path)
        self.ll_config = self._load_config(ll_config_path)
        
        # Create environment
        print("Creating BiArt environment...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode=render_mode,
            joint_type='revolute'
        )
        
        # Load HL policy
        print(f"Loading {hl_policy_type} high-level policy...")
        self.hl_policy = self._load_hl_policy(hl_policy_type, hl_checkpoint_path)
        
        # Load LL policy (PPO)
        print("Loading low-level RL policy...")
        self.ll_policy = self._load_ll_policy(ll_checkpoint_path)
        
        # Create screw-decomposed impedance controller
        print("Creating screw-decomposed impedance controller...")
        self.controller = self._create_controller()
        
        # Observation buffers for HL policy
        self.obs_horizon = self.hl_config['model'].get('obs_horizon', 1)
        self.pred_horizon = self.hl_config['model'].get('pred_horizon', 10)
        self.img_buffer = []
        self.proprio_buffer = []
        
        # Action chunk buffer
        self.action_chunk: Optional[np.ndarray] = None
        self.action_chunk_idx: int = 0
        
        # Trajectory for LL policy
        self.trajectories = [None, None]
        self.hl_chunk_duration = 1.0  # Duration of HL chunk in seconds
        self.elapsed_time_in_chunk = 0.0
        
        # Screw axes (updated from environment)
        self.screw_axes = np.array([
            [1.0, 0.0, 0.0],  # Default revolute
            [1.0, 0.0, 0.0],
        ])
        
        # Current reference twists
        self.current_ref_twists = np.zeros((2, 3))
        
        # LL policy observation normalization
        self._setup_ll_normalization()
        
        # Display settings
        self.show_impedance_info = True
        self.font = None
        self.font_small = None
        
        # Colors for visualization
        self.chunk_colors = [
            (100, 149, 237),  # Cornflower blue (EE0 future)
            (205, 92, 92),    # Indian red (EE1 future)
        ]
        self.current_colors = [
            (65, 105, 225),   # Royal blue (EE0 current target)
            (220, 20, 60),    # Crimson (EE1 current target)
        ]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.isabs(config_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            config_path = os.path.join(project_root, config_path)
            
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_hl_policy(self, policy_type: str, checkpoint_path: str):
        """Load high-level policy with normalizer."""
        policy = create_policy(policy_type, self.hl_config, self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        policy.load_state_dict(checkpoint['model_state_dict'])
        
        self.hl_normalizer = LinearNormalizer()
        if 'normalizer' in checkpoint:
            self.hl_normalizer.load_state_dict(checkpoint['normalizer'])
            print("✓ Loaded HL normalizer from checkpoint")
            
        policy.eval()
        return policy
    
    def _load_ll_policy(self, checkpoint_path: str):
        """Load low-level PPO policy from SB3 checkpoint."""
        from stable_baselines3 import PPO
        
        # Load the PPO model
        policy = PPO.load(checkpoint_path, device=self.device)
        print(f"✓ Loaded LL policy from {checkpoint_path}")
        
        return policy
    
    def _create_controller(self) -> MultiGripperSE2ScrewDecomposedImpedanceController:
        """Create screw-decomposed impedance controller."""
        ll_cfg = self.ll_config.get('ll_controller', {})
        robot_cfg = ll_cfg.get('robot', {})
        screw_cfg = ll_cfg.get('screw_decomposed', {})
        
        robot_params = SE2RobotParams(
            mass=robot_cfg.get('mass', 1.2),
            inertia=robot_cfg.get('inertia', 97.6)
        )
        
        controller = MultiGripperSE2ScrewDecomposedImpedanceController(
            num_grippers=2,
            robot_params=robot_params,
            max_force=screw_cfg.get('max_force', 100.0),
            max_torque=screw_cfg.get('max_torque', 500.0)
        )
        
        return controller
    
    def _setup_ll_normalization(self):
        """Setup normalization constants for LL policy observations."""
        self.NORM_POS_SCALE = 512.0
        self.NORM_ANGLE_SCALE = np.pi
        self.NORM_WRENCH_SCALE = 100.0
        self.NORM_TWIST_LINEAR = 500.0
        self.NORM_TWIST_ANGULAR = 10.0
        
        # Action bounds from config
        screw_cfg = self.ll_config.get('ll_controller', {}).get('screw_decomposed', {})
        self.min_d_parallel = screw_cfg.get('min_d_parallel', 1.0)
        self.max_d_parallel = screw_cfg.get('max_d_parallel', 50.0)
        self.min_d_perp = screw_cfg.get('min_d_perp', 10.0)
        self.max_d_perp = screw_cfg.get('max_d_perp', 200.0)
        self.min_k_p = screw_cfg.get('min_k_p', 0.5)
        self.max_k_p = screw_cfg.get('max_k_p', 10.0)
        self.min_alpha = screw_cfg.get('min_alpha', 1.0)
        self.max_alpha = screw_cfg.get('max_alpha', 50.0)
    
    def _normalize_ll_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation for LL policy.
        
        Per SWIVL Paper Appendix B, observation structure:
        [ref_twists(6), screw_axes(6), wrenches(6), poses(6), twists(6)] = 30D
        """
        normalized = obs.copy()
        
        # Reference twists [omega, vx, vy] * 2 (indices 0-5)
        normalized[[0, 3]] /= self.NORM_TWIST_ANGULAR
        normalized[[1, 2, 4, 5]] /= self.NORM_TWIST_LINEAR
        
        # Screw axes [s_omega, s_x, s_y] * 2 (indices 6-11)
        # Already normalized, no scaling needed
        
        # Wrenches [tau, fx, fy] * 2 (indices 12-17)
        normalized[[12, 15]] /= self.NORM_WRENCH_SCALE * 0.5
        normalized[[13, 14, 16, 17]] /= self.NORM_WRENCH_SCALE
        
        # Poses [x, y, theta] * 2 (indices 18-23)
        normalized[[18, 19, 21, 22]] /= self.NORM_POS_SCALE
        normalized[[20, 23]] /= self.NORM_ANGLE_SCALE
        
        # Twists [omega, vx, vy] * 2 (indices 24-29)
        normalized[[24, 27]] /= self.NORM_TWIST_ANGULAR
        normalized[[25, 26, 28, 29]] /= self.NORM_TWIST_LINEAR
        
        return normalized
    
    def _scale_action(self, normalized_value: float, min_val: float, max_val: float) -> float:
        """Scale normalized action from [-1, 1] to [min_val, max_val]."""
        return min_val + (normalized_value + 1.0) * 0.5 * (max_val - min_val)
    
    def _decode_impedance_action(self, action: np.ndarray) -> Dict[str, float]:
        """
        Decode LL policy action to impedance parameters.
        
        Action: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7
        """
        return {
            'd_l_parallel': self._scale_action(action[0], self.min_d_parallel, self.max_d_parallel),
            'd_r_parallel': self._scale_action(action[1], self.min_d_parallel, self.max_d_parallel),
            'd_l_perp': self._scale_action(action[2], self.min_d_perp, self.max_d_perp),
            'd_r_perp': self._scale_action(action[3], self.min_d_perp, self.max_d_perp),
            'k_p_l': self._scale_action(action[4], self.min_k_p, self.max_k_p),
            'k_p_r': self._scale_action(action[5], self.min_k_p, self.max_k_p),
            'alpha': self._scale_action(action[6], self.min_alpha, self.max_alpha),
        }
    
    def get_hl_proprio(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract proprioception vector for HL policy."""
        ee_poses = obs['ee_poses'].flatten()
        ee_velocities = obs['ee_velocities'].flatten()
        wrenches = obs['external_wrenches'].flatten()
        return np.concatenate([ee_poses, ee_velocities, wrenches])
    
    def process_hl_observation(self, obs: Dict[str, np.ndarray], img: np.ndarray):
        """Process observation for HL policy input."""
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        proprio = self.get_hl_proprio(obs)
        
        self.img_buffer.append(img_tensor)
        self.proprio_buffer.append(proprio)
        
        if len(self.img_buffer) > self.obs_horizon:
            self.img_buffer.pop(0)
            self.proprio_buffer.pop(0)
        
        curr_imgs = list(self.img_buffer)
        curr_proprio = list(self.proprio_buffer)
        
        while len(curr_imgs) < self.obs_horizon:
            curr_imgs.insert(0, curr_imgs[0])
            curr_proprio.insert(0, curr_proprio[0])
        
        imgs_stack = torch.stack(curr_imgs)
        proprio_stack = np.stack(curr_proprio)
        
        proprio_norm = self.hl_normalizer.normalize(proprio_stack, 'proprio')
        proprio_tensor = torch.from_numpy(proprio_norm).float().to(self.device)
        
        imgs_batch = imgs_stack.unsqueeze(0).to(self.device)
        proprio_batch = proprio_tensor.unsqueeze(0)
        
        return imgs_batch, proprio_batch
    
    def get_new_action_chunk(self, obs: Dict[str, np.ndarray], img: np.ndarray) -> np.ndarray:
        """Run HL policy to get new action chunk."""
        imgs_tensor, proprio_tensor = self.process_hl_observation(obs, img)
        
        action_norm = self.hl_policy.inference(imgs_tensor, proprio_tensor)
        
        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.cpu().numpy()
        
        action = self.hl_normalizer.denormalize(action_norm, 'action')
        
        if action.ndim == 3:
            action = action[0]
        
        action_chunk = action.reshape(-1, 2, 3)
        action_chunk[:, :, :2] = np.clip(action_chunk[:, :, :2], 20.0, 492.0)
        
        return action_chunk
    
    def update_trajectories(self, obs: Dict[str, np.ndarray], action_chunk: np.ndarray):
        """Update trajectories from action chunk."""
        num_steps = action_chunk.shape[0]
        chunk_dt = self.hl_chunk_duration / num_steps
        times = np.linspace(0.0, self.hl_chunk_duration, num_steps + 1)
        
        for i in range(2):
            current_pose = obs['ee_poses'][i]
            waypoints = np.vstack([current_pose[np.newaxis, :], action_chunk[:, i, :]])
            
            self.trajectories[i] = CubicSplineTrajectory(
                waypoints=waypoints,
                times=times,
                boundary_conditions='natural'
            )
            self.trajectories[i].set_duration(self.hl_chunk_duration)
    
    def get_trajectory_targets(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get desired poses and twists from trajectories."""
        desired_poses = []
        desired_twists = []
        
        for i in range(2):
            if self.trajectories[i] is None:
                desired_poses.append(np.zeros(3))
                desired_twists.append(np.zeros(3))
            else:
                eval_t = np.clip(t, 0.0, self.hl_chunk_duration)
                traj_point = self.trajectories[i].evaluate(eval_t)
                desired_poses.append(traj_point.pose)
                desired_twists.append(traj_point.velocity_body)
        
        return np.array(desired_poses), np.array(desired_twists)
    
    def point_velocity_to_body_twist(self, pose: np.ndarray, point_velocity: np.ndarray) -> np.ndarray:
        """Convert world frame point velocity [vx, vy, omega] to body twist [omega, vx_b, vy_b]."""
        vx_world, vy_world, omega = point_velocity
        theta = pose[2]
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        vx_body = cos_theta * vx_world + sin_theta * vy_world
        vy_body = -sin_theta * vx_world + cos_theta * vy_world
        
        return np.array([omega, vx_body, vy_body])
    
    def compute_reference_twists(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray,
        desired_twists: np.ndarray,
        k_p_l: float = 3.0,
        k_p_r: float = 3.0
    ) -> np.ndarray:
        """
        Compute reference twists per SWIVL paper Eq. (6):
        V_ref = Ad_{T_bd} V_des + k_p * E
        """
        ref_twists = np.zeros((2, 3))
        k_p_values = [k_p_l, k_p_r]
        
        for i in range(2):
            theta_b = current_poses[i, 2]
            theta_d = desired_poses[i, 2]
            
            dx = desired_poses[i, 0] - current_poses[i, 0]
            dy = desired_poses[i, 1] - current_poses[i, 1]
            
            cos_b, sin_b = np.cos(theta_b), np.sin(theta_b)
            e_x = cos_b * dx + sin_b * dy
            e_y = -sin_b * dx + cos_b * dy
            
            # Angle error
            e_theta = theta_d - theta_b
            while e_theta > np.pi:
                e_theta -= 2 * np.pi
            while e_theta < -np.pi:
                e_theta += 2 * np.pi
            
            E = np.array([e_theta, e_x, e_y])
            ref_twists[i] = desired_twists[i] + k_p_values[i] * E
        
        return ref_twists
    
    def get_ll_observation(
        self,
        obs: Dict[str, np.ndarray],
        ref_twists: np.ndarray
    ) -> np.ndarray:
        """
        Construct LL policy observation.
        
        Per SWIVL Paper Appendix B, observation structure:
        [ref_twists(6), screw_axes(6), wrenches(6), poses(6), twists(6)] = 30D
        """
        # Get body twists
        current_twists = []
        for i in range(2):
            twist = self.point_velocity_to_body_twist(
                obs['ee_poses'][i],
                obs['ee_velocities'][i]
            )
            current_twists.append(twist)
        current_twists = np.array(current_twists)
        
        # Construct observation vector per SWIVL Paper Appendix B
        ll_obs = np.concatenate([
            ref_twists.flatten(),                 # 6: Reference twists
            self.screw_axes.flatten(),            # 6: Screw axes
            obs['external_wrenches'].flatten(),   # 6: Wrench feedback
            obs['ee_poses'].flatten(),            # 6: EE poses
            current_twists.flatten(),             # 6: EE body twists
        ])
        
        return ll_obs.astype(np.float32)
    
    def draw_action_chunk(self, screen: pygame.Surface, action_chunk: np.ndarray, current_idx: int):
        """Draw the entire action chunk trajectory with current target highlighted."""
        if action_chunk is None:
            return
            
        pred_horizon = action_chunk.shape[0]
        overlay = pygame.Surface((512, 512), pygame.SRCALPHA)
        
        for ee_idx in range(2):
            trajectory_points = []
            for t in range(current_idx, pred_horizon):
                x, y, _ = action_chunk[t, ee_idx]
                if 0 <= x <= 512 and 0 <= y <= 512:
                    trajectory_points.append((int(x), int(y)))
            
            if len(trajectory_points) > 1:
                pygame.draw.lines(overlay, (*self.chunk_colors[ee_idx], 150),
                                  False, trajectory_points, 2)
            
            for t in range(current_idx + 1, pred_horizon):
                x, y, theta = action_chunk[t, ee_idx]
                if not (0 <= x <= 512 and 0 <= y <= 512):
                    continue
                size = max(2, 6 - (t - current_idx))
                alpha = max(50, 180 - (t - current_idx) * 15)
                pygame.draw.circle(overlay, (*self.chunk_colors[ee_idx], alpha),
                                   (int(x), int(y)), size)
            
            if current_idx < pred_horizon:
                x, y, theta = action_chunk[current_idx, ee_idx]
                if 0 <= x <= 512 and 0 <= y <= 512:
                    self._draw_pose_arrow(overlay, x, y, theta,
                                          self.current_colors[ee_idx], size=25, alpha=180)
        
        screen.blit(overlay, (0, 0))
        self._draw_chunk_progress(screen, current_idx, pred_horizon)
    
    def _draw_pose_arrow(self, surface, x, y, theta, color, size=25, alpha=180):
        """Draw an arrow indicating pose orientation."""
        jaw_angle = theta + np.pi / 2
        cos_j, sin_j = np.cos(jaw_angle), np.sin(jaw_angle)
        
        arrow_length = size
        arrow_width = size * 0.32
        
        tip_x = x + arrow_length * cos_j
        tip_y = y + arrow_length * sin_j
        
        base_left_x = x - arrow_width * sin_j
        base_left_y = y + arrow_width * cos_j
        base_right_x = x + arrow_width * sin_j
        base_right_y = y - arrow_width * cos_j
        
        arrow_points = [
            (int(tip_x), int(tip_y)),
            (int(base_left_x), int(base_left_y)),
            (int(base_right_x), int(base_right_y))
        ]
        
        pygame.draw.polygon(surface, (*color, alpha), arrow_points)
        pygame.draw.polygon(surface, color, arrow_points, 2)
        pygame.draw.circle(surface, color, (int(x), int(y)), 5, 2)
    
    def _draw_chunk_progress(self, screen, current_idx: int, total: int):
        """Draw progress bar for action chunk execution."""
        bar_width = 150
        bar_height = 12
        bar_x = 10
        bar_y = 10
        
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        progress = (current_idx + 1) / total
        pygame.draw.rect(screen, (100, 200, 100),
                         (bar_x, bar_y, int(bar_width * progress), bar_height))
        
        pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), 1)
        
        if self.font_small is None:
            self.font_small = pygame.font.Font(None, 20)
        text = self.font_small.render(f"Chunk: {current_idx+1}/{total}", True, (255, 255, 255))
        screen.blit(text, (bar_x + bar_width + 10, bar_y - 2))
    
    def draw_impedance_info(self, screen: pygame.Surface, impedance_params: Dict[str, float]):
        """Draw impedance parameter information."""
        if not self.show_impedance_info:
            return
            
        if self.font_small is None:
            self.font_small = pygame.font.Font(None, 18)
        
        # Background
        panel_x, panel_y = 10, 35
        panel_width, panel_height = 200, 120
        
        overlay = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (30, 30, 30, 180), (0, 0, panel_width, panel_height))
        screen.blit(overlay, (panel_x, panel_y))
        
        # Title
        title = self.font_small.render("Impedance Parameters", True, (255, 255, 100))
        screen.blit(title, (panel_x + 5, panel_y + 5))
        
        # Parameters
        lines = [
            f"d_∥ L:{impedance_params['d_l_parallel']:.1f} R:{impedance_params['d_r_parallel']:.1f}",
            f"d_⊥ L:{impedance_params['d_l_perp']:.1f} R:{impedance_params['d_r_perp']:.1f}",
            f"k_p L:{impedance_params['k_p_l']:.2f} R:{impedance_params['k_p_r']:.2f}",
            f"α: {impedance_params['alpha']:.2f}",
        ]
        
        for i, line in enumerate(lines):
            text = self.font_small.render(line, True, (220, 220, 220))
            screen.blit(text, (panel_x + 5, panel_y + 25 + i * 20))
    
    def draw_wrench_arrow(self, surface: pygame.Surface, pose: np.ndarray, wrench: np.ndarray, scale: float = 1.0):
        """Draw an arrow representing the external wrench."""
        x, y, theta = pose
        tau, fx, fy = wrench
        
        force_mag = np.linalg.norm([fx, fy])
        if force_mag < 0.1:
            return
        
        start_pos = (int(x), int(y))
        end_x = x + fx * scale
        end_y = y + fy * scale
        end_pos = (int(end_x), int(end_y))
        
        color = (128, 0, 128)
        if force_mag > 20.0:
            color = (255, 0, 0)
        
        pygame.draw.line(surface, color, start_pos, end_pos, 3)
        
        angle = np.arctan2(fy, fx)
        head_size = 8
        arrow_p1 = (
            int(end_x - head_size * np.cos(angle - np.pi/6)),
            int(end_y - head_size * np.sin(angle - np.pi/6))
        )
        arrow_p2 = (
            int(end_x - head_size * np.cos(angle + np.pi/6)),
            int(end_y - head_size * np.sin(angle + np.pi/6))
        )
        
        pygame.draw.polygon(surface, color, [end_pos, arrow_p1, arrow_p2])
    
    def run_episode(self) -> EvalMetrics:
        """Run a single evaluation episode."""
        metrics = EvalMetrics()
        
        obs, _ = self.env.reset()
        self.controller.reset()
        self.img_buffer = []
        self.proprio_buffer = []
        self.action_chunk = None
        self.action_chunk_idx = 0
        self.trajectories = [None, None]
        self.elapsed_time_in_chunk = 0.0
        
        # Update screw axes from environment
        screw_axes = self.env.get_joint_axis_screws()
        if screw_axes is not None:
            B_left, B_right = screw_axes
            self.screw_axes = np.array([B_left, B_right])
            self.controller.set_screw_axes(self.screw_axes)
        
        episode_impedance = {
            'd_l_parallel': [], 'd_r_parallel': [],
            'd_l_perp': [], 'd_r_perp': [],
            'k_p_l': [], 'k_p_r': [],
            'alpha': []
        }
        
        done = False
        while not done:
            # 1. Check if we need a new action chunk from HL policy
            need_new_chunk = (
                self.action_chunk is None or
                self.action_chunk_idx >= min(self.execution_horizon, len(self.action_chunk))
            )
            
            if need_new_chunk:
                img = self.env._render_frame(visualize=False)
                self.action_chunk = self.get_new_action_chunk(obs, img)
                self.update_trajectories(obs, self.action_chunk)
                self.action_chunk_idx = 0
                self.elapsed_time_in_chunk = 0.0
            
            # 2. Get trajectory targets
            desired_poses, desired_twists = self.get_trajectory_targets(self.elapsed_time_in_chunk)
            
            # 3. Compute reference twists (per SWIVL paper Eq. 6)
            ref_twists = self.compute_reference_twists(
                obs['ee_poses'], desired_poses, desired_twists
            )
            self.current_ref_twists = ref_twists
            
            # 4. Get LL observation and run RL policy
            ll_obs = self.get_ll_observation(obs, ref_twists)
            ll_obs_norm = self._normalize_ll_obs(ll_obs)
            
            ll_action, _ = self.ll_policy.predict(ll_obs_norm, deterministic=True)
            impedance_params = self._decode_impedance_action(ll_action)
            
            # Track impedance parameters
            for key in episode_impedance:
                episode_impedance[key].append(impedance_params[key])
            
            # 5. Apply impedance parameters to controller
            self.controller.set_impedance_variables(
                d_l_parallel=impedance_params['d_l_parallel'],
                d_r_parallel=impedance_params['d_r_parallel'],
                d_l_perp=impedance_params['d_l_perp'],
                d_r_perp=impedance_params['d_r_perp'],
                k_p_l=impedance_params['k_p_l'],
                k_p_r=impedance_params['k_p_r'],
                alpha=impedance_params['alpha']
            )
            
            # 6. Compute control wrenches
            current_twists = []
            for i in range(2):
                twist = self.point_velocity_to_body_twist(obs['ee_poses'][i], obs['ee_velocities'][i])
                current_twists.append(twist)
            current_twists = np.array(current_twists)
            
            wrenches, control_info = self.controller.compute_wrenches(
                obs['ee_poses'],
                desired_poses,
                current_twists,
                desired_twists,
                obs['external_wrenches']
            )
            
            # 7. Step environment
            env_action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            
            # Update metrics
            metrics.episode_reward += reward
            metrics.episode_length += 1
            metrics.fighting_force += control_info.get('total_fighting_force', 0.0)
            
            # Tracking error
            for i in range(2):
                metrics.tracking_error += np.linalg.norm(obs['ee_poses'][i] - desired_poses[i])
            
            # Update timing
            self.elapsed_time_in_chunk += self.env.dt
            self.action_chunk_idx += 1
        
        # Compute averages
        if metrics.episode_length > 0:
            metrics.tracking_error /= metrics.episode_length
            metrics.fighting_force /= metrics.episode_length
            
            for key in episode_impedance:
                avg_key = 'avg_' + key.replace('d_l_parallel', 'd_parallel').replace('d_r_parallel', 'd_parallel')
                avg_key = avg_key.replace('d_l_perp', 'd_perp').replace('d_r_perp', 'd_perp')
                avg_key = avg_key.replace('k_p_l', 'k_p').replace('k_p_r', 'k_p')
                if 'avg_d_parallel' not in metrics.__dict__ or avg_key == 'avg_d_parallel':
                    setattr(metrics, avg_key, np.mean(episode_impedance[key]))
        
        return metrics
    
    def run_interactive(self):
        """Run interactive evaluation with visualization."""
        print("\n" + "=" * 60)
        print("SWIVL Hierarchical Policy Evaluation")
        print("=" * 60)
        print(f"HL Policy: {type(self.hl_policy).__name__}")
        print(f"Prediction Horizon: {self.pred_horizon}")
        print(f"Execution Horizon: {self.execution_horizon}")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  R     - Reset episode")
        print("  I     - Toggle impedance info")
        print("  ESC   - Exit")
        print("=" * 60 + "\n")
        
        obs, _ = self.env.reset()
        self.controller.reset()
        self.img_buffer = []
        self.proprio_buffer = []
        self.action_chunk = None
        self.action_chunk_idx = 0
        self.trajectories = [None, None]
        self.elapsed_time_in_chunk = 0.0
        
        # Update screw axes
        screw_axes = self.env.get_joint_axis_screws()
        if screw_axes is not None:
            B_left, B_right = screw_axes
            self.controller.set_screw_axes(np.array([B_left, B_right]))
        
        running = True
        paused = False
        step = 0
        episode = 0
        inference_count = 0
        current_impedance_params = None
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        self.controller.reset()
                        self.img_buffer = []
                        self.proprio_buffer = []
                        self.action_chunk = None
                        self.action_chunk_idx = 0
                        self.trajectories = [None, None]
                        self.elapsed_time_in_chunk = 0.0
                        step = 0
                        episode += 1
                        inference_count = 0
                        screw_axes = self.env.get_joint_axis_screws()
                        if screw_axes is not None:
                            B_left, B_right = screw_axes
                            self.controller.set_screw_axes(np.array([B_left, B_right]))
                        print(f"\n--- Episode {episode} Reset ---")
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
                    elif event.key == pygame.K_i:
                        self.show_impedance_info = not self.show_impedance_info
                        
            if paused:
                self.env.clock.tick(30)
                continue
            
            # 1. Check if we need new action chunk
            need_new_chunk = (
                self.action_chunk is None or
                self.action_chunk_idx >= min(self.execution_horizon, len(self.action_chunk))
            )
            
            if need_new_chunk:
                img = self.env._render_frame(visualize=False)
                self.action_chunk = self.get_new_action_chunk(obs, img)
                self.update_trajectories(obs, self.action_chunk)
                self.action_chunk_idx = 0
                self.elapsed_time_in_chunk = 0.0
                inference_count += 1
                
                print(f"\n[HL Inference #{inference_count}] Step {step}")
            
            # 2. Get trajectory targets
            desired_poses, desired_twists = self.get_trajectory_targets(self.elapsed_time_in_chunk)
            
            # 3. Compute reference twists (per SWIVL paper Eq. 6)
            ref_twists = self.compute_reference_twists(
                obs['ee_poses'], desired_poses, desired_twists
            )
            self.current_ref_twists = ref_twists
            
            # 4. Get LL observation and run RL policy
            ll_obs = self.get_ll_observation(obs, ref_twists)
            ll_obs_norm = self._normalize_ll_obs(ll_obs)
            
            ll_action, _ = self.ll_policy.predict(ll_obs_norm, deterministic=True)
            current_impedance_params = self._decode_impedance_action(ll_action)
            
            # 5. Apply impedance parameters
            self.controller.set_impedance_variables(**current_impedance_params)
            
            # 6. Compute wrenches
            current_twists = []
            for i in range(2):
                twist = self.point_velocity_to_body_twist(obs['ee_poses'][i], obs['ee_velocities'][i])
                current_twists.append(twist)
            current_twists = np.array(current_twists)
            
            wrenches, control_info = self.controller.compute_wrenches(
                obs['ee_poses'],
                desired_poses,
                current_twists,
                desired_twists,
                obs['external_wrenches']
            )
            
            # 7. Step environment
            env_action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            
            # Update timing
            self.elapsed_time_in_chunk += self.env.dt
            self.action_chunk_idx += 1
            step += 1
            
            # 8. Render
            if self.env.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()
                
                screen = self.env._draw()
                
                # Draw visualizations
                self.draw_action_chunk(screen, self.action_chunk,
                                       min(self.action_chunk_idx - 1, len(self.action_chunk) - 1))
                
                if current_impedance_params:
                    self.draw_impedance_info(screen, current_impedance_params)
                
                for i in range(2):
                    pose = obs['ee_poses'][i]
                    wrench = obs['external_wrenches'][i]
                    self.draw_wrench_arrow(screen, pose, wrench)
                
                self.env.window.blit(screen, screen.get_rect())
                pygame.display.flip()
                self.env.clock.tick(self.env.metadata['render_fps'])
            
            # 9. Check episode end
            if terminated or truncated:
                print(f"\n{'='*40}")
                print(f"Episode {episode} finished!")
                print(f"  Total steps: {step}")
                print(f"  HL Inference calls: {inference_count}")
                print(f"{'='*40}")
                
                obs, _ = self.env.reset()
                self.controller.reset()
                self.img_buffer = []
                self.proprio_buffer = []
                self.action_chunk = None
                self.action_chunk_idx = 0
                self.trajectories = [None, None]
                self.elapsed_time_in_chunk = 0.0
                step = 0
                episode += 1
                inference_count = 0
                
                screw_axes = self.env.get_joint_axis_screws()
                if screw_axes is not None:
                    B_left, B_right = screw_axes
                    self.controller.set_screw_axes(np.array([B_left, B_right]))
        
        self.env.close()
        pygame.quit()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Run batch evaluation without visualization.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with aggregated metrics
        """
        all_metrics = []
        
        print(f"\nEvaluating {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            metrics = self.run_episode()
            all_metrics.append(metrics)
            
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"reward={metrics.episode_reward:.2f}, "
                  f"len={metrics.episode_length}, "
                  f"tracking_err={metrics.tracking_error:.2f}, "
                  f"fighting_force={metrics.fighting_force:.2f}")
        
        # Aggregate
        aggregated = {
            'mean_reward': np.mean([m.episode_reward for m in all_metrics]),
            'std_reward': np.std([m.episode_reward for m in all_metrics]),
            'mean_length': np.mean([m.episode_length for m in all_metrics]),
            'mean_tracking_error': np.mean([m.tracking_error for m in all_metrics]),
            'mean_fighting_force': np.mean([m.fighting_force for m in all_metrics]),
            'mean_d_parallel': np.mean([m.avg_d_parallel for m in all_metrics]),
            'mean_d_perp': np.mean([m.avg_d_perp for m in all_metrics]),
            'mean_k_p': np.mean([m.avg_k_p for m in all_metrics]),
            'mean_alpha': np.mean([m.avg_alpha for m in all_metrics]),
        }
        
        return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SWIVL Hierarchical Policy (HL + LL + Controller)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--hl_policy', type=str, required=True,
        choices=['flow_matching', 'diffusion', 'act'],
        help='High-level policy type'
    )
    parser.add_argument(
        '--hl_checkpoint', type=str, required=True,
        help='Path to HL policy checkpoint'
    )
    parser.add_argument(
        '--ll_checkpoint', type=str, required=True,
        help='Path to LL (RL) policy checkpoint (.zip)'
    )
    parser.add_argument(
        '--hl_config', type=str, default='scripts/configs/hl_policy_config.yaml',
        help='Path to HL policy config'
    )
    parser.add_argument(
        '--ll_config', type=str, default='scripts/configs/rl_config.yaml',
        help='Path to LL config'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--execution_horizon', type=int, default=10,
        help='Steps to execute before re-planning'
    )
    parser.add_argument(
        '--num_episodes', type=int, default=None,
        help='Number of episodes for batch evaluation (omit for interactive)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save evaluation results'
    )
    
    args = parser.parse_args()
    
    evaluator = HierarchicalPolicyEvaluator(
        hl_policy_type=args.hl_policy,
        hl_checkpoint_path=args.hl_checkpoint,
        ll_checkpoint_path=args.ll_checkpoint,
        hl_config_path=args.hl_config,
        ll_config_path=args.ll_config,
        device=args.device,
        render_mode='human' if args.num_episodes is None else None,
        execution_horizon=args.execution_horizon
    )
    
    if args.num_episodes is not None:
        # Batch evaluation
        results = evaluator.evaluate(num_episodes=args.num_episodes)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_path}")
    else:
        # Interactive evaluation
        evaluator.run_interactive()


if __name__ == '__main__':
    main()

