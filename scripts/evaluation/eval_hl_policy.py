import argparse
import numpy as np
import torch
import pygame
import cv2
import os
import sys
import yaml
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains
from scripts.training.train_hl_policy import create_policy, LinearNormalizer

class HLPolicyEvaluator:
    """Evaluates trained High-Level Policies with Action Chunking."""

    def __init__(
        self, 
        policy_type: str, 
        checkpoint_path: str,
        config_path: str = 'scripts/configs/hl_policy_config.yaml',
        no_wrench: bool = False,
        device: str = 'auto',
        render_mode: str = 'human',
        execution_horizon: int = 10
    ):
        self.device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        
        self.execution_horizon = execution_horizon
        self.no_wrench = no_wrench
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Override config with CLI args
        if no_wrench:
            self.config['model']['use_external_wrench'] = False
            
        # Create environment
        print("Creating BiArt environment...")
        self.env = BiArtEnv(
            obs_type='state',
            render_mode=render_mode,
            joint_type='revolute'
        )
        
        # Create PD Controller
        gains = PDGains(
            kp_linear=1500.0,
            kd_linear=1000.0,
            kp_angular=750.0,
            kd_angular=500.0
        )
        self.controller = MultiGripperPDController(num_grippers=2, gains=gains)
        self.controller.set_timestep(self.env.dt)
        
        # Load Policy
        self.policy = create_policy(policy_type, self.config, self.device)
        self.load_checkpoint(checkpoint_path)
        self.policy.eval()
        
        # Observation buffers
        self.obs_horizon = self.config['model'].get('obs_horizon', 1)
        self.pred_horizon = self.config['model'].get('pred_horizon', 10)
        self.img_buffer = []
        self.proprio_buffer = []
        
        # Action chunk buffer
        self.action_chunk: Optional[np.ndarray] = None  # (pred_horizon, 2, 3)
        self.action_chunk_idx: int = 0
        
        # Display settings
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
        
    def load_checkpoint(self, path: str):
        """Load model weights and normalizer."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(checkpoint['normalizer'])
        print("Checkpoint loaded successfully.")

    def get_proprio(self, obs):
        """Extract proprioception vector from observation."""
        ee_poses = obs['ee_poses'].flatten()
        ee_velocities = obs['ee_velocities'].flatten() # Point velocities
        
        proprio_list = [ee_poses, ee_velocities]
        
        if not self.no_wrench:
            wrenches = obs['external_wrenches'].flatten()
            proprio_list.append(wrenches)
            
        return np.concatenate(proprio_list)

    def process_observation(self, obs, img):
        """Process observation for policy input."""
        # Image: (H, W, 3) -> (3, H, W) -> float32 -> normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Proprio
        proprio = self.get_proprio(obs)
        
        # Update buffers
        self.img_buffer.append(img_tensor)
        self.proprio_buffer.append(proprio)
        
        # Maintain horizon size
        if len(self.img_buffer) > self.obs_horizon:
            self.img_buffer.pop(0)
            self.proprio_buffer.pop(0)
            
        # Pad if buffer is not full (start of episode)
        curr_imgs = list(self.img_buffer)
        curr_proprio = list(self.proprio_buffer)
        
        while len(curr_imgs) < self.obs_horizon:
            curr_imgs.insert(0, curr_imgs[0])
            curr_proprio.insert(0, curr_proprio[0])
            
        # Stack
        imgs_stack = torch.stack(curr_imgs) # (T, 3, H, W)
        proprio_stack = np.stack(curr_proprio) # (T, D)
        
        # Normalize proprio
        proprio_norm = self.normalizer.normalize(proprio_stack, 'proprio')
        proprio_tensor = torch.from_numpy(proprio_norm).float().to(self.device)
        
        # Add batch dim
        imgs_batch = imgs_stack.unsqueeze(0).to(self.device) # (1, T, 3, H, W)
        proprio_batch = proprio_tensor.unsqueeze(0) # (1, T, D)
        
        return imgs_batch, proprio_batch

    def draw_action_chunk(self, screen, action_chunk: np.ndarray, current_idx: int):
        """
        Draw the entire action chunk trajectory with current target highlighted.
        
        Args:
            screen: Pygame surface
            action_chunk: (pred_horizon, 2, 3) array of poses
            current_idx: Current index being executed
        """
        if action_chunk is None:
            return
            
        pred_horizon = action_chunk.shape[0]
        
        # Create transparent overlay for trajectory
        overlay = pygame.Surface((512, 512), pygame.SRCALPHA)
        
        for ee_idx in range(2):
            # Draw trajectory line connecting all future poses
            trajectory_points = []
            for t in range(current_idx, pred_horizon):
                x, y, _ = action_chunk[t, ee_idx]
                if 0 <= x <= 512 and 0 <= y <= 512:
                    trajectory_points.append((int(x), int(y)))
            
            # Draw trajectory line
            if len(trajectory_points) > 1:
                pygame.draw.lines(overlay, (*self.chunk_colors[ee_idx], 150), 
                                  False, trajectory_points, 2)
            
            # Draw future waypoints (small circles)
            for t in range(current_idx + 1, pred_horizon):
                x, y, theta = action_chunk[t, ee_idx]
                if not (0 <= x <= 512 and 0 <= y <= 512):
                    continue
                    
                # Size decreases for further waypoints
                size = max(2, 6 - (t - current_idx))
                alpha = max(50, 180 - (t - current_idx) * 15)
                pygame.draw.circle(overlay, (*self.chunk_colors[ee_idx], alpha), 
                                   (int(x), int(y)), size)
            
            # Draw current target (larger, solid arrow)
            if current_idx < pred_horizon:
                x, y, theta = action_chunk[current_idx, ee_idx]
                if 0 <= x <= 512 and 0 <= y <= 512:
                    self._draw_pose_arrow(overlay, x, y, theta, 
                                          self.current_colors[ee_idx], size=25, alpha=180)
        
        screen.blit(overlay, (0, 0))
        
        # Draw chunk progress indicator
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
        
        # Background
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        # Progress
        progress = (current_idx + 1) / total
        pygame.draw.rect(screen, (100, 200, 100), 
                         (bar_x, bar_y, int(bar_width * progress), bar_height))
        
        # Border
        pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Text
        if self.font_small is None:
            self.font_small = pygame.font.Font(None, 20)
        text = self.font_small.render(f"Chunk: {current_idx+1}/{total}", True, (255, 255, 255))
        screen.blit(text, (bar_x + bar_width + 10, bar_y - 2))

    def get_new_action_chunk(self, obs, img) -> np.ndarray:
        """
        Run policy inference to get a new action chunk.
        
        Returns:
            action_chunk: (pred_horizon, 2, 3) array of desired poses
        """
        # Process observation
        imgs_tensor, proprio_tensor = self.process_observation(obs, img)
        
        # Inference
        with torch.no_grad():
            if hasattr(self.policy, '_denoise'):  # Diffusion
                action_norm = self.policy._denoise(imgs_tensor, proprio_tensor)
            elif hasattr(self.policy, '_sample_flow'):  # Flow Matching
                action_norm = self.policy._sample_flow(imgs_tensor, proprio_tensor)
            elif hasattr(self.policy, 'inference'):  # ACT
                action_norm = self.policy.inference(imgs_tensor, proprio_tensor)
            else:
                raise NotImplementedError(f"Policy {type(self.policy)} inference method not found")
        
        # Convert to numpy
        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.cpu().numpy()
        
        # Denormalize
        action = self.normalizer.denormalize(action_norm, 'action')
        
        # Handle batch dimension
        if action.ndim == 3:
            action = action[0]  # (1, pred_horizon, 6) -> (pred_horizon, 6)
        
        # Reshape to (pred_horizon, 2, 3)
        action_chunk = action.reshape(-1, 2, 3)
        
        # Clip positions to valid range
        action_chunk[:, :, :2] = np.clip(action_chunk[:, :, :2], 20.0, 492.0)
        
        return action_chunk

    def run(self):
        """Run evaluation loop with action chunking."""
        print("\n" + "="*60)
        print("Starting Action Chunk Evaluation")
        print("="*60)
        print(f"Prediction Horizon: {self.pred_horizon}")
        print(f"Execution Horizon: {self.execution_horizon}")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  R     - Reset episode")
        print("  ESC   - Exit")
        print("="*60 + "\n")
        
        obs, _ = self.env.reset()
        self.controller.reset()
        self.img_buffer = []
        self.proprio_buffer = []
        self.action_chunk = None
        self.action_chunk_idx = 0
        
        running = True
        paused = False
        step = 0
        episode = 0
        inference_count = 0
        
        while running:
            # Handle events
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
                        step = 0
                        episode += 1
                        inference_count = 0
                        print(f"\n--- Episode {episode} Reset ---")
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
                        
            if paused:
                self.env.clock.tick(30)
                continue
            
            # 1. Check if we need a new action chunk
            need_new_chunk = (
                self.action_chunk is None or 
                self.action_chunk_idx >= min(self.execution_horizon, len(self.action_chunk))
            )
            
            if need_new_chunk:
                # Get observation image
                img = self.env._render_frame(visualize=False)
                
                # Run inference
                self.action_chunk = self.get_new_action_chunk(obs, img)
                self.action_chunk_idx = 0
                inference_count += 1
                
                print(f"\n[Inference #{inference_count}] Step {step}")
                print(f"  New action chunk generated ({len(self.action_chunk)} steps)")
                print(f"  First pose EE0: x={self.action_chunk[0,0,0]:.1f}, y={self.action_chunk[0,0,1]:.1f}, θ={self.action_chunk[0,0,2]:.3f}")
                print(f"  First pose EE1: x={self.action_chunk[0,1,0]:.1f}, y={self.action_chunk[0,1,1]:.1f}, θ={self.action_chunk[0,1,2]:.3f}")
            
            # 2. Get current target from action chunk
            target_pose = self.action_chunk[self.action_chunk_idx]  # (2, 3)
            
            # 3. Compute desired velocity from chunk
            if self.action_chunk_idx + 1 < len(self.action_chunk):
                next_pose = self.action_chunk[self.action_chunk_idx + 1]
                desired_vel = (next_pose - target_pose) * self.env.control_hz
            else:
                desired_vel = np.zeros((2, 3))
            
            # 4. Compute PD control
            wrenches = self.controller.compute_wrenches(
                obs['ee_poses'],
                target_pose,
                desired_vel,
                current_velocities=obs['ee_velocities']
            )
            
            # 5. Step environment
            env_action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            
            # 6. Advance chunk index
            self.action_chunk_idx += 1
            step += 1
            
            # 7. Render
            if self.env.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()
                    
                screen = self.env._draw()
                
                # Draw action chunk visualization
                self.draw_action_chunk(screen, self.action_chunk, 
                                       min(self.action_chunk_idx - 1, len(self.action_chunk) - 1))
                
                self.env.window.blit(screen, screen.get_rect())
                pygame.display.flip()
                self.env.clock.tick(self.env.metadata['render_fps'])
            
            # 8. Check episode termination
            if terminated or truncated:
                print(f"\n{'='*40}")
                print(f"Episode {episode} finished!")
                print(f"  Total steps: {step}")
                print(f"  Inference calls: {inference_count}")
                print(f"  Reward: {reward:.4f}")
                print(f"{'='*40}")
                
                obs, _ = self.env.reset()
                self.controller.reset()
                self.img_buffer = []
                self.proprio_buffer = []
                self.action_chunk = None
                self.action_chunk_idx = 0
                step = 0
                episode += 1
                inference_count = 0
                
        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HL Policy with Action Chunking")
    parser.add_argument('--policy', type=str, required=True, choices=['flow_matching', 'diffusion', 'act'],
                        help='Policy type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/hl_policy_config.yaml',
                        help='Path to config file')
    parser.add_argument('--no_wrench', action='store_true',
                        help='Disable external wrench in observation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--execution_horizon', type=int, default=10,
                        help='Number of steps to execute before re-planning (default: 10, full chunk)')
    
    args = parser.parse_args()
    
    evaluator = HLPolicyEvaluator(
        policy_type=args.policy,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        no_wrench=args.no_wrench,
        device=args.device,
        execution_horizon=args.execution_horizon
    )
    evaluator.run()
