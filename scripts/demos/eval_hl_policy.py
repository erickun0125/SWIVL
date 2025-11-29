import argparse
import numpy as np
import torch
import pygame
import cv2
import os
import sys
import yaml
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs.biart import BiArtEnv
from src.envs.object_manager import JointType
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains
from scripts.training.train_hl_policy import create_policy, LinearNormalizer

class HLPolicyEvaluator:
    """Evaluates trained High-Level Policies."""

    def __init__(
        self, 
        policy_type: str, 
        checkpoint_path: str,
        config_path: str = 'scripts/configs/hl_policy_config.yaml',
        no_wrench: bool = False,
        device: str = 'auto',
        render_mode: str = 'human',
        execution_horizon: int = 8
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
            joint_type='revolute' # Default to revolute for now, or make configurable
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
        self.img_buffer = []
        self.proprio_buffer = []
        
        # Display settings
        self.font = None
        self.font_small = None
        
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

    def draw_desired_frames(self, screen, desired_poses):
        """Draw desired pose frames."""
        if desired_poses is None: return
        
        colors = [(65, 105, 225), (220, 20, 60)] # Blue, Red
        
        for i, pose in enumerate(desired_poses):
            x, y, theta = pose
            if not (0 <= x <= 512 and 0 <= y <= 512): continue
            
            arrow_length = 25
            arrow_width = 8
            jaw_angle = theta + np.pi / 2
            cos_j, sin_j = np.cos(jaw_angle), np.sin(jaw_angle)
            
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
            
            arrow_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
            color_with_alpha = (*colors[i], 120)
            pygame.draw.polygon(arrow_surface, color_with_alpha, arrow_points)
            screen.blit(arrow_surface, (0, 0))
            pygame.draw.polygon(screen, colors[i], arrow_points, 2)
            pygame.draw.circle(screen, colors[i], (int(x), int(y)), 5, 2)

    def run(self):
        """Run evaluation loop."""
        print("\nStarting evaluation...")
        print("Controls: Space (Pause), R (Reset), ESC (Exit)")
        
        obs, _ = self.env.reset()
        self.controller.reset()
        self.img_buffer = []
        self.proprio_buffer = []
        
        running = True
        paused = False
        step = 0
        
        while running:
            # Events
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
                        step = 0
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        
            if paused:
                self.env.clock.tick(30)
                continue
                
            # 1. Get Observation
            # Capture image (96x96)
            img = self.env._render_frame(visualize=False)
            
            # Process for policy
            imgs_tensor, proprio_tensor = self.process_observation(obs, img)
            
            # 2. Inference
            with torch.no_grad():
                if hasattr(self.policy, '_denoise'): # Diffusion
                    action_norm = self.policy._denoise(imgs_tensor, proprio_tensor)
                elif hasattr(self.policy, '_sample_flow'): # Flow Matching
                    action_norm = self.policy._sample_flow(imgs_tensor, proprio_tensor)
                elif hasattr(self.policy, 'inference'): # ACT
                    action_norm = self.policy.inference(imgs_tensor, proprio_tensor)
                else:
                    # Fallback to get_action if possible, but it expects dict
                    # Construct dummy dict if needed or raise error
                    raise NotImplementedError(f"Policy {type(self.policy)} inference method not found")
                
            # Denormalize
            if isinstance(action_norm, torch.Tensor):
                action_norm = action_norm.cpu().numpy()
            action = self.normalizer.denormalize(action_norm, 'action')
            if action.ndim == 2:
                # Already (pred_horizon, 6)
                pass
            elif action.ndim == 3:
                # Remove batch dim: (1, pred_horizon, 6) -> (pred_horizon, 6)
                action = action[0]
            
            # 3. Execute Action Chunk (Receding Horizon)
            # We execute 'execution_horizon' steps from the predicted sequence
            # But since we run inference every step (or every N steps), let's just take the next few.
            # For smoothest closed-loop control, usually we execute just the first step (H=1)
            # or a few steps if inference is slow.
            
            # Let's execute just 1 step for now (Closed-Loop)
            # Or use execution_horizon if specified.
            
            # Current desired poses from action
            # Action is (T, 6) -> (T, 2, 3)
            desired_poses_seq = action.reshape(-1, 2, 3)
            
            # Execute first step
            target_pose = desired_poses_seq[0] # (2, 3)
            
            # Clip target pose
            target_pose[:, :2] = np.clip(target_pose[:, :2], 20.0, 492.0)
            
            # Debug: Print action statistics every 50 steps
            if step % 50 == 0:
                print(f"\n[Step {step}] Action range:")
                print(f"  Normalized action: min={action_norm.min():.3f}, max={action_norm.max():.3f}")
                print(f"  Denormalized action[0]: {action[0]}")
                print(f"  Target pose EE0: x={target_pose[0,0]:.1f}, y={target_pose[0,1]:.1f}, theta={target_pose[0,2]:.3f}")
                print(f"  Target pose EE1: x={target_pose[1,0]:.1f}, y={target_pose[1,1]:.1f}, theta={target_pose[1,2]:.3f}")
            
            # Compute PD control
            # We need desired velocity. We can estimate it from the sequence or set to 0.
            # Simple approach: set desired velocity to 0 (step input)
            # Better approach: (next_pose - curr_pose) / dt
            if desired_poses_seq.shape[0] > 1:
                next_pose = desired_poses_seq[1]
                desired_vel = (next_pose - target_pose) * self.env.control_hz
            else:
                desired_vel = np.zeros((2, 3))
                
            wrenches = self.controller.compute_wrenches(
                obs['ee_poses'],
                target_pose,
                desired_vel,
                current_velocities=obs['ee_velocities']
            )
            
            # Step Env
            env_action = np.concatenate([wrenches[0], wrenches[1]])
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            step += 1
            
            # Render
            if self.env.render_mode == 'human':
                if self.env.window is None:
                    self.env.window = pygame.display.set_mode((512, 512))
                if self.env.clock is None:
                    self.env.clock = pygame.time.Clock()
                    
                screen = self.env._draw()
                self.draw_desired_frames(screen, target_pose)
                self.env.window.blit(screen, screen.get_rect())
                pygame.display.flip()
                self.env.clock.tick(self.env.metadata['render_fps'])
                
            if terminated or truncated:
                print(f"Episode finished. Reward: {reward}")
                obs, _ = self.env.reset()
                self.controller.reset()
                self.img_buffer = []
                self.proprio_buffer = []
                step = 0
                
        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True, choices=['flow_matching', 'diffusion', 'act'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='scripts/configs/hl_policy_config.yaml')
    parser.add_argument('--no_wrench', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    evaluator = HLPolicyEvaluator(
        policy_type=args.policy,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        no_wrench=args.no_wrench,
        device=args.device
    )
    evaluator.run()
