"""
Flow Matching Policy for Bimanual Manipulation

Implements conditional flow matching for generating manipulation trajectories.
The policy learns to generate smooth, feasible trajectories conditioned on:
- Initial state
- Goal specification
- Object configuration

The policy outputs desired poses for both end-effectors at 10 Hz.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching policy."""
    state_dim: int = 18  # 12 (poses+twists) + 6 (wrench, optional)
    action_dim: int = 6  # 2 EE desired poses (3 each)
    pred_horizon: int = 10 # Number of steps to predict (chunk size)
    obs_horizon: int = 1  # Number of past observations to condition on
    use_external_wrench: bool = True
    hidden_dim: int = 256
    num_layers: int = 4
    num_diffusion_steps: int = 10
    output_frequency: float = 10.0  # Hz
    context_length: int = 10  # Number of past states to condition on
    
    @property
    def proprio_dim(self):
        return 18 if self.use_external_wrench else 12


class ImageEncoder(nn.Module):
    """
    Simple CNN for image encoding.
    Input: (batch, seq, 3, 96, 96)
    Output: (batch, seq, hidden_dim)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 48
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 24
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 12
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 6
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 3, H, W)
        batch, seq, c, h, w = x.shape
        x = x.view(batch * seq, c, h, w)
        feat = self.net(x)
        return feat.view(batch, seq, -1)


class FlowMatchingNetwork(nn.Module):
    """
    Neural network for flow matching.

    Predicts velocity field for flow ODE:
        dx/dt = v_θ(x, t, context)
    """

    def __init__(self, config: FlowMatchingConfig):
        """
        Initialize flow matching network.

        Args:
            config: Flow matching configuration
        """
        super().__init__()
        self.config = config
        
        # Determine effective action dimension (flattened chunk)
        self.effective_action_dim = config.action_dim * config.pred_horizon

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim)
        )

        # Image encoder
        self.image_encoder = ImageEncoder(config.hidden_dim)
        
        # Proprio encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.proprio_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Fusion projection
        self.fusion_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Action/trajectory embedding
        # Input is flattened trajectory (batch, pred_horizon * action_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(self.effective_action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Main network
        layers = []
        for _ in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())

        self.main_net = nn.Sequential(*layers)

        # Output network
        # Output is velocity field for flattened trajectory
        self.output_net = nn.Linear(config.hidden_dim, self.effective_action_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        img_cond: torch.Tensor,
        prop_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Current trajectory sample (batch_size, pred_horizon, action_dim) or flattened
            t: Time parameter in [0, 1] (batch_size, 1)
            img_cond: Image condition (batch_size, 3, H, W) - using most recent frame
            prop_cond: Proprio condition (batch_size, proprio_dim) - using most recent

        Returns:
            Predicted velocity field (batch_size, pred_horizon * action_dim)
        """
        batch_size = x.shape[0]
        
        # Flatten input if needed
        if x.dim() > 2:
            x = x.reshape(batch_size, -1)
            
        # Embed time
        t_emb = self.time_mlp(t)

        # Embed images (add seq dim 1 for encoder)
        img_cond = img_cond.unsqueeze(1) # (batch, 1, 3, H, W)
        img_feat = self.image_encoder(img_cond).squeeze(1) # (batch, hidden)
        
        # Embed proprio
        prop_feat = self.proprio_encoder(prop_cond)
        
        # Fuse
        state_emb = self.fusion_proj(torch.cat([img_feat, prop_feat], dim=-1))

        # Embed current action
        action_emb = self.action_mlp(x)

        # Combine embeddings
        combined = state_emb + action_emb + t_emb.expand_as(state_emb)

        # Process through main network
        features = self.main_net(combined)

        # Output velocity
        velocity = self.output_net(features)

        return velocity


class FlowMatchingPolicy(nn.Module):
    """
    Flow matching policy for bimanual manipulation.

    Generates desired poses for both end-effectors using flow matching.
    """

    def __init__(
        self,
        config: Optional[FlowMatchingConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize flow matching policy.

        Args:
            config: Flow matching configuration
            device: Device for computation ('cpu' or 'cuda')
        """
        super().__init__()
        self.config = config if config is not None else FlowMatchingConfig()

        # Create network
        self.network = FlowMatchingNetwork(self.config)

        # State history for context
        self.image_history: List[np.ndarray] = []
        self.proprio_history: List[np.ndarray] = []
        self.normalizer = None

        self.to(torch.device(device))

    def set_normalizer(self, normalizer):
        """Set normalizer for input/output scaling."""
        self.normalizer = normalizer

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, img_cond: torch.Tensor, prop_cond: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.network(x_t, t, img_cond, prop_cond)

    def compute_loss(self, images: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            images: (batch, obs_horizon, 3, H, W)
            proprio: (batch, obs_horizon, proprio_dim)
            actions: (batch, pred_horizon, action_dim)
        """
        batch_size = images.shape[0]
        device = images.device

        # Use most recent observation as context
        img_cond = images[:, -1] # (batch, 3, H, W)
        prop_cond = proprio[:, -1] # (batch, proprio_dim)

        # Flatten actions: (batch, pred_horizon * action_dim)
        target_actions_flat = actions.reshape(batch_size, -1)

        t = torch.rand(batch_size, 1, device=device)
        noise = torch.randn_like(target_actions_flat)
        
        # Interpolate between noise and data
        x_t = (1 - t) * noise + t * target_actions_flat
        
        # Target velocity: (data - noise)
        target_velocity = target_actions_flat - noise

        # Predict velocity
        pred_velocity = self.network(x_t, t, img_cond, prop_cond)
        
        # Compute loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def reset(self):
        """Reset policy state."""
        self.image_history = []
        self.proprio_history = []

    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get action (desired poses) from current observation.

        Args:
            observation: Current observation dict
            goal: Optional goal specification

        Returns:
            Desired poses (2, 3) for both end-effectors
        """
        # Construct state vector
        image, proprio = self._construct_state(observation)

        # Normalize state if normalizer is present
        if self.normalizer is not None:
            image = image.astype(np.float32) / 255.0
            proprio = self.normalizer.normalize(proprio, 'proprio')

        # Add to history
        self.image_history.append(image)
        self.proprio_history.append(proprio)
        
        if len(self.image_history) > self.config.context_length:
            self.image_history.pop(0)
            self.proprio_history.pop(0)

        # Pad history if needed
        img_cond, prop_cond = self._get_context()

        # Sample from flow
        with torch.no_grad():
            action = self._sample_flow(img_cond, prop_cond, goal)

        # Denormalize action
        if self.normalizer is not None:
            action = self.normalizer.denormalize(action, 'action')

        # Convert to desired poses
        desired_poses = action.reshape(2, 3)

        return desired_poses

    def get_action_chunk(
        self,
        observation: Dict[str, np.ndarray],
        goal: Optional[np.ndarray] = None,
        chunk_size: int = 10
    ) -> np.ndarray:
        """
        Get action chunk. 
        Since Flow Matching implementation here is typically 1-step,
        we implement an autoregressive roll-out or simple repetition if needed.
        
        Ideally, the model should be trained to predict a horizon.
        Assuming the trained model predicts 1 step:
        We will simulate autoregressive prediction by feeding back the prediction.
        
        Args:
            observation: Current observation dict
            goal: Optional goal
            chunk_size: Number of steps to predict (default 10 for 1.0s duration)

        Returns:
            Action chunk (chunk_size, 2, 3)
        """
        if self.config.pred_horizon > 1:
            # Model predicts full chunk at once
            # Construct state vector from observation
            image, proprio = self._construct_state(observation)
            
            # Normalize state if normalizer is present
            if self.normalizer is not None:
                image = image.astype(np.float32) / 255.0
                proprio = self.normalizer.normalize(proprio, 'proprio')
            
            self.image_history.append(image)
            self.proprio_history.append(proprio)
            
            if len(self.image_history) > self.config.context_length:
                self.image_history.pop(0)
                self.proprio_history.pop(0)
                
            img_cond, prop_cond = self._get_context()
            
            with torch.no_grad():
                action = self._sample_flow(img_cond, prop_cond, goal)
                
            if self.normalizer is not None:
                action = self.normalizer.denormalize(action, 'action')
                
            # Reshape based on chunk size
            # Expected: (batch=1, pred_horizon * action_dim) -> (pred_horizon, 2, 3)
            return action.reshape(self.config.pred_horizon, 2, 3)
        else:
            # Autoregressive generation (approximate)
            # Warning: This assumes the model can take its own output as next state input
            # which isn't strictly true because state includes wrenches/velocities.
            # For this specific request "HL Policy outputs 10 timesteps",
            # we assume the underlying policy is CAPABLE/TRAINED to do so.
            # If not, we perform a naive 1-step prediction and repeat (hold) or linear extrapolation.
            # Better approach: Just use 1-step prediction 10 times? No, state won't change.
            # We will return 10 copies of the same action if horizon is 1.
            
            # Standard get_action
            action_step = self.get_action(observation, goal) # (2, 3)
            
            # Repeat for chunk size
            return np.tile(action_step[np.newaxis, :, :], (chunk_size, 1, 1))

    def _construct_state(self, observation: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Construct state vector from observation."""
        # Image
        image = observation['image'] # (H, W, 3)
        image = np.transpose(image, (2, 0, 1)) # (3, H, W)
        
        # Proprio
        ee_poses = observation['ee_poses'].flatten()
        ee_velocities = observation['ee_velocities'].flatten()  # point velocities [vx, vy, omega]
        
        proprio_list = [ee_poses, ee_velocities]
        
        if self.config.use_external_wrench:
            wrenches = observation['external_wrenches'].flatten()
            proprio_list.append(wrenches)

        proprio = np.concatenate(proprio_list)
        return image, proprio

    def _get_context(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get context from state history."""
        if len(self.image_history) == 0:
            img_ctx = np.zeros((3, 96, 96), dtype=np.float32)
            prop_ctx = np.zeros(self.config.proprio_dim, dtype=np.float32)
        else:
            # Use most recent state as context
            img_ctx = self.image_history[-1]
            prop_ctx = self.proprio_history[-1]

        img_tensor = torch.FloatTensor(img_ctx).unsqueeze(0).to(self._device())
        prop_tensor = torch.FloatTensor(prop_ctx).unsqueeze(0).to(self._device())
        
        return img_tensor, prop_tensor

    def _sample_flow(
        self,
        img_cond: torch.Tensor,
        prop_cond: torch.Tensor,
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Sample from flow model using ODE integration.

        Args:
            img_cond: Image condition
            prop_cond: Proprio condition
            goal: Optional goal

        Returns:
            Sampled action chunk (flattened)
        """
        # Start from noise (or goal if provided)
        device = self._device()
        effective_dim = self.config.action_dim * self.config.pred_horizon
        
        if goal is not None:
            # Not fully implemented for chunk goal yet, fallback to noise
            x = torch.randn(1, effective_dim, device=device)
        else:
            x = torch.randn(1, effective_dim, device=device)

        # Integrate flow ODE using Euler method
        dt = 1.0 / self.config.num_diffusion_steps

        for step in range(self.config.num_diffusion_steps):
            t = torch.full((1, 1), step * dt, dtype=torch.float32, device=device)

            # Predict velocity
            velocity = self.network(x, t, img_cond, prop_cond)

            # Euler step
            x = x + velocity * dt

        return x.cpu().numpy()

    def train_step(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        """
        Single training step.

        Args:
            images: Batch of images (batch_size, obs_horizon, 3, H, W)
            proprio: Batch of proprio (batch_size, obs_horizon, proprio_dim)
            actions: Batch of actions (batch_size, pred_horizon, action_dim)

        Returns:
            Loss value
        """
        batch_size = images.shape[0]
        device = images.device

        # Use most recent observation as context
        img_cond = images[:, -1]  # (batch, 3, H, W)
        prop_cond = proprio[:, -1]  # (batch, proprio_dim)

        # Flatten actions
        actions_flat = actions.reshape(batch_size, -1)

        # Sample time
        t = torch.rand(batch_size, 1, device=device)

        # Sample noise
        noise = torch.randn_like(actions_flat)

        # Interpolate between noise and data
        x_t = (1 - t) * noise + t * actions_flat

        # Target velocity: (data - noise)
        target_velocity = actions_flat - noise

        # Predict velocity
        pred_velocity = self.network(x_t, t, img_cond, prop_cond)

        # Compute loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)

        return loss.item()

    def save(self, path: str):
        """Save policy to file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=self._device())
        self.config = checkpoint['config']
        self.network = FlowMatchingNetwork(self.config).to(self._device())
        self.network.load_state_dict(checkpoint['network_state_dict'])
