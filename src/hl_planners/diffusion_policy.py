"""
Diffusion Policy for Bimanual Manipulation

Implements diffusion-based imitation learning for generating manipulation trajectories.
Based on "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al., 2023)

The policy uses denoising diffusion to generate action sequences conditioned on:
- Current observation
- Past observations (observation history)
- Goal specification (optional)

The policy outputs desired poses for both end-effectors at 10 Hz.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiffusionPolicyConfig:
    """Configuration for diffusion policy."""
    state_dim: int = 24  # includes ee poses, link poses, external wrenches, body twists
    action_dim: int = 6  # 2 EE desired poses (3 each)
    action_horizon: int = 8  # Number of actions to predict
    obs_horizon: int = 2  # Number of past observations to condition on
    hidden_dim: int = 256
    num_layers: int = 3
    num_diffusion_steps: int = 100  # Total diffusion steps
    num_inference_steps: int = 10  # Inference denoising steps
    output_frequency: float = 10.0  # Hz
    beta_schedule: str = 'squaredcos_cap_v2'  # Noise schedule


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) timesteps
        Returns:
            (batch_size, dim) embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionNoisePredictor(nn.Module):
    """
    Noise prediction network for diffusion policy.

    Predicts noise to be removed at each diffusion timestep,
    conditioned on observation history.
    """

    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.state_dim * config.obs_horizon, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim * config.action_horizon, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Main network
        layers = []
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Mish()
            ])
        self.main_net = nn.Sequential(*layers)

        # Output network
        self.output_net = nn.Linear(config.hidden_dim, config.action_dim * config.action_horizon)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in noisy actions.

        Args:
            noisy_actions: (batch_size, action_horizon, action_dim) noisy action sequence
            timestep: (batch_size,) diffusion timesteps
            obs_cond: (batch_size, obs_horizon, state_dim) observation history

        Returns:
            Predicted noise (batch_size, action_horizon, action_dim)
        """
        batch_size = noisy_actions.shape[0]

        # Flatten inputs
        noisy_actions_flat = noisy_actions.reshape(batch_size, -1)
        obs_cond_flat = obs_cond.reshape(batch_size, -1)

        # Encode timestep
        time_emb = self.time_mlp(timestep)

        # Encode observation
        obs_emb = self.obs_encoder(obs_cond_flat)

        # Encode noisy actions
        action_emb = self.action_encoder(noisy_actions_flat)

        # Combine embeddings
        combined = obs_emb + action_emb + time_emb

        # Process through main network
        features = self.main_net(combined)

        # Output noise prediction
        noise_pred = self.output_net(features)
        noise_pred = noise_pred.reshape(batch_size, self.config.action_horizon, self.config.action_dim)

        return noise_pred


class DiffusionPolicy(nn.Module):
    """
    Diffusion policy for bimanual manipulation.

    Generates action sequences (desired poses) using denoising diffusion.
    """

    def __init__(
        self,
        config: Optional[DiffusionPolicyConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize diffusion policy.

        Args:
            config: Diffusion policy configuration
            device: Device for computation ('cpu' or 'cuda')
        """
        super().__init__()
        self.config = config if config is not None else DiffusionPolicyConfig()

        # Create noise prediction network
        self.noise_pred_net = DiffusionNoisePredictor(self.config)

        # Initialize noise schedule as buffers
        betas = self._get_noise_schedule()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Observation history
        self.obs_history: List[np.ndarray] = []

        # Action buffer for temporal consistency
        self.action_buffer: Optional[np.ndarray] = None
        self.action_idx = 0
        self.normalizer = None

        self.to(torch.device(device))

    def set_normalizer(self, normalizer):
        """Set normalizer for input/output scaling."""
        self.normalizer = normalizer

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_noise_schedule(self) -> torch.Tensor:
        """Get noise schedule (beta values)."""
        if self.config.beta_schedule == 'linear':
            return torch.linspace(1e-4, 0.02, self.config.num_diffusion_steps)
        elif self.config.beta_schedule == 'squaredcos_cap_v2':
            # Improved cosine schedule
            steps = self.config.num_diffusion_steps
            s = 0.008
            timesteps = torch.arange(steps + 1) / steps
            alphas_cumprod = torch.cos((timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")

    def reset(self):
        """Reset policy state."""
        self.obs_history = []
        self.action_buffer = None
        self.action_idx = 0

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
        state = self._construct_state(observation)

        # Normalize state if normalizer is present
        if self.normalizer is not None:
            state = self.normalizer.normalize(state, 'state')

        # Add to history
        self.obs_history.append(state)
        if len(self.obs_history) > self.config.obs_horizon:
            self.obs_history.pop(0)

        # Check if we can reuse buffered actions
        if self.action_buffer is not None and self.action_idx < self.config.action_horizon:
            action = self.action_buffer[self.action_idx]
            self.action_idx += 1
        else:
            # Generate new action sequence via diffusion
            obs_cond = self._get_obs_condition()

            with torch.no_grad():
                action_sequence = self._denoise(obs_cond, goal)

            # Buffer actions
            if self.normalizer is not None:
                action_sequence = self.normalizer.denormalize(action_sequence, 'action')

            self.action_buffer = action_sequence
            self.action_idx = 1  # Use first action now
            action = action_sequence[0]

        # Convert to desired poses
        desired_poses = action.reshape(2, 3)

        return desired_poses

    def get_action_chunk(
        self,
        observation: Dict[str, np.ndarray],
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get full action chunk (sequence of desired poses) from current observation.

        Args:
            observation: Current observation dict
            goal: Optional goal specification

        Returns:
            Action chunk (action_horizon, 2, 3)
        """
        # Construct state vector
        state = self._construct_state(observation)

        # Normalize state if normalizer is present
        if self.normalizer is not None:
            state = self.normalizer.normalize(state, 'state')

        # Add to history
        self.obs_history.append(state)
        if len(self.obs_history) > self.config.obs_horizon:
            self.obs_history.pop(0)

        # Generate new action sequence via diffusion
        obs_cond = self._get_obs_condition()

        with torch.no_grad():
            action_sequence = self._denoise(obs_cond, goal)

        # Denormalize actions if normalizer is present
        if self.normalizer is not None:
            action_sequence = self.normalizer.denormalize(action_sequence, 'action')

        # Reshape to (action_horizon, 2, 3)
        horizon = action_sequence.shape[0]
        return action_sequence.reshape(horizon, 2, 3)

    def _construct_state(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct state vector from observation."""
        ee_poses = observation['ee_poses'].flatten()
        link_poses = observation['link_poses'].flatten()
        wrenches = observation['external_wrenches'].flatten()

        state = np.concatenate([ee_poses, link_poses, wrenches])
        return state

    def _get_obs_condition(self) -> torch.Tensor:
        """Get observation condition from history."""
        # Pad history if needed
        while len(self.obs_history) < self.config.obs_horizon:
            if len(self.obs_history) == 0:
                self.obs_history.append(np.zeros(self.config.state_dim))
            else:
                self.obs_history.insert(0, self.obs_history[0])

        # Stack observations
        obs_array = np.stack(self.obs_history[-self.config.obs_horizon:], axis=0)
        obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self._device())

        return obs_tensor

    def _denoise(
        self,
        obs_cond: torch.Tensor,
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Denoise to generate action sequence.

        Args:
            obs_cond: Observation condition (1, obs_horizon, state_dim)
            goal: Optional goal

        Returns:
            Action sequence (action_horizon, action_dim)
        """
        # Start from random noise
        if goal is not None:
            # Can incorporate goal as initialization
            noisy_action = torch.as_tensor(goal, dtype=torch.float32, device=self._device()).unsqueeze(0)
            if noisy_action.shape[1] < self.config.action_horizon:
                # Repeat goal for action horizon
                noisy_action = noisy_action.repeat(1, self.config.action_horizon, 1)
        else:
            noisy_action = torch.randn(
                1,
                self.config.action_horizon,
                self.config.action_dim,
                device=self._device()
            )

        # DDIM sampling for faster inference
        timesteps = torch.linspace(
            self.config.num_diffusion_steps - 1,
            0,
            self.config.num_inference_steps,
            device=self._device()
        ).long()

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(noisy_action.shape[0])

            # Predict noise
            predicted_noise = self.noise_pred_net(noisy_action, t_batch, obs_cond)

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self._device())

            # Predict x0
            pred_x0 = (noisy_action - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise

            # Compute x_{t-1}
            noisy_action = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return noisy_action.squeeze(0).cpu().numpy()

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        """
        batch_size = actions.shape[0]
        device = states.device

        timesteps = torch.randint(
            0,
            self.config.num_diffusion_steps,
            (batch_size,),
            device=device
        ).long()

        noise = torch.randn_like(actions)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])[:, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])[:, None, None]

        noisy_actions = sqrt_alphas_cumprod * actions + sqrt_one_minus_alphas_cumprod * noise

        predicted_noise = self.noise_pred_net(noisy_actions, timesteps, states)
        return F.mse_loss(predicted_noise, noise)

    def save(self, path: str):
        """Save policy to file."""
        torch.save({
            'noise_pred_net_state_dict': self.noise_pred_net.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.noise_pred_net = DiffusionNoisePredictor(self.config).to(self.device)
        self.noise_pred_net.load_state_dict(checkpoint['noise_pred_net_state_dict'])

        # Reinitialize noise schedule
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=self.device),
            self.alphas_cumprod[:-1]
        ])
