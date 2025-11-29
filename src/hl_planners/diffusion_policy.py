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
import torchvision.models as models


@dataclass
class DiffusionPolicyConfig:
    """Configuration for diffusion policy."""
    state_dim: int = 18  # 12 (poses+twists) + 6 (wrench, optional)
    action_dim: int = 6  # 2 EE desired poses (3 each)
    pred_horizon: int = 10  # Number of actions to predict
    obs_horizon: int = 1  # Number of past observations to condition on
    use_external_wrench: bool = True
    hidden_dim: int = 256
    num_layers: int = 3
    num_diffusion_steps: int = 100  # Total diffusion steps
    num_inference_steps: int = 10  # Inference denoising steps
    output_frequency: float = 10.0  # Hz
    beta_schedule: str = 'squaredcos_cap_v2'  # Noise schedule
    
    @property
    def proprio_dim(self):
        return 18 if self.use_external_wrench else 12


class ImageEncoder(nn.Module):
    """
    ResNet18-based image encoder with ImageNet pretrained weights.
    Input: (batch, seq, 3, 96, 96)
    Output: (batch, seq, hidden_dim)
    """
    def __init__(self, hidden_dim: int, freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer
        # ResNet18 outputs 512-dim features before fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Project to hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        
        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 3, H, W), assumed to be in [0, 1] range
        batch, seq, c, h, w = x.shape
        x = x.view(batch * seq, c, h, w)
        
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        
        # Extract features
        feat = self.backbone(x)  # (batch*seq, 512, h', w')
        feat = self.avgpool(feat)  # (batch*seq, 512, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (batch*seq, 512)
        feat = self.fc(feat)  # (batch*seq, hidden_dim)
        
        return feat.view(batch, seq, -1)


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

        # Image encoder
        self.image_encoder = ImageEncoder(config.hidden_dim)
        
        # Proprio encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.proprio_dim * config.obs_horizon, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Fusion projection
        self.fusion_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim * config.pred_horizon, config.hidden_dim),
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
        self.output_net = nn.Linear(config.hidden_dim, config.action_dim * config.pred_horizon)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in noisy actions.

        Args:
            noisy_actions: (batch_size, pred_horizon, action_dim) noisy action sequence
            timestep: (batch_size,) diffusion timesteps
            images: (batch_size, obs_horizon, 3, H, W)
            proprio: (batch_size, obs_horizon, proprio_dim)

        Returns:
            Predicted noise (batch_size, pred_horizon, action_dim)
        """
        batch_size = noisy_actions.shape[0]

        # Flatten inputs
        noisy_actions_flat = noisy_actions.reshape(batch_size, -1)
        proprio_flat = proprio.reshape(batch_size, -1)

        # Encode timestep
        time_emb = self.time_mlp(timestep)

        # Encode images
        img_feat = self.image_encoder(images) # (batch, obs_horizon, hidden)
        img_emb = img_feat.mean(dim=1) # Average pooling over time for simplicity, or flatten
        
        # Encode proprio
        proprio_emb = self.proprio_encoder(proprio_flat)
        
        # Fuse observation
        obs_emb = self.fusion_proj(torch.cat([img_emb, proprio_emb], dim=-1))

        # Encode noisy actions
        action_emb = self.action_encoder(noisy_actions_flat)

        # Combine embeddings
        combined = obs_emb + action_emb + time_emb

        # Process through main network
        features = self.main_net(combined)

        # Output noise prediction
        noise_pred = self.output_net(features)
        noise_pred = noise_pred.reshape(batch_size, self.config.pred_horizon, self.config.action_dim)

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
        self.image_history: List[np.ndarray] = []
        self.proprio_history: List[np.ndarray] = []

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
        self.image_history = []
        self.proprio_history = []
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
        image, proprio = self._construct_state(observation)

        # Normalize state if normalizer is present
        if self.normalizer is not None:
            image = image.astype(np.float32) / 255.0
            proprio = self.normalizer.normalize(proprio, 'proprio')

        # Add to history
        self.image_history.append(image)
        self.proprio_history.append(proprio)
        
        if len(self.image_history) > self.config.obs_horizon:
            self.image_history.pop(0)
            self.proprio_history.pop(0)

        # Check if we can reuse buffered actions
        if self.action_buffer is not None and self.action_idx < self.config.pred_horizon:
            action = self.action_buffer[self.action_idx]
            self.action_idx += 1
        else:
            # Generate new action sequence via diffusion
            img_cond, prop_cond = self._get_obs_condition()

            with torch.no_grad():
                action_sequence = self._denoise(img_cond, prop_cond, goal)

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
        image, proprio = self._construct_state(observation)

        # Normalize state if normalizer is present
        if self.normalizer is not None:
            image = image.astype(np.float32) / 255.0
            proprio = self.normalizer.normalize(proprio, 'proprio')

        # Add to history
        self.image_history.append(image)
        self.proprio_history.append(proprio)
        
        if len(self.image_history) > self.config.obs_horizon:
            self.image_history.pop(0)
            self.proprio_history.pop(0)

        # Generate new action sequence via diffusion
        img_cond, prop_cond = self._get_obs_condition()

        with torch.no_grad():
            action_sequence = self._denoise(img_cond, prop_cond, goal)

        # Denormalize actions if normalizer is present
        if self.normalizer is not None:
            action_sequence = self.normalizer.denormalize(action_sequence, 'action')

        # Reshape to (pred_horizon, 2, 3)
        horizon = action_sequence.shape[0]
        return action_sequence.reshape(horizon, 2, 3)

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

    def _get_obs_condition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get observation condition from history."""
        # Pad history if needed
        while len(self.image_history) < self.config.obs_horizon:
            if len(self.image_history) == 0:
                # Initialize with zeros if empty
                self.image_history.append(np.zeros((3, 96, 96), dtype=np.float32))
                self.proprio_history.append(np.zeros(self.config.proprio_dim, dtype=np.float32))
            else:
                self.image_history.insert(0, self.image_history[0])
                self.proprio_history.insert(0, self.proprio_history[0])

        # Stack observations
        img_array = np.stack(self.image_history[-self.config.obs_horizon:], axis=0)
        prop_array = np.stack(self.proprio_history[-self.config.obs_horizon:], axis=0)
        
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).to(self._device())
        prop_tensor = torch.FloatTensor(prop_array).unsqueeze(0).to(self._device())

        return img_tensor, prop_tensor

    def _denoise(
        self,
        img_cond: torch.Tensor,
        prop_cond: torch.Tensor,
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Denoise to generate action sequence.

        Args:
            img_cond: Image condition (1, obs_horizon, 3, H, W)
            prop_cond: Proprio condition (1, obs_horizon, proprio_dim)
            goal: Optional goal

        Returns:
            Action sequence (pred_horizon, action_dim)
        """
        # Start from random noise
        if goal is not None:
            # Can incorporate goal as initialization
            noisy_action = torch.as_tensor(goal, dtype=torch.float32, device=self._device()).unsqueeze(0)
            if noisy_action.shape[1] < self.config.pred_horizon:
                # Repeat goal for action horizon
                noisy_action = noisy_action.repeat(1, self.config.pred_horizon, 1)
        else:
            noisy_action = torch.randn(
                1,
                self.config.pred_horizon,
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
            predicted_noise = self.noise_pred_net(noisy_action, t_batch, img_cond, prop_cond)

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
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        """
        batch_size = actions.shape[0]
        device = images.device

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

        predicted_noise = self.noise_pred_net(noisy_actions, timesteps, images, proprio)
        return F.mse_loss(predicted_noise, noise)

    def save(self, path: str):
        """Save policy to file."""
        torch.save({
            'noise_pred_net_state_dict': self.noise_pred_net.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load policy from file."""
        device = self._device()
        checkpoint = torch.load(path, map_location=device)
        self.config = checkpoint['config']
        self.noise_pred_net = DiffusionNoisePredictor(self.config).to(device)
        self.noise_pred_net.load_state_dict(checkpoint['noise_pred_net_state_dict'])

        # Reinitialize noise schedule as buffers
        betas = self._get_noise_schedule()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
