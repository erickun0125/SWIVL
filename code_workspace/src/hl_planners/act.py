"""
ACT (Action Chunking with Transformers) for Bimanual Manipulation

Implements Action Chunking with Transformers for imitation learning.
Based on "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)

ACT combines:
- CVAE (Conditional Variational Autoencoder) for action distribution learning
- Transformer encoder-decoder for temporal modeling
- Action chunking for temporal consistency

The policy outputs action chunks (sequences of desired poses) for both end-effectors.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from src.hl_planners.image_encoder import ImageEncoder


@dataclass
class ACTConfig:
    """Configuration for ACT policy."""
    state_dim: int = 18  # 12 (poses+twists) + 6 (wrench, optional)
    action_dim: int = 6  # 2 EE desired poses (3 each)
    pred_horizon: int = 10  # Number of actions in a chunk
    obs_horizon: int = 1  # Number of past observations to condition on
    use_external_wrench: bool = True
    hidden_dim: int = 256
    latent_dim: int = 32  # Latent dimension for CVAE
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    num_heads: int = 8
    dropout: float = 0.1
    output_frequency: float = 10.0  # Hz
    kl_weight: float = 10.0  # Weight for KL divergence loss

    @property
    def proprio_dim(self):
        return 18 if self.use_external_wrench else 12


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return x


class ACTEncoder(nn.Module):
    """
    Transformer encoder for ACT.

    Encodes observation history into latent representation.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # Image encoder
        self.image_encoder = ImageEncoder(config.hidden_dim)
        
        # Proprio projection
        self.proprio_proj = nn.Linear(config.proprio_dim, config.hidden_dim)
        
        # Fusion (Image + Proprio)
        self.fusion_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Encode observation sequence.

        Args:
            images: (batch_size, obs_horizon, 3, H, W)
            proprio: (batch_size, obs_horizon, proprio_dim)

        Returns:
            Encoded features (obs_horizon, batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = proprio.shape

        # Encode images
        img_feat = self.image_encoder(images) # (batch, seq, hidden)
        
        # Encode proprio
        proprio_feat = self.proprio_proj(proprio) # (batch, seq, hidden)
        
        # Fuse
        combined = torch.cat([img_feat, proprio_feat], dim=-1)
        x = self.fusion_proj(combined) # (batch, seq, hidden)

        # Transpose for transformer
        x = x.transpose(0, 1)  # (seq_len, batch, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        encoded = self.transformer_encoder(x)  # (seq_len, batch, hidden_dim)

        return encoded


class ACTDecoder(nn.Module):
    """
    Transformer decoder for ACT.

    Decodes latent code and observation encoding into action sequence.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # Latent projection
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # Action query embeddings (learnable)
        self.action_queries = nn.Parameter(
            torch.randn(config.pred_horizon, 1, config.hidden_dim)
        )

        # Positional encoding
        self.pos_decoder = PositionalEncoding(config.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(
        self,
        latent: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent code into action sequence.

        Args:
            latent: (batch_size, latent_dim)
            encoder_output: (obs_horizon, batch_size, hidden_dim)

        Returns:
            Action sequence (batch_size, pred_horizon, action_dim)
        """
        batch_size = latent.shape[0]

        # Project latent
        latent_feat = self.latent_proj(latent)  # (batch, hidden_dim)
        latent_feat = latent_feat.unsqueeze(0)  # (1, batch, hidden_dim)

        # Combine latent with encoder output
        memory = torch.cat([latent_feat, encoder_output], dim=0)

        # Prepare action queries
        queries = self.action_queries.expand(-1, batch_size, -1)  # (pred_horizon, batch, hidden_dim)
        queries = self.pos_decoder(queries)

        # Decode
        decoded = self.transformer_decoder(queries, memory)  # (pred_horizon, batch, hidden_dim)

        # Transpose back
        decoded = decoded.transpose(0, 1)  # (batch, chunk_size, hidden_dim)

        # Project to actions
        actions = self.output_proj(decoded)  # (batch, chunk_size, action_dim)

        return actions


class ACTCVAE(nn.Module):
    """
    Conditional VAE for ACT.

    Learns a latent distribution over action sequences conditioned on observations.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # Encoder (for training)
        self.encoder = ACTEncoder(config)

        # Action encoder (for training)
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim * config.pred_horizon, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim * 2, config.latent_dim)

        # Decoder
        self.decoder = ACTDecoder(config)

    def encode(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        action_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observations and actions to latent distribution.

        Args:
            images: (batch_size, obs_horizon, 3, H, W)
            proprio: (batch_size, obs_horizon, proprio_dim)
            action_seq: (batch_size, pred_horizon, action_dim) - only for training

        Returns:
            mu, logvar, encoder_output
        """
        # Encode observations
        encoder_output = self.encoder(images, proprio)  # (obs_horizon, batch, hidden_dim)

        if action_seq is not None:
            # Training: use action encoder
            batch_size = action_seq.shape[0]
            action_flat = action_seq.reshape(batch_size, -1)
            action_feat = self.action_encoder(action_flat)  # (batch, hidden_dim)

            # Pool encoder output
            obs_feat = encoder_output.mean(dim=0)  # (batch, hidden_dim)

            # Combine
            combined = torch.cat([obs_feat, action_feat], dim=-1)  # (batch, hidden_dim*2)
        else:
            # Inference: use only observations
            obs_feat = encoder_output.mean(dim=0)  # (batch, hidden_dim)
            combined = torch.cat([obs_feat, obs_feat], dim=-1)  # (batch, hidden_dim*2)

        # Latent parameters
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar, encoder_output

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        latent: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent to action sequence.

        Args:
            latent: (batch_size, latent_dim)
            encoder_output: (obs_horizon, batch_size, hidden_dim)

        Returns:
            Action sequence (batch_size, pred_horizon, action_dim)
        """
        return self.decoder(latent, encoder_output)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        action_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: (batch_size, obs_horizon, 3, H, W)
            proprio: (batch_size, obs_horizon, proprio_dim)
            action_seq: (batch_size, pred_horizon, action_dim) - only for training

        Returns:
            reconstructed_actions, mu, logvar
        """
        mu, logvar, encoder_output = self.encode(images, proprio, action_seq)
        latent = self.reparameterize(mu, logvar)
        reconstructed_actions = self.decode(latent, encoder_output)

        return reconstructed_actions, mu, logvar


class ACTPolicy(nn.Module):
    """
    ACT (Action Chunking with Transformers) policy for bimanual manipulation.

    Generates action chunks (sequences of desired poses) using CVAE and Transformers.
    """

    def __init__(
        self,
        config: Optional[ACTConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize ACT policy.

        Args:
            config: ACT configuration
            device: Device for computation ('cpu' or 'cuda')
        """
        super().__init__()
        self.config = config if config is not None else ACTConfig()

        # Create CVAE model
        self.model = ACTCVAE(self.config)

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

    def compute_loss(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ACT CVAE loss (reconstruction + KL).
        """
        reconstructed_actions, mu, logvar = self.model(images, proprio, actions)
        recon_loss = F.mse_loss(reconstructed_actions, actions, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        return recon_loss + self.config.kl_weight * kl_loss

    def inference(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor
    ) -> np.ndarray:
        """
        Run inference to generate action chunk.
        
        Args:
            images: (batch, obs_horizon, 3, H, W)
            proprio: (batch, obs_horizon, proprio_dim)
            
        Returns:
            Action sequence (pred_horizon, action_dim) - normalized
        """
        self.model.eval()
        batch_size = images.shape[0]
        
        with torch.no_grad():
            # Sample latent from prior (zero mean, unit variance)
            latent = torch.randn(batch_size, self.config.latent_dim, device=self._device())
            
            # Encode observations
            encoder_output = self.model.encoder(images, proprio)
            
            # Decode to actions
            action_chunk = self.model.decode(latent, encoder_output)
            
            # Clamp output to normalized range [-1, 1]
            action_chunk = action_chunk.clamp(-1.0, 1.0)
        
        return action_chunk.squeeze(0).cpu().numpy()

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
            # Image is already 0-255 uint8, need to float and normalize
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
            # Generate new action chunk
            img_cond, prop_cond = self._get_obs_condition()

            with torch.no_grad():
                self.model.eval()

                # Sample latent from prior
                batch_size = 1
                latent = torch.randn(batch_size, self.config.latent_dim, device=self._device())

                # Encode observations
                encoder_output = self.model.encoder(img_cond, prop_cond)

                # Decode to actions
                action_chunk = self.model.decode(latent, encoder_output)
                
                # Clamp to normalized range [-1, 1]
                action_chunk = action_chunk.clamp(-1.0, 1.0)

            # Buffer actions
            action_chunk = action_chunk.squeeze(0).cpu().numpy()
            
            # Denormalize actions if normalizer is present
            if self.normalizer is not None:
                action_chunk = self.normalizer.denormalize(action_chunk, 'action')
                
            self.action_buffer = action_chunk
            self.action_idx = 1  # Use first action now
            action = self.action_buffer[0]

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
            Action chunk (chunk_size, 2, 3) containing desired poses for both EEs
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

        # Generate new action chunk
        img_cond, prop_cond = self._get_obs_condition()

        with torch.no_grad():
            self.model.eval()
            batch_size = 1
            latent = torch.randn(batch_size, self.config.latent_dim, device=self._device())
            encoder_output = self.model.encoder(img_cond, prop_cond)
            action_chunk = self.model.decode(latent, encoder_output)
            
            # Clamp to normalized range [-1, 1]
            action_chunk = action_chunk.clamp(-1.0, 1.0)

        action_chunk = action_chunk.squeeze(0).cpu().numpy()
        
        # Denormalize actions if normalizer is present
        if self.normalizer is not None:
            action_chunk = self.normalizer.denormalize(action_chunk, 'action')
            
        # Reshape to (chunk_size, 2, 3)
        # Assuming action_dim is 6 (2 EEs * 3 dims)
        chunk_size = action_chunk.shape[0]
        return action_chunk.reshape(chunk_size, 2, 3)

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

    def train_step(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Single training step.

        Args:
            images: Batch of images (batch_size, obs_horizon, 3, H, W)
            proprio: Batch of proprio (batch_size, obs_horizon, proprio_dim)
            actions: Batch of action sequences (batch_size, chunk_size, action_dim)

        Returns:
            total_loss, reconstruction_loss, kl_loss
        """
        self.model.train()

        # Forward pass
        reconstructed_actions, mu, logvar = self.model(images, proprio, actions)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed_actions, actions, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

        # Total loss
        total_loss = recon_loss + self.config.kl_weight * kl_loss

        return total_loss.item(), recon_loss.item(), kl_loss.item()

    def save(self, path: str):
        """Save policy to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """
        Load policy from file.
        
        Handles two checkpoint formats:
        1. From save() method: {'config', 'model_state_dict'}
        2. From training script: {'model_state_dict', 'normalizer', ...}
        """
        device = self._device()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # If checkpoint has 'config', create new model from it
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            self.model = ACTCVAE(self.config).to(device)
        
        # Load model weights (model must already exist)
        if self.model is None:
            raise ValueError("Model not initialized. Create policy with config first.")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
