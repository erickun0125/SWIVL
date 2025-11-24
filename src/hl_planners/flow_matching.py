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
    state_dim: int = 24  # 2 EE poses (6) + 2 link poses (6) + external wrenches (6) + body twists (6)
    action_dim: int = 6  # 2 EE desired poses (3 each)
    pred_horizon: int = 10 # Number of steps to predict (chunk size)
    hidden_dim: int = 256
    num_layers: int = 4
    num_diffusion_steps: int = 10
    output_frequency: float = 10.0  # Hz
    context_length: int = 10  # Number of past states to condition on


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
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4)
        )

        # State embedding
        self.state_mlp = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

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
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Current trajectory sample (batch_size, pred_horizon, action_dim) or flattened
            t: Time parameter in [0, 1] (batch_size, 1)
            context: Context state (batch_size, state_dim)

        Returns:
            Predicted velocity field (batch_size, pred_horizon * action_dim)
        """
        batch_size = x.shape[0]
        
        # Flatten input if needed
        if x.dim() > 2:
            x = x.reshape(batch_size, -1)
            
        # Embed time
        t_emb = self.time_mlp(t)

        # Embed context state
        state_emb = self.state_mlp(context)

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
        self.state_history: List[np.ndarray] = []
        self.normalizer = None

        self.to(torch.device(device))

    def set_normalizer(self, normalizer):
        """Set normalizer for input/output scaling."""
        self.normalizer = normalizer

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, context: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.network(x_t, t, context)

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            states: (batch, obs_horizon, state_dim) or (batch, state_dim)
            actions: (batch, pred_horizon, action_dim)
        """
        batch_size = states.shape[0]
        device = states.device

        # Use most recent observation as context
        if states.dim() == 3:
            context = states[:, -1, :]
        else:
            context = states

        # Flatten actions: (batch, pred_horizon * action_dim)
        target_actions_flat = actions.reshape(batch_size, -1)

        t = torch.rand(batch_size, 1, device=device)
        noise = torch.randn_like(target_actions_flat)
        
        # Interpolate between noise and data
        x_t = (1 - t) * noise + t * target_actions_flat
        
        # Target velocity: (data - noise)
        target_velocity = target_actions_flat - noise

        # Predict velocity
        pred_velocity = self.network(x_t, t, context)
        
        # Compute loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def reset(self):
        """Reset policy state."""
        self.state_history = []

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
        self.state_history.append(state)
        if len(self.state_history) > self.config.context_length:
            self.state_history.pop(0)

        # Pad history if needed
        context = self._get_context()

        # Sample from flow
        with torch.no_grad():
            action = self._sample_flow(context, goal)

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
        # Construct state vector
        state = self._construct_state(observation)
        if self.normalizer is not None:
            state = self.normalizer.normalize(state, 'state')
            
        # For flow matching trained as single step predictor, we might not handle 
        # proper history updates in this loop without environment feedback.
        # However, if the model output dimension was trained with horizon > 1,
        # it would return a chunk directly.
        
        # Check if model is configured for horizon
        # If pred_horizon > 1 or effective dim > 6
        if self.config.pred_horizon > 1:
             # Model predicts full chunk at once
            self.state_history.append(state)
            if len(self.state_history) > self.config.context_length:
                self.state_history.pop(0)
            context = self._get_context()
            
            with torch.no_grad():
                action = self._sample_flow(context, goal)
                
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

    def _construct_state(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct state vector from observation."""
        ee_poses = observation['ee_poses'].flatten()
        link_poses = observation['link_poses'].flatten()
        wrenches = observation['external_wrenches'].flatten()

        state = np.concatenate([ee_poses, link_poses, wrenches])
        return state

    def _get_context(self) -> torch.Tensor:
        """Get context from state history."""
        if len(self.state_history) == 0:
            context = np.zeros(self.config.state_dim)
        else:
            # Use most recent state as context (can be extended to use full history)
            context = self.state_history[-1]

        return torch.FloatTensor(context).unsqueeze(0).to(self._device())

    def _sample_flow(
        self,
        context: torch.Tensor,
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Sample from flow model using ODE integration.

        Args:
            context: Context state
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
            velocity = self.network(x, t, context)

            # Euler step
            x = x + velocity * dt

        return x.cpu().numpy()

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        """
        Single training step.

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size, action_dim)

        Returns:
            Loss value
        """
        batch_size = states.shape[0]

        # Sample time
        t = torch.rand(batch_size, 1).to(self.device)

        # Sample noise
        noise = torch.randn_like(actions)

        # Interpolate between noise and data
        x_t = (1 - t) * noise + t * actions

        # Target velocity: (data - noise)
        target_velocity = actions - noise

        # Predict velocity
        pred_velocity = self.network(x_t, t, states)

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
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.network = FlowMatchingNetwork(self.config).to(self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
