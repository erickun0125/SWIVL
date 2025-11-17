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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching policy."""
    state_dim: int = 18  # 2 EE poses (6) + 2 link poses (6) + external wrenches (6)
    action_dim: int = 6  # 2 EE desired poses (3 each)
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
        self.action_mlp = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim),
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
        self.output_net = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Current trajectory sample (batch_size, action_dim)
            t: Time parameter in [0, 1] (batch_size, 1)
            context: Context state (batch_size, state_dim)

        Returns:
            Predicted velocity field (batch_size, action_dim)
        """
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


class FlowMatchingPolicy:
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
        self.config = config if config is not None else FlowMatchingConfig()
        self.device = device

        # Create network
        self.network = FlowMatchingNetwork(self.config).to(device)

        # State history for context
        self.state_history: List[np.ndarray] = []

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
            observation: Current observation dict with keys:
                - 'ee_poses': (2, 3) array of EE poses
                - 'link_poses': (2, 3) array of link poses
                - 'external_wrenches': (2, 3) array of wrenches
            goal: Optional goal specification

        Returns:
            Desired poses (2, 3) for both end-effectors
        """
        # Construct state vector
        state = self._construct_state(observation)

        # Add to history
        self.state_history.append(state)
        if len(self.state_history) > self.config.context_length:
            self.state_history.pop(0)

        # Pad history if needed
        context = self._get_context()

        # Sample from flow
        with torch.no_grad():
            action = self._sample_flow(context, goal)

        # Convert to desired poses
        desired_poses = action.reshape(2, 3)

        return desired_poses

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

        return torch.FloatTensor(context).unsqueeze(0).to(self.device)

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
            Sampled action
        """
        # Start from noise (or goal if provided)
        if goal is not None:
            x = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        else:
            x = torch.randn(1, self.config.action_dim).to(self.device)

        # Integrate flow ODE using Euler method
        dt = 1.0 / self.config.num_diffusion_steps

        for step in range(self.config.num_diffusion_steps):
            t = torch.FloatTensor([step * dt]).unsqueeze(0).to(self.device)

            # Predict velocity
            velocity = self.network(x, t, context)

            # Euler step
            x = x + velocity * dt

        return x.cpu().numpy().flatten()

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
