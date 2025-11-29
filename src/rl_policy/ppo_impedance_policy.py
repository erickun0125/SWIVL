"""
SWIVL PPO-based Impedance Parameter Learning Policy

Implements a PPO (Proximal Policy Optimization) agent for learning
optimal impedance modulation variables for bimanual manipulation.

The policy learns the SWIVL Layer 3 action:
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

Training objectives:
1. Track desired trajectories from high-level policies
2. Minimize fighting force (F_⊥) during manipulation
3. Adapt impedance parameters based on task requirements

Uses Stable-Baselines3 for the PPO implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym


class SWIVLFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for SWIVL impedance learning.

    Processes bimanual observations with separate encoders for:
    - External wrenches (contact/interaction forces)
    - Pose information (current + desired)
    - Twist information (current + desired velocities)

    Architecture designed to capture:
    - Per-arm force/torque patterns
    - Bimanual coordination information
    - Tracking error dynamics
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Initialize feature extractor.

        Args:
            observation_space: Observation space (30D for bimanual)
            features_dim: Dimension of output features
        """
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]

        # Each component is 6D (3D per arm * 2 arms)
        # Structure: [wrenches(6), poses(6), twists(6), des_poses(6), des_twists(6)]
        self.component_dim = 6

        # Wrench encoder - captures force/torque patterns
        self.wrench_encoder = nn.Sequential(
            nn.Linear(self.component_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Pose encoder - captures position errors (current + desired = 12D)
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.component_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU()
        )

        # Twist encoder - captures velocity errors (current + desired = 12D)
        self.twist_encoder = nn.Sequential(
            nn.Linear(self.component_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU()
        )

        # Combine all features
        combined_dim = 64 + 96 + 96  # 256
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: (batch_size, 30) observations

        Returns:
            (batch_size, features_dim) features
        """
        # Split observations: [wrenches, poses, twists, des_poses, des_twists]
        dim = self.component_dim

        wrenches = observations[:, 0:dim]            # External wrenches
        current_poses = observations[:, dim:2*dim]    # Current poses
        current_twists = observations[:, 2*dim:3*dim] # Current twists
        desired_poses = observations[:, 3*dim:4*dim]  # Desired poses
        desired_twists = observations[:, 4*dim:5*dim] # Desired twists

        # Encode wrenches (6D)
        wrench_features = self.wrench_encoder(wrenches)

        # Encode poses with error context (12D)
        pose_input = torch.cat([current_poses, desired_poses], dim=-1)
        pose_features = self.pose_encoder(pose_input)

        # Encode twists with error context (12D)
        twist_input = torch.cat([current_twists, desired_twists], dim=-1)
        twist_features = self.twist_encoder(twist_input)

        # Combine features
        combined = torch.cat([wrench_features, pose_features, twist_features], dim=-1)
        features = self.combiner(combined)

        return features


class SWIVLLoggingCallback(BaseCallback):
    """
    Callback for logging SWIVL impedance learning metrics.

    Logs:
    - Impedance parameter statistics (d_∥, d_⊥, k_p, α)
    - Fighting force metrics
    - Tracking performance
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at each step."""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if not hasattr(self.model, 'rollout_buffer') or not self.model.rollout_buffer.full:
            return

        actions = self.model.rollout_buffer.actions
        rewards = self.model.rollout_buffer.rewards

        # Flatten to (N, action_dim)
        flat_actions = actions.reshape(-1, actions.shape[-1])
        action_dim = flat_actions.shape[-1]

        if action_dim == 7:
            # SWIVL screw-decomposed: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)
            d_parallel_mean = np.mean(flat_actions[:, 0:2])
            d_parallel_std = np.std(flat_actions[:, 0:2])
            d_perp_mean = np.mean(flat_actions[:, 2:4])
            d_perp_std = np.std(flat_actions[:, 2:4])
            k_p_mean = np.mean(flat_actions[:, 4:6])
            k_p_std = np.std(flat_actions[:, 4:6])
            alpha_mean = np.mean(flat_actions[:, 6])
            alpha_std = np.std(flat_actions[:, 6])

            self.logger.record('swivl/d_parallel_mean', d_parallel_mean)
            self.logger.record('swivl/d_parallel_std', d_parallel_std)
            self.logger.record('swivl/d_perp_mean', d_perp_mean)
            self.logger.record('swivl/d_perp_std', d_perp_std)
            self.logger.record('swivl/k_p_mean', k_p_mean)
            self.logger.record('swivl/k_p_std', k_p_std)
            self.logger.record('swivl/alpha_mean', alpha_mean)
            self.logger.record('swivl/alpha_std', alpha_std)

        elif action_dim == 12:
            # Classical SE(2) impedance
            damping_mean = np.mean(flat_actions[:, [0, 1, 2, 6, 7, 8]])
            damping_std = np.std(flat_actions[:, [0, 1, 2, 6, 7, 8]])
            stiffness_mean = np.mean(flat_actions[:, [3, 4, 5, 9, 10, 11]])
            stiffness_std = np.std(flat_actions[:, [3, 4, 5, 9, 10, 11]])

            self.logger.record('impedance/damping_mean', damping_mean)
            self.logger.record('impedance/damping_std', damping_std)
            self.logger.record('impedance/stiffness_mean', stiffness_mean)
            self.logger.record('impedance/stiffness_std', stiffness_std)

        # Log reward statistics
        flat_rewards = rewards.flatten()
        self.logger.record('rollout/reward_mean', np.mean(flat_rewards))
        self.logger.record('rollout/reward_std', np.std(flat_rewards))


class PPOImpedancePolicy:
    """
    PPO-based policy for SWIVL impedance parameter learning.

    Learns to modulate impedance variables:
    - Parallel damping d_∥ (internal motion, compliant)
    - Perpendicular damping d_⊥ (bulk motion, stiff)
    - Pose error gain k_p
    - Metric tensor scale α

    Uses custom feature extractor optimized for bimanual manipulation.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        features_dim: int = 256,
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: Optional[str] = None
    ):
        """
        Initialize PPO impedance policy.

        Args:
            env: Impedance learning environment
            learning_rate: Learning rate
            n_steps: Steps per environment per update
            batch_size: Minibatch size
            n_epochs: Optimization epochs per update
            gamma: Discount factor
            gae_lambda: GAE parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Gradient clipping
            features_dim: Feature extractor output dimension
            device: Computation device
            verbose: Verbosity level
            tensorboard_log: TensorBoard log path
        """
        self.env = env
        self.features_dim = features_dim

        # Custom policy with SWIVL feature extractor
        policy_kwargs = dict(
            features_extractor_class=SWIVLFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log
        )

        # SWIVL logging callback
        self.logging_callback = SWIVLLoggingCallback(verbose=verbose)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5
    ) -> 'PPOImpedancePolicy':
        """
        Train the policy.

        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback(s)
            log_interval: Logging interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Evaluation episodes

        Returns:
            self
        """
        callbacks = [self.logging_callback]
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)

        # Add evaluation callback
        if eval_env is not None:
            from stable_baselines3.common.callbacks import EvalCallback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval
        )

        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Predict action (impedance parameters) from observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            Action (impedance parameters in [-1, 1])
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def get_impedance_parameters(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get decoded impedance parameters from observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            Dictionary with impedance parameters
        """
        action = self.predict(observation, deterministic=deterministic)

        # Use environment's decoder if available
        if hasattr(self.env, 'config'):
            config = self.env.config

            if hasattr(config, 'controller_type') and config.controller_type == 'screw_decomposed':
                # SWIVL action: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)
                def scale(val, min_v, max_v):
                    return min_v + (val + 1.0) * 0.5 * (max_v - min_v)

                return {
                    'd_l_parallel': scale(action[0], config.min_d_parallel, config.max_d_parallel),
                    'd_r_parallel': scale(action[1], config.min_d_parallel, config.max_d_parallel),
                    'd_l_perp': scale(action[2], config.min_d_perp, config.max_d_perp),
                    'd_r_perp': scale(action[3], config.min_d_perp, config.max_d_perp),
                    'k_p_l': scale(action[4], config.min_k_p, config.max_k_p),
                    'k_p_r': scale(action[5], config.min_k_p, config.max_k_p),
                    'alpha': scale(action[6], config.min_alpha, config.max_alpha),
                }

        return {'raw_action': action}

    def save(self, path: str):
        """Save policy to file."""
        self.model.save(path)

    def load(self, path: str, env: Optional[gym.Env] = None):
        """Load policy from file."""
        if env is None:
            env = self.env
        self.model = PPO.load(path, env=env)

    @classmethod
    def load_from_file(cls, path: str, env: gym.Env) -> 'PPOImpedancePolicy':
        """Load policy from file (class method)."""
        policy = cls(env, verbose=0)
        policy.load(path, env)
        return policy

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate policy.

        Args:
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
            render: Render episodes

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        total_fighting_force = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            ep_fighting_force = 0.0

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Track fighting force if available
                if 'control_info' in info and info['control_info']:
                    ep_fighting_force += info['control_info'].get('total_fighting_force', 0.0)

                if render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_fighting_force.append(ep_fighting_force / max(episode_length, 1))

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_fighting_force': np.mean(total_fighting_force),
            'std_fighting_force': np.std(total_fighting_force)
        }
