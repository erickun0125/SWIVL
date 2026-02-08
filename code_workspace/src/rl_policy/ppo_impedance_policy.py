"""
SWIVL PPO-based Impedance Parameter Learning Policy (Layer 3)

Implements a PPO (Proximal Policy Optimization) agent for learning
optimal impedance modulation variables for bimanual manipulation.

The policy learns the SWIVL Layer 3 action:
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

Per SWIVL Paper Section 3.3:
- Observation: Reference twists, Screw axes, Wrenches, Proprioception (30D)
- Architecture: FiLM-conditioned multi-stream encoder
- Training: PPO with GAE

Policy Architecture (per SWIVL Paper Section 3.3.3):
- Separate encoders for reference twists, wrenches, and proprioception
- FiLM (Feature-wise Linear Modulation) layers inject screw axes
- Enables dynamic adaptation across joint types

Uses Stable-Baselines3 for the PPO implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Per SWIVL Paper Section 3.3.3:
    FiLM layers inject object geometry (screw axes) into feature processing,
    enabling dynamic adaptation across joint types.
    
    FiLM modulation: y = γ(c) * x + β(c)
    where γ, β are learned functions of conditioning input c (screw axes).
    
    Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
    """
    
    def __init__(self, feature_dim: int, condition_dim: int):
        """
        Args:
            feature_dim: Dimension of features to modulate
            condition_dim: Dimension of conditioning input (screw axes)
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Conditioning networks for γ (scale) and β (shift)
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Initialize γ to 1 and β to 0 for identity at init
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            x: Features to modulate (batch, feature_dim)
            condition: Conditioning input (batch, condition_dim)
            
        Returns:
            Modulated features (batch, feature_dim)
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma * x + beta


class SWIVLFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for SWIVL impedance learning with FiLM conditioning.

    Per SWIVL Paper Section 3.3.3:
    - Multi-stream architecture with separate encoders for:
      - Reference twists (from Layer 2)
      - Wrench feedback (from F/T sensors)
      - Proprioception (poses + twists)
    - FiLM layers modulate features based on screw axes {B_l, B_r}
    - Enables dynamic adaptation across joint types (revolute, prismatic)
    
    Observation structure (per SWIVL Paper Appendix B):
    [ref_twists(6), screw_axes(6), wrenches(6), poses(6), twists(6)] = 30D
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

        # Per SWIVL Paper Appendix B:
        # [ref_twists(6), screw_axes(6), wrenches(6), poses(6), twists(6)]
        self.ref_twist_dim = 6   # Reference twists
        self.screw_dim = 6       # Screw axes (conditioning)
        self.wrench_dim = 6      # Wrench feedback
        self.pose_dim = 6        # EE poses
        self.twist_dim = 6       # EE body twists

        # =====================================================================
        # Reference Twist Encoder
        # Encodes V_l^ref, V_r^ref from Layer 2
        # =====================================================================
        self.ref_twist_encoder = nn.Sequential(
            nn.Linear(self.ref_twist_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # FiLM layer for ref twist features (conditioned on screw axes)
        self.ref_twist_film = FiLMLayer(64, self.screw_dim)

        # =====================================================================
        # Wrench Encoder
        # Encodes F_l, F_r from F/T sensors
        # =====================================================================
        self.wrench_encoder = nn.Sequential(
            nn.Linear(self.wrench_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # FiLM layer for wrench features (conditioned on screw axes)
        # This enables learning force decomposition based on joint type
        self.wrench_film = FiLMLayer(64, self.screw_dim)

        # =====================================================================
        # Proprioception Encoder
        # Encodes poses and twists together
        # =====================================================================
        proprio_input_dim = self.pose_dim + self.twist_dim  # 12D
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU()
        )
        
        # FiLM layer for proprio features
        self.proprio_film = FiLMLayer(96, self.screw_dim)

        # =====================================================================
        # Feature Combiner
        # Combines all FiLM-modulated features
        # =====================================================================
        combined_dim = 64 + 64 + 96  # 224
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations with FiLM conditioning.

        Args:
            observations: (batch_size, 30) observations
                Structure: [ref_twists, screw_axes, wrenches, poses, twists]

        Returns:
            (batch_size, features_dim) features
        """
        # Parse observation per SWIVL Paper Appendix B
        ref_twists = observations[:, 0:6]         # Reference twists
        screw_axes = observations[:, 6:12]        # Screw axes (conditioning)
        wrenches = observations[:, 12:18]         # Wrench feedback
        poses = observations[:, 18:24]            # EE poses
        twists = observations[:, 24:30]           # EE body twists

        # =====================================================================
        # Encode each stream
        # =====================================================================
        
        # Reference twist features with FiLM
        ref_features = self.ref_twist_encoder(ref_twists)
        ref_features = self.ref_twist_film(ref_features, screw_axes)
        
        # Wrench features with FiLM
        wrench_features = self.wrench_encoder(wrenches)
        wrench_features = self.wrench_film(wrench_features, screw_axes)
        
        # Proprioception features with FiLM
        proprio_input = torch.cat([poses, twists], dim=-1)
        proprio_features = self.proprio_encoder(proprio_input)
        proprio_features = self.proprio_film(proprio_features, screw_axes)

        # =====================================================================
        # Combine all features
        # =====================================================================
        combined = torch.cat([ref_features, wrench_features, proprio_features], dim=-1)
        features = self.combiner(combined)

        return features


class SWIVLLoggingCallback(BaseCallback):
    """
    Callback for logging SWIVL impedance learning metrics.

    Per SWIVL Paper Section 3.3:
    Logs:
    - Impedance parameter statistics (d_∥, d_⊥, k_p, α)
    - Fighting force metrics (F_⊥)
    - G-metric tracking performance
    - Per-arm impedance modulation patterns
    - Termination statistics (grasp drift, wrench limit)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.fighting_forces = []
        self.failure_terminations = []
        self.termination_reasons = {'grasp_drift': 0, 'wrench_limit': 0, 'success': 0}

    def _on_step(self) -> bool:
        """Called at each step."""
        # Track fighting force and termination info
        for info in self.locals.get('infos', []):
            if 'control_info' in info and info['control_info']:
                ff = info['control_info'].get('total_fighting_force', 0.0)
                self.fighting_forces.append(ff)
            
            # Track failure terminations
            if info.get('failure_termination', False):
                self.failure_terminations.append(1)
                reason = info.get('termination_reason', 'unknown')
                if reason in self.termination_reasons:
                    self.termination_reasons[reason] += 1
            elif self.locals.get('dones', [False])[0]:
                self.termination_reasons['success'] += 1
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if not hasattr(self.model, 'rollout_buffer') or not self.model.rollout_buffer.full:
            return

        actions = self.model.rollout_buffer.actions
        rewards = self.model.rollout_buffer.rewards
        observations = self.model.rollout_buffer.observations

        # Flatten to (N, action_dim)
        flat_actions = actions.reshape(-1, actions.shape[-1])
        flat_obs = observations.reshape(-1, observations.shape[-1])
        action_dim = flat_actions.shape[-1]

        if action_dim == 7:
            # =====================================================================
            # SWIVL screw-decomposed: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)
            # =====================================================================
            
            # Per-arm parallel damping (internal motion)
            self.logger.record('swivl/d_l_parallel_mean', np.mean(flat_actions[:, 0]))
            self.logger.record('swivl/d_r_parallel_mean', np.mean(flat_actions[:, 1]))
            self.logger.record('swivl/d_parallel_mean', np.mean(flat_actions[:, 0:2]))
            self.logger.record('swivl/d_parallel_std', np.std(flat_actions[:, 0:2]))
            
            # Per-arm perpendicular damping (bulk motion)
            self.logger.record('swivl/d_l_perp_mean', np.mean(flat_actions[:, 2]))
            self.logger.record('swivl/d_r_perp_mean', np.mean(flat_actions[:, 3]))
            self.logger.record('swivl/d_perp_mean', np.mean(flat_actions[:, 2:4]))
            self.logger.record('swivl/d_perp_std', np.std(flat_actions[:, 2:4]))
            
            # Per-arm stiffness (pose error correction)
            self.logger.record('swivl/k_p_l_mean', np.mean(flat_actions[:, 4]))
            self.logger.record('swivl/k_p_r_mean', np.mean(flat_actions[:, 5]))
            self.logger.record('swivl/k_p_mean', np.mean(flat_actions[:, 4:6]))
            self.logger.record('swivl/k_p_std', np.std(flat_actions[:, 4:6]))
            
            # Characteristic length α (metric tensor)
            self.logger.record('swivl/alpha_mean', np.mean(flat_actions[:, 6]))
            self.logger.record('swivl/alpha_std', np.std(flat_actions[:, 6]))
            
            # Damping ratio (d_perp / d_parallel) - should be > 1 for stiff bulk motion
            d_parallel_avg = np.mean(flat_actions[:, 0:2], axis=1) + 1e-6
            d_perp_avg = np.mean(flat_actions[:, 2:4], axis=1)
            damping_ratio = d_perp_avg / d_parallel_avg
            self.logger.record('swivl/damping_ratio_mean', np.mean(damping_ratio))

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
        self.logger.record('rollout/reward_min', np.min(flat_rewards))
        self.logger.record('rollout/reward_max', np.max(flat_rewards))
        
        # Log fighting force if tracked
        if len(self.fighting_forces) > 0:
            self.logger.record('swivl/fighting_force_mean', np.mean(self.fighting_forces))
            self.logger.record('swivl/fighting_force_max', np.max(self.fighting_forces))
            self.fighting_forces = []  # Reset for next rollout
        
        # Log termination statistics
        total_terms = sum(self.termination_reasons.values())
        if total_terms > 0:
            self.logger.record('swivl/term_grasp_drift_rate', 
                             self.termination_reasons['grasp_drift'] / total_terms)
            self.logger.record('swivl/term_wrench_limit_rate', 
                             self.termination_reasons['wrench_limit'] / total_terms)
            self.logger.record('swivl/term_success_rate', 
                             self.termination_reasons['success'] / total_terms)
            self.logger.record('swivl/failure_termination_count', 
                             len(self.failure_terminations))
        
        # Reset termination tracking
        self.failure_terminations = []
        self.termination_reasons = {'grasp_drift': 0, 'wrench_limit': 0, 'success': 0}


class PPOImpedancePolicy:
    """
    PPO-based policy for SWIVL impedance parameter learning.

    Learns to modulate impedance variables:
    - Parallel damping d_∥ (internal motion, compliant)
    - Perpendicular damping d_⊥ (bulk motion, stiff)
    - Pose error gain k_p
    - Metric tensor scale α

    Uses custom feature extractor optimized for bimanual manipulation.
    
    Prefer using from_config() class method for creation from rl_config.yaml.
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

    @classmethod
    def from_config(
        cls,
        env: gym.Env,
        config: Dict[str, Any],
        device: str = 'auto'
    ) -> 'PPOImpedancePolicy':
        """
        Create PPO policy from configuration dictionary.
        
        This is the preferred way to create a policy instance.
        Load config from rl_config.yaml for consistent settings.
        
        Args:
            env: Impedance learning environment
            config: Full configuration dictionary from rl_config.yaml
            device: Device for computation ('auto', 'cpu', or 'cuda')
            
        Returns:
            PPOImpedancePolicy instance
            
        Example:
            import yaml
            with open('scripts/configs/rl_config.yaml') as f:
                config = yaml.safe_load(f)
            policy = PPOImpedancePolicy.from_config(env, config)
        """
        rl_cfg = config.get('rl_training', {})
        ppo_cfg = rl_cfg.get('ppo', {})
        network_cfg = rl_cfg.get('network', {})
        logging_cfg = rl_cfg.get('logging', {})
        
        return cls(
            env=env,
            learning_rate=ppo_cfg.get('learning_rate', 3e-4),
            n_steps=ppo_cfg.get('n_steps', 2048),
            batch_size=ppo_cfg.get('batch_size', 64),
            n_epochs=ppo_cfg.get('n_epochs', 10),
            gamma=ppo_cfg.get('gamma', 0.99),
            gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
            clip_range=ppo_cfg.get('clip_range', 0.2),
            ent_coef=ppo_cfg.get('ent_coef', 0.01),
            vf_coef=ppo_cfg.get('vf_coef', 0.5),
            max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
            features_dim=network_cfg.get('features_dim', 256),
            device=device,
            verbose=logging_cfg.get('verbose', 1),
            tensorboard_log=logging_cfg.get('tensorboard_log', './logs/impedance_rl/')
        )

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
