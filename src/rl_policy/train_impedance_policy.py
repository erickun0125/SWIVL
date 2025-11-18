"""
Training Script for Impedance Parameter Learning

This script trains a PPO agent to learn optimal impedance parameters
for bimanual manipulation tasks.

The training pipeline:
1. Load or create a pre-trained high-level policy (Flow Matching, Diffusion, or ACT)
2. Create impedance learning environment
3. Train PPO agent to optimize impedance parameters
4. Evaluate and save the trained policy

Usage:
    python -m src.rl_policy.train_impedance_policy \
        --hl_policy flow_matching \
        --hl_policy_path checkpoints/flow_matching.pth \
        --total_timesteps 1000000 \
        --output_path checkpoints/impedance_policy.zip

Example:
    # Train with Flow Matching policy
    python -m src.rl_policy.train_impedance_policy \
        --hl_policy flow_matching \
        --total_timesteps 500000

    # Train with Diffusion Policy
    python -m src.rl_policy.train_impedance_policy \
        --hl_policy diffusion \
        --hl_policy_path checkpoints/diffusion.pth \
        --total_timesteps 1000000

    # Train with ACT
    python -m src.rl_policy.train_impedance_policy \
        --hl_policy act \
        --hl_policy_path checkpoints/act.pth \
        --total_timesteps 1000000

    # Train without high-level policy (random exploration)
    python -m src.rl_policy.train_impedance_policy \
        --hl_policy none \
        --total_timesteps 200000
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.rl_policy.impedance_learning_env import ImpedanceLearningEnv, ImpedanceLearningConfig
from src.rl_policy.ppo_impedance_policy import PPOImpedancePolicy
from src.hl_planners.flow_matching import FlowMatchingPolicy
from src.hl_planners.diffusion_policy import DiffusionPolicy
from src.hl_planners.act import ACTPolicy


def load_hl_policy(policy_type: str, policy_path: Optional[str] = None, device: str = 'cpu'):
    """
    Load high-level policy.

    Args:
        policy_type: Type of policy ('flow_matching', 'diffusion', 'act', or 'none')
        policy_path: Path to policy checkpoint
        device: Device for computation

    Returns:
        Loaded policy or None
    """
    if policy_type == 'none':
        print("Training without high-level policy (random exploration)")
        return None

    print(f"Loading {policy_type} policy...")

    if policy_type == 'flow_matching':
        policy = FlowMatchingPolicy(device=device)
        if policy_path is not None and os.path.exists(policy_path):
            policy.load(policy_path)
            print(f"Loaded policy from {policy_path}")
        else:
            print("Warning: No checkpoint provided, using randomly initialized policy")

    elif policy_type == 'diffusion':
        policy = DiffusionPolicy(device=device)
        if policy_path is not None and os.path.exists(policy_path):
            policy.load(policy_path)
            print(f"Loaded policy from {policy_path}")
        else:
            print("Warning: No checkpoint provided, using randomly initialized policy")

    elif policy_type == 'act':
        policy = ACTPolicy(device=device)
        if policy_path is not None and os.path.exists(policy_path):
            policy.load(policy_path)
            print(f"Loaded policy from {policy_path}")
        else:
            print("Warning: No checkpoint provided, using randomly initialized policy")

    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    return policy


def create_env(
    hl_policy,
    render_mode: Optional[str] = None,
    config: Optional[ImpedanceLearningConfig] = None
) -> ImpedanceLearningEnv:
    """
    Create impedance learning environment.

    Args:
        hl_policy: High-level policy
        render_mode: Rendering mode
        config: Environment configuration

    Returns:
        Environment
    """
    if config is None:
        config = ImpedanceLearningConfig()

    env = ImpedanceLearningEnv(
        config=config,
        hl_policy=hl_policy,
        render_mode=render_mode
    )

    return env


def train(
    hl_policy_type: str = 'flow_matching',
    hl_policy_path: Optional[str] = None,
    total_timesteps: int = 1000000,
    output_path: str = 'checkpoints/impedance_policy.zip',
    tensorboard_log: str = './logs/impedance_rl/',
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    device: str = 'auto',
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    verbose: int = 1
):
    """
    Train impedance parameter learning policy.

    Args:
        hl_policy_type: Type of high-level policy
        hl_policy_path: Path to high-level policy checkpoint
        total_timesteps: Total training timesteps
        output_path: Output path for trained policy
        tensorboard_log: Path for tensorboard logs
        learning_rate: Learning rate
        n_steps: Number of steps per update
        batch_size: Batch size
        device: Device for computation
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
    """
    print("=" * 80)
    print("Training Impedance Parameter Learning Policy")
    print("=" * 80)

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load high-level policy
    hl_policy = load_hl_policy(hl_policy_type, hl_policy_path, device)

    # Create environment
    print("\nCreating environment...")
    env_config = ImpedanceLearningConfig(
        max_episode_steps=1000,
        tracking_weight=1.0,
        wrench_weight=0.1,
        smoothness_weight=0.01
    )
    env = create_env(hl_policy, render_mode=None, config=env_config)
    print("Environment created")

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_env(hl_policy, render_mode=None, config=env_config)
    print("Evaluation environment created")

    # Create PPO policy
    print("\nCreating PPO policy...")
    ppo_policy = PPOImpedancePolicy(
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        features_dim=256,
        device=device,
        verbose=verbose,
        tensorboard_log=tensorboard_log
    )
    print("PPO policy created")

    # Train
    print("\n" + "=" * 80)
    print(f"Starting training for {total_timesteps} timesteps...")
    print("=" * 80)

    ppo_policy.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # Evaluate final policy
    print("\nEvaluating final policy...")
    eval_results = ppo_policy.evaluate(n_episodes=10, deterministic=True)
    print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Mean episode length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")

    # Save policy
    print(f"\nSaving policy to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ppo_policy.save(output_path)
    print("Policy saved!")

    # Close environments
    env.close()
    eval_env.close()

    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Train PPO agent for impedance parameter learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # High-level policy arguments
    parser.add_argument(
        '--hl_policy',
        type=str,
        default='flow_matching',
        choices=['flow_matching', 'diffusion', 'act', 'none'],
        help='Type of high-level policy'
    )
    parser.add_argument(
        '--hl_policy_path',
        type=str,
        default=None,
        help='Path to high-level policy checkpoint'
    )

    # Training arguments
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=1000000,
        help='Total number of training timesteps'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=2048,
        help='Number of steps per update'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )

    # Output arguments
    parser.add_argument(
        '--output_path',
        type=str,
        default='checkpoints/impedance_policy.zip',
        help='Output path for trained policy'
    )
    parser.add_argument(
        '--tensorboard_log',
        type=str,
        default='./logs/impedance_rl/',
        help='Path for tensorboard logs'
    )

    # Evaluation arguments
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=10000,
        help='Evaluation frequency'
    )
    parser.add_argument(
        '--n_eval_episodes',
        type=int,
        default=5,
        help='Number of evaluation episodes'
    )

    # Device argument
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for computation'
    )

    # Verbosity
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level'
    )

    args = parser.parse_args()

    # Train
    train(
        hl_policy_type=args.hl_policy,
        hl_policy_path=args.hl_policy_path,
        total_timesteps=args.total_timesteps,
        output_path=args.output_path,
        tensorboard_log=args.tensorboard_log,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
