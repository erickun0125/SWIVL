"""
Low-Level Policy Training Script

Trains RL policy for learning optimal impedance parameters in hierarchical control.
Works with ANY high-level policy type (Flow Matching, Diffusion, ACT) and
BOTH controller types (SE(2) impedance, screw-decomposed impedance).

The training is fully configurable via YAML configuration file.

Usage:
    python scripts/training/train_ll_policy.py --config configs/rl_config.yaml
    python scripts/training/train_ll_policy.py --hl_policy flow_matching --controller se2_impedance
    python scripts/training/train_ll_policy.py --hl_policy diffusion --controller screw_decomposed

Example:
    # Train with Flow Matching HL policy and SE(2) impedance controller
    python scripts/training/train_ll_policy.py --hl_policy flow_matching --controller se2_impedance

    # Train with Diffusion HL policy and screw-decomposed controller
    python scripts/training/train_ll_policy.py --hl_policy diffusion --controller screw_decomposed

    # Train with custom config
    python scripts/training/train_ll_policy.py --config my_config.yaml

    # Train without HL policy (random exploration)
    python scripts/training/train_ll_policy.py --hl_policy none --controller se2_impedance
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rl_policy.impedance_learning_env import ImpedanceLearningEnv, ImpedanceLearningConfig
from src.rl_policy.ppo_impedance_policy import PPOImpedancePolicy
from src.hl_planners.flow_matching import FlowMatchingPolicy
from src.hl_planners.diffusion_policy import DiffusionPolicy
from src.hl_planners.act import ACTPolicy


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_hl_policy(
    policy_type: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu'
):
    """
    Load high-level policy.

    Args:
        policy_type: Type of policy ('flow_matching', 'diffusion', 'act', or 'none')
        checkpoint_path: Path to policy checkpoint
        device: Device for computation

    Returns:
        Loaded policy or None
    """
    if policy_type == 'none':
        print("Training without high-level policy (random exploration)")
        return None

    print(f"Loading {policy_type} high-level policy...")

    if policy_type == 'flow_matching':
        policy = FlowMatchingPolicy(device=device)
    elif policy_type == 'diffusion':
        policy = DiffusionPolicy(device=device)
    elif policy_type == 'act':
        policy = ACTPolicy(device=device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        policy.load(checkpoint_path)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint provided, using randomly initialized policy")

    return policy


def create_env_config(config: Dict[str, Any], controller_type: str) -> ImpedanceLearningConfig:
    """
    Create environment configuration from YAML config.

    Args:
        config: Full configuration dictionary
        controller_type: Controller type ('se2_impedance' or 'screw_decomposed')

    Returns:
        ImpedanceLearningConfig instance
    """
    env_config = config.get('environment', {})
    ll_config = config.get('ll_controller', {})
    rl_config = config.get('rl_training', {})

    # Base configuration
    env_cfg = ImpedanceLearningConfig(
        controller_type=controller_type,
        robot_mass=ll_config.get('robot', {}).get('mass', 1.0),
        robot_inertia=ll_config.get('robot', {}).get('inertia', 0.1),
        control_dt=env_config.get('control_dt', 0.01),
        policy_dt=env_config.get('policy_dt', 0.1),
        max_episode_steps=env_config.get('max_episode_steps', 1000)
    )

    # Controller-specific parameters
    if controller_type == 'se2_impedance':
        se2_cfg = ll_config.get('se2_impedance', {})
        env_cfg.min_damping_linear = se2_cfg.get('min_damping_linear', 1.0)
        env_cfg.max_damping_linear = se2_cfg.get('max_damping_linear', 50.0)
        env_cfg.min_damping_angular = se2_cfg.get('min_damping_angular', 0.5)
        env_cfg.max_damping_angular = se2_cfg.get('max_damping_angular', 20.0)
        env_cfg.min_stiffness_linear = se2_cfg.get('min_stiffness_linear', 10.0)
        env_cfg.max_stiffness_linear = se2_cfg.get('max_stiffness_linear', 200.0)
        env_cfg.min_stiffness_angular = se2_cfg.get('min_stiffness_angular', 5.0)
        env_cfg.max_stiffness_angular = se2_cfg.get('max_stiffness_angular', 100.0)

    elif controller_type == 'screw_decomposed':
        screw_cfg = ll_config.get('screw_decomposed', {})
        env_cfg.min_damping_parallel = screw_cfg.get('min_damping_parallel', 1.0)
        env_cfg.max_damping_parallel = screw_cfg.get('max_damping_parallel', 50.0)
        env_cfg.min_stiffness_parallel = screw_cfg.get('min_stiffness_parallel', 5.0)
        env_cfg.max_stiffness_parallel = screw_cfg.get('max_stiffness_parallel', 100.0)
        env_cfg.min_damping_perpendicular = screw_cfg.get('min_damping_perpendicular', 5.0)
        env_cfg.max_damping_perpendicular = screw_cfg.get('max_damping_perpendicular', 100.0)
        env_cfg.min_stiffness_perpendicular = screw_cfg.get('min_stiffness_perpendicular', 20.0)
        env_cfg.max_stiffness_perpendicular = screw_cfg.get('max_stiffness_perpendicular', 500.0)

    # Reward weights
    reward_cfg = rl_config.get('reward', {})
    env_cfg.tracking_weight = reward_cfg.get('tracking_weight', 1.0)
    env_cfg.wrench_weight = reward_cfg.get('wrench_weight', 0.1)
    env_cfg.smoothness_weight = reward_cfg.get('smoothness_weight', 0.01)

    return env_cfg


def create_env(
    hl_policy,
    env_config: ImpedanceLearningConfig,
    render_mode: Optional[str] = None
) -> ImpedanceLearningEnv:
    """Create impedance learning environment."""
    env = ImpedanceLearningEnv(
        config=env_config,
        hl_policy=hl_policy,
        render_mode=render_mode
    )
    return env


def train(
    config_path: str = 'scripts/configs/rl_config.yaml',
    hl_policy_type: Optional[str] = None,
    hl_checkpoint: Optional[str] = None,
    controller_type: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    device: str = 'auto',
    output_dir: Optional[str] = None,
    verbose: int = 1
):
    """
    Train low-level impedance learning policy.

    Args:
        config_path: Path to configuration file
        hl_policy_type: High-level policy type (overrides config)
        hl_checkpoint: Path to HL policy checkpoint (overrides config)
        controller_type: Controller type (overrides config)
        total_timesteps: Total training timesteps (overrides config)
        device: Device for computation
        output_dir: Output directory for checkpoints (overrides config)
        verbose: Verbosity level
    """
    print("=" * 80)
    print("Training Low-Level Impedance Learning Policy")
    print("=" * 80)

    # Load configuration
    config = load_config(config_path)
    print(f"✓ Loaded configuration from {config_path}")

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Override config with command-line arguments
    if hl_policy_type is None:
        hl_policy_type = config.get('hl_policy', {}).get('type', 'flow_matching')
    if hl_checkpoint is None:
        hl_checkpoint = config.get('hl_policy', {}).get('checkpoint', None)
    if controller_type is None:
        controller_type = config.get('ll_controller', {}).get('type', 'se2_impedance')
    if total_timesteps is None:
        total_timesteps = config.get('rl_training', {}).get('total_timesteps', 1000000)
    if output_dir is None:
        output_dir = config.get('rl_training', {}).get('output', {}).get('checkpoint_dir', './checkpoints')

    print(f"\nConfiguration:")
    print(f"  High-level policy: {hl_policy_type}")
    print(f"  HL checkpoint: {hl_checkpoint if hl_checkpoint else 'None (random init)'}")
    print(f"  Controller type: {controller_type}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Output directory: {output_dir}")

    # Load high-level policy
    hl_policy = load_hl_policy(hl_policy_type, hl_checkpoint, device)

    # Create environment configuration
    print(f"\nCreating environment configuration...")
    env_config = create_env_config(config, controller_type)
    print(f"✓ Environment configured:")
    print(f"  Controller type: {env_config.controller_type}")
    print(f"  Control dt: {env_config.control_dt} s")
    print(f"  Policy dt: {env_config.policy_dt} s")
    print(f"  Max episode steps: {env_config.max_episode_steps}")

    # Create training environment
    print(f"\nCreating training environment...")
    env = create_env(hl_policy, env_config, render_mode=None)
    print(f"✓ Training environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create evaluation environment
    print(f"Creating evaluation environment...")
    eval_env = create_env(hl_policy, env_config, render_mode=None)
    print(f"✓ Evaluation environment created")

    # Get PPO hyperparameters from config
    ppo_config = config.get('rl_training', {}).get('ppo', {})
    network_config = config.get('rl_training', {}).get('policy_network', {})
    logging_config = config.get('rl_training', {}).get('logging', {})
    eval_config = config.get('rl_training', {}).get('evaluation', {})

    # Create PPO policy
    print(f"\nCreating PPO policy...")
    ppo_policy = PPOImpedancePolicy(
        env=env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        features_dim=network_config.get('features_dim', 256),
        device=device,
        verbose=verbose,
        tensorboard_log=logging_config.get('tensorboard_log', './logs/impedance_rl/')
    )
    print(f"✓ PPO policy created")

    # Training
    print("\n" + "=" * 80)
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("=" * 80)

    ppo_policy.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=eval_config.get('eval_freq', 10000),
        n_eval_episodes=eval_config.get('n_eval_episodes', 5)
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # Evaluate final policy
    print(f"\nEvaluating final policy...")
    eval_results = ppo_policy.evaluate(
        n_episodes=10,
        deterministic=eval_config.get('deterministic', True)
    )
    print(f"✓ Final evaluation:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean episode length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")

    # Save policy
    output_name = config.get('rl_training', {}).get('output', {}).get('checkpoint_name', 'impedance_policy.zip')
    output_path = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving policy to {output_path}...")
    ppo_policy.save(output_path)
    print(f"✓ Policy saved!")

    # Save configuration alongside checkpoint
    config_save_path = os.path.join(output_dir, f"{output_name.replace('.zip', '_config.yaml')}")
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'hl_policy_type': hl_policy_type,
            'hl_checkpoint': hl_checkpoint,
            'controller_type': controller_type,
            'total_timesteps': total_timesteps,
            'final_eval_results': eval_results
        }, f)
    print(f"✓ Configuration saved to {config_save_path}")

    # Close environments
    env.close()
    eval_env.close()

    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Train RL policy for impedance parameter learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/configs/rl_config.yaml',
        help='Path to configuration file'
    )

    # High-level policy arguments
    parser.add_argument(
        '--hl_policy',
        type=str,
        default=None,
        choices=['flow_matching', 'diffusion', 'act', 'none'],
        help='Type of high-level policy (overrides config)'
    )
    parser.add_argument(
        '--hl_checkpoint',
        type=str,
        default=None,
        help='Path to high-level policy checkpoint (overrides config)'
    )

    # Controller type
    parser.add_argument(
        '--controller',
        type=str,
        default=None,
        choices=['se2_impedance', 'screw_decomposed'],
        help='Type of low-level controller (overrides config)'
    )

    # Training arguments
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=None,
        help='Total training timesteps (overrides config)'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for checkpoints (overrides config)'
    )

    # Device
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
        help='Verbosity level (0=none, 1=info, 2=debug)'
    )

    args = parser.parse_args()

    # Train
    train(
        config_path=args.config,
        hl_policy_type=args.hl_policy,
        hl_checkpoint=args.hl_checkpoint,
        controller_type=args.controller,
        total_timesteps=args.total_timesteps,
        device=args.device,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
