#!/usr/bin/env python3
"""
SWIVL Low-Level Policy Training Script (Layer 3)

Trains RL policy for learning optimal impedance modulation variables
in hierarchical bimanual manipulation control.

SWIVL Layer 3 Action Space:
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

All settings are loaded from rl_config.yaml (single source of truth).

Usage:
    # Train with default config
    python scripts/training/train_ll_policy.py
    
    # Train with custom config
    python scripts/training/train_ll_policy.py --config path/to/config.yaml
    
    # Override specific settings via command line
    python scripts/training/train_ll_policy.py --hl_policy flow_matching --total_timesteps 100000
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
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.rl_policy.impedance_learning_env import ImpedanceLearningEnv, ImpedanceLearningConfig
from src.rl_policy.ppo_impedance_policy import PPOImpedancePolicy


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_hl_policy(config: Dict[str, Any], device: str):
    """
    Load high-level policy from config.
    
    Args:
        config: Full configuration dictionary
        device: Device for computation
        
    Returns:
        Loaded policy or None
    """
    hl_cfg = config.get('hl_policy', {})
    policy_type = hl_cfg.get('type', 'none')
    checkpoint_path = hl_cfg.get('checkpoint', None)
    hl_config_path = hl_cfg.get('config_path', 'scripts/configs/hl_policy_config.yaml')
    
    if policy_type == 'none':
        print("Training without high-level policy (hold position mode)")
        return None

    print(f"Loading {policy_type} high-level policy...")

    try:
        # Load HL policy config
        if not os.path.isabs(hl_config_path):
            hl_config_path = os.path.join(_PROJECT_ROOT, hl_config_path)
        
        if os.path.exists(hl_config_path):
            with open(hl_config_path, 'r') as f:
                hl_config = yaml.safe_load(f)
        else:
            hl_config = {}
        
        # Import create_policy and LinearNormalizer from train_hl_policy
        from scripts.training.train_hl_policy import create_policy, LinearNormalizer
        
        # Create policy with proper config
        policy = create_policy(policy_type, hl_config, device)
        
        if checkpoint_path is not None:
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(_PROJECT_ROOT, checkpoint_path)
                
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                policy.load_state_dict(checkpoint['model_state_dict'])
                policy.eval()
                
                if 'normalizer' in checkpoint:
                    normalizer = LinearNormalizer()
                    normalizer.load_state_dict(checkpoint['normalizer'])
                    policy.normalizer = normalizer
                    print(f"✓ Loaded normalizer from checkpoint")
                
                print(f"✓ Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                print("  Using randomly initialized policy")
        else:
            print("⚠ No checkpoint provided, using randomly initialized policy")

        return policy

    except Exception as e:
        print(f"⚠ Could not load {policy_type} policy: {e}")
        import traceback
        traceback.print_exc()
        print("  Falling back to no high-level policy")
        return None


def create_ppo_policy(env, config: Dict[str, Any], device: str) -> PPOImpedancePolicy:
    """
    Create PPO policy from config.
    
    Args:
        env: Training environment
        config: Full configuration dictionary
        device: Device for computation
        
    Returns:
        PPOImpedancePolicy instance
    """
    rl_cfg = config.get('rl_training', {})
    ppo_cfg = rl_cfg.get('ppo', {})
    network_cfg = rl_cfg.get('network', {})
    logging_cfg = rl_cfg.get('logging', {})
    
    return PPOImpedancePolicy(
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


def train(config: Dict[str, Any], device: str):
    """
    Train SWIVL low-level impedance learning policy.
    
    Args:
        config: Full configuration dictionary
        device: Device for computation
    """
    print("=" * 80)
    print("SWIVL Layer 3: Impedance Modulation Policy Training")
    print("=" * 80)
    
    # Get settings from config
    rl_cfg = config.get('rl_training', {})
    output_cfg = rl_cfg.get('output', {})
    eval_cfg = rl_cfg.get('evaluation', {})
    
    total_timesteps = rl_cfg.get('total_timesteps', 500000)
    output_dir = output_cfg.get('checkpoint_dir', './checkpoints')
    checkpoint_name = output_cfg.get('checkpoint_name', 'impedance_policy.zip')
    
    # Load high-level policy
    hl_policy = load_hl_policy(config, device)
    
    # Create environment configuration from YAML
    print("\nCreating SWIVL environment...")
    env_config = ImpedanceLearningConfig.from_dict(config)
    
    print(f"✓ Environment configured:")
    print(f"  Controller type: {env_config.controller_type}")
    print(f"  Control dt: {env_config.control_dt} s ({int(1/env_config.control_dt)} Hz)")
    print(f"  Max episode steps: {env_config.max_episode_steps}")
    
    if env_config.controller_type == 'screw_decomposed':
        print(f"  Action space: 7D (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)")
        print(f"    d_∥ range: [{env_config.min_d_parallel}, {env_config.max_d_parallel}]")
        print(f"    d_⊥ range: [{env_config.min_d_perp}, {env_config.max_d_perp}]")
        print(f"    k_p range: [{env_config.min_k_p}, {env_config.max_k_p}]")
        print(f"    α range: [{env_config.min_alpha}, {env_config.max_alpha}]")
    
    print(f"\n  Reward weights:")
    print(f"    Tracking: {env_config.tracking_weight}")
    print(f"    Fighting force: {env_config.fighting_force_weight}")
    print(f"    Twist acceleration: {env_config.twist_accel_weight}")
    
    # Create training environment
    print(f"\nCreating training environment...")
    env = ImpedanceLearningEnv(config=env_config, hl_policy=hl_policy, render_mode=None)
    print(f"✓ Training environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    # Create evaluation environment
    print(f"Creating evaluation environment...")
    eval_env = ImpedanceLearningEnv(config=env_config, hl_policy=hl_policy, render_mode=None)
    print(f"✓ Evaluation environment created")
    
    # Create PPO policy
    print(f"\nCreating SWIVL PPO policy...")
    ppo_policy = create_ppo_policy(env, config, device)
    print(f"✓ PPO policy created")
    
    # Training
    print("\n" + "=" * 80)
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("=" * 80)
    
    ppo_policy.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=eval_cfg.get('eval_freq', 10000),
        n_eval_episodes=eval_cfg.get('n_eval_episodes', 5)
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Evaluate final policy
    print(f"\nEvaluating final policy...")
    eval_results = ppo_policy.evaluate(
        n_episodes=10,
        deterministic=eval_cfg.get('deterministic', True)
    )
    print(f"✓ Final evaluation:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean episode length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
    print(f"  Mean fighting force: {eval_results['mean_fighting_force']:.2f} ± {eval_results['std_fighting_force']:.2f}")
    
    # Save policy
    output_path = os.path.join(output_dir, checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving policy to {output_path}...")
    ppo_policy.save(output_path)
    print(f"✓ Policy saved!")
    
    # Save configuration alongside checkpoint
    config_save_path = output_path.replace('.zip', '_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'training_config': config,
            'final_eval_results': eval_results
        }, f, default_flow_style=False)
    print(f"✓ Configuration saved to {config_save_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Train SWIVL Layer 3: Impedance Modulation Policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file (single source of truth)
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/configs/rl_config.yaml',
        help='Path to configuration file (default: scripts/configs/rl_config.yaml)'
    )
    
    # Optional overrides (for convenience, but prefer editing config file)
    parser.add_argument(
        '--hl_policy',
        type=str,
        default=None,
        choices=['flow_matching', 'diffusion', 'act', 'none'],
        help='Override: High-level policy type'
    )
    parser.add_argument(
        '--hl_checkpoint',
        type=str,
        default=None,
        help='Override: Path to high-level policy checkpoint'
    )
    parser.add_argument(
        '--controller',
        type=str,
        default=None,
        choices=['se2_impedance', 'screw_decomposed'],
        help='Override: Low-level controller type'
    )
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=None,
        help='Override: Total training timesteps'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override: Output directory for checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['auto', 'cpu', 'cuda'],
        help='Override: Device for computation'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=None,
        help='Override: Verbosity level (0=none, 1=info, 2=debug)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")
    
    # Apply command-line overrides
    if args.hl_policy is not None:
        config.setdefault('hl_policy', {})['type'] = args.hl_policy
    if args.hl_checkpoint is not None:
        config.setdefault('hl_policy', {})['checkpoint'] = args.hl_checkpoint
    if args.controller is not None:
        config.setdefault('ll_controller', {})['type'] = args.controller
    if args.total_timesteps is not None:
        config.setdefault('rl_training', {})['total_timesteps'] = args.total_timesteps
    if args.output_dir is not None:
        config.setdefault('rl_training', {}).setdefault('output', {})['checkpoint_dir'] = args.output_dir
    if args.verbose is not None:
        config.setdefault('rl_training', {}).setdefault('logging', {})['verbose'] = args.verbose
    
    # Determine device
    device_cfg = config.get('device', {})
    device = args.device if args.device else device_cfg.get('type', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Print configuration summary
    hl_cfg = config.get('hl_policy', {})
    ll_cfg = config.get('ll_controller', {})
    rl_cfg = config.get('rl_training', {})
    
    print(f"\nConfiguration summary:")
    print(f"  High-level policy: {hl_cfg.get('type', 'none')}")
    print(f"  HL checkpoint: {hl_cfg.get('checkpoint', 'None')}")
    print(f"  Controller type: {ll_cfg.get('type', 'screw_decomposed')}")
    print(f"  Total timesteps: {rl_cfg.get('total_timesteps', 500000):,}")
    print(f"  Output dir: {rl_cfg.get('output', {}).get('checkpoint_dir', './checkpoints')}")
    
    # Train
    train(config, device)


if __name__ == '__main__':
    main()
