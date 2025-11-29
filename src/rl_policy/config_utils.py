"""
SWIVL RL Configuration Utilities

Provides utilities for loading and validating RL configuration from YAML files.
All RL training should use rl_config.yaml as the single source of truth.

Usage:
    from src.rl_policy.config_utils import load_rl_config, get_env_config
    
    config = load_rl_config('scripts/configs/rl_config.yaml')
    env_config = get_env_config(config)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_rl_config(config_path: str) -> Dict[str, Any]:
    """
    Load RL configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (relative or absolute)
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    # Resolve path
    if not os.path.isabs(config_path):
        config_path = PROJECT_ROOT / config_path
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_rl_config(config: Dict[str, Any]) -> bool:
    """
    Validate RL configuration has required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ['ll_controller', 'environment', 'rl_training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return True


def get_device(config: Dict[str, Any]) -> str:
    """
    Get compute device from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cpu' or 'cuda')
    """
    import torch
    
    device_cfg = config.get('device', {})
    device = device_cfg.get('type', 'auto')
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return device


def get_hl_policy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract high-level policy configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        HL policy configuration
    """
    return config.get('hl_policy', {
        'type': 'none',
        'checkpoint': None,
        'config_path': 'scripts/configs/hl_policy_config.yaml'
    })


def get_ll_controller_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract low-level controller configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        LL controller configuration
    """
    return config.get('ll_controller', {
        'type': 'screw_decomposed',
        'robot': {'mass': 1.2, 'inertia': 97.6}
    })


def get_rl_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract RL training configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        RL training configuration
    """
    return config.get('rl_training', {
        'total_timesteps': 500000,
        'ppo': {},
        'network': {},
        'reward': {},
        'evaluation': {},
        'logging': {},
        'output': {}
    })


def get_ppo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PPO hyperparameters from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        PPO hyperparameters
    """
    rl_cfg = config.get('rl_training', {})
    ppo_cfg = rl_cfg.get('ppo', {})
    
    # Return with defaults
    return {
        'learning_rate': ppo_cfg.get('learning_rate', 3e-4),
        'n_steps': ppo_cfg.get('n_steps', 2048),
        'batch_size': ppo_cfg.get('batch_size', 64),
        'n_epochs': ppo_cfg.get('n_epochs', 10),
        'gamma': ppo_cfg.get('gamma', 0.99),
        'gae_lambda': ppo_cfg.get('gae_lambda', 0.95),
        'clip_range': ppo_cfg.get('clip_range', 0.2),
        'ent_coef': ppo_cfg.get('ent_coef', 0.01),
        'vf_coef': ppo_cfg.get('vf_coef', 0.5),
        'max_grad_norm': ppo_cfg.get('max_grad_norm', 0.5),
    }


def get_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract network architecture configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Network configuration
    """
    rl_cfg = config.get('rl_training', {})
    network_cfg = rl_cfg.get('network', {})
    
    return {
        'features_dim': network_cfg.get('features_dim', 256),
        'policy_layers': network_cfg.get('policy_layers', [256, 128]),
        'value_layers': network_cfg.get('value_layers', [256, 128]),
        'activation': network_cfg.get('activation', 'relu'),
    }


def get_reward_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract reward weights from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Reward weights
    """
    rl_cfg = config.get('rl_training', {})
    reward_cfg = rl_cfg.get('reward', {})
    
    return {
        'tracking_weight': reward_cfg.get('tracking_weight', 1.0),
        'fighting_force_weight': reward_cfg.get('fighting_force_weight', 0.5),
        'twist_accel_weight': reward_cfg.get('twist_accel_weight', 0.01),
    }


def get_output_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract output/checkpoint configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Output configuration
    """
    rl_cfg = config.get('rl_training', {})
    output_cfg = rl_cfg.get('output', {})
    
    return {
        'checkpoint_dir': output_cfg.get('checkpoint_dir', './checkpoints'),
        'checkpoint_name': output_cfg.get('checkpoint_name', 'impedance_policy.zip'),
        'save_freq': output_cfg.get('save_freq', 50000),
    }


def print_config_summary(config: Dict[str, Any]):
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    hl_cfg = get_hl_policy_config(config)
    ll_cfg = get_ll_controller_config(config)
    rl_cfg = get_rl_training_config(config)
    reward_cfg = get_reward_config(config)
    output_cfg = get_output_config(config)
    
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"\nHigh-Level Policy:")
    print(f"  Type: {hl_cfg.get('type', 'none')}")
    print(f"  Checkpoint: {hl_cfg.get('checkpoint', 'None')}")
    
    print(f"\nLow-Level Controller:")
    print(f"  Type: {ll_cfg.get('type', 'screw_decomposed')}")
    print(f"  Robot mass: {ll_cfg.get('robot', {}).get('mass', 1.2)} kg")
    print(f"  Robot inertia: {ll_cfg.get('robot', {}).get('inertia', 97.6)} kg⋅m²")
    
    print(f"\nRL Training:")
    print(f"  Total timesteps: {rl_cfg.get('total_timesteps', 500000):,}")
    print(f"  Learning rate: {rl_cfg.get('ppo', {}).get('learning_rate', 3e-4)}")
    
    print(f"\nReward Weights:")
    print(f"  Tracking: {reward_cfg['tracking_weight']}")
    print(f"  Fighting force: {reward_cfg['fighting_force_weight']}")
    print(f"  Twist acceleration: {reward_cfg['twist_accel_weight']}")
    
    print(f"\nOutput:")
    print(f"  Directory: {output_cfg['checkpoint_dir']}")
    print(f"  Filename: {output_cfg['checkpoint_name']}")
    print("=" * 60)


# Convenience function for quick config loading
def load_default_config() -> Dict[str, Any]:
    """
    Load default rl_config.yaml from standard location.
    
    Returns:
        Configuration dictionary
    """
    return load_rl_config('scripts/configs/rl_config.yaml')


if __name__ == '__main__':
    # Test configuration loading
    print("Testing config_utils...")
    
    try:
        config = load_default_config()
        validate_rl_config(config)
        print_config_summary(config)
        print("\n✓ Configuration loading successful!")
    except FileNotFoundError as e:
        print(f"✗ Config file not found: {e}")
    except ValueError as e:
        print(f"✗ Invalid config: {e}")

