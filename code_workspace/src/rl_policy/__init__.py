"""
SWIVL RL-based Impedance Parameter Learning (Layer 3)

This package provides RL policies that learn optimal impedance modulation
variables for compliant bimanual manipulation tasks.

The RL policy implements SWIVL Layer 3 (Impedance Variable Modulation):
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

Configuration:
    All settings should be managed in scripts/configs/rl_config.yaml
    Use ImpedanceLearningConfig.from_dict(config) to create env config
    Use PPOImpedancePolicy.from_config(env, config) to create policy

Training:
    python scripts/training/train_ll_policy.py --config scripts/configs/rl_config.yaml

Algorithm: PPO (Proximal Policy Optimization) from Stable-Baselines3
"""

from src.rl_policy.impedance_learning_env import (
    ImpedanceLearningEnv,
    ImpedanceLearningConfig
)
from src.rl_policy.ppo_impedance_policy import (
    PPOImpedancePolicy,
    SWIVLFeatureExtractor,
    SWIVLLoggingCallback
)
from src.rl_policy.config_utils import (
    load_rl_config,
    validate_rl_config,
    get_device,
    get_ppo_config,
    get_network_config,
    get_reward_config,
    get_output_config,
    print_config_summary,
    load_default_config
)

__all__ = [
    # Environment
    'ImpedanceLearningEnv',
    'ImpedanceLearningConfig',
    # Policy
    'PPOImpedancePolicy',
    'SWIVLFeatureExtractor',
    'SWIVLLoggingCallback',
    # Config utilities
    'load_rl_config',
    'validate_rl_config',
    'get_device',
    'get_ppo_config',
    'get_network_config',
    'get_reward_config',
    'get_output_config',
    'print_config_summary',
    'load_default_config',
]
