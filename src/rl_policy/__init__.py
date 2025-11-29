"""
SWIVL RL-based Impedance Parameter Learning (Layer 3)

This package provides RL policies that learn optimal impedance modulation
variables for compliant bimanual manipulation tasks.

The RL policy implements SWIVL Layer 3 (Impedance Variable Modulation):
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

The learned impedance parameters control:
- d_∥: Parallel damping (internal motion, compliant along joint axis)
- d_⊥: Perpendicular damping (bulk motion, stiff to maintain grasp)
- k_p: Pose error correction gain for reference twist field
- α: Characteristic length for metric tensor G

The RL policy works in conjunction with:
1. Layer 1: High-level policy generates desired poses (ACT, Diffusion, Flow Matching)
2. Layer 2: Reference twist field generator
3. Layer 3: RL policy determines optimal impedance parameters ← THIS MODULE
4. Layer 4: Screw-decomposed impedance controller

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

__all__ = [
    # Environment
    'ImpedanceLearningEnv',
    'ImpedanceLearningConfig',
    # Policy
    'PPOImpedancePolicy',
    'SWIVLFeatureExtractor',
    'SWIVLLoggingCallback',
]
