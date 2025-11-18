"""
RL-based Impedance Parameter Learning

This package provides RL policies that learn optimal impedance parameters
for compliant manipulation tasks. The RL policy learns to adapt the stiffness
and damping parameters of the impedance controller based on:
- External force/torque feedback
- Tracking error
- Desired trajectory information

The RL policy works in conjunction with a pre-trained high-level policy:
1. High-level policy generates desired poses
2. Trajectory generator creates smooth trajectories
3. RL policy determines optimal impedance parameters (D, K)
4. Impedance controller tracks the trajectory with learned compliance

Algorithm: PPO (Proximal Policy Optimization) from Stable-Baselines3
"""

from src.rl_policy.impedance_learning_env import ImpedanceLearningEnv
from src.rl_policy.ppo_impedance_policy import PPOImpedancePolicy

__all__ = [
    'ImpedanceLearningEnv',
    'PPOImpedancePolicy',
]
