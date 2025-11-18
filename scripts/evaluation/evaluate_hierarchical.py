"""
Hierarchical Policy Evaluation Script

Evaluates the full hierarchical control pipeline:
    High-Level Policy → Trajectory Generation → Low-Level Impedance Controller

Supports all HL policy types (Flow Matching, Diffusion, ACT) and
both controller types (SE(2) impedance, screw-decomposed impedance).

Usage:
    python scripts/evaluation/evaluate_hierarchical.py --config configs/rl_config.yaml
    python scripts/evaluation/evaluate_hierarchical.py --hl_checkpoint checkpoints/flow_matching_best.pth
    python scripts/evaluation/evaluate_hierarchical.py --ll_checkpoint checkpoints/impedance_policy.zip

Example:
    # Evaluate with trained policies
    python scripts/evaluation/evaluate_hierarchical.py \
        --hl_checkpoint checkpoints/flow_matching_best.pth \
        --ll_checkpoint checkpoints/impedance_policy.zip

    # Evaluate with custom config
    python scripts/evaluation/evaluate_hierarchical.py --config my_config.yaml

    # Save trajectories and videos
    python scripts/evaluation/evaluate_hierarchical.py \
        --save_trajectories --save_videos --visualize
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import numpy as np
import torch
import yaml
from tqdm import tqdm

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


def load_hl_policy(policy_type: str, checkpoint_path: str, device: str):
    """Load high-level policy from checkpoint."""
    print(f"Loading {policy_type} high-level policy...")

    if policy_type == 'flow_matching':
        policy = FlowMatchingPolicy(device=device)
    elif policy_type == 'diffusion':
        policy = DiffusionPolicy(device=device)
    elif policy_type == 'act':
        policy = ACTPolicy(device=device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        policy.load(checkpoint_path)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No valid checkpoint provided, using randomly initialized policy")

    return policy


def create_env_config(config: Dict[str, Any], controller_type: str) -> ImpedanceLearningConfig:
    """Create environment configuration from YAML config."""
    env_config = config.get('environment', {})
    ll_config = config.get('ll_controller', {})
    eval_config = config.get('evaluation', {})

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

    return env_cfg


def evaluate_episode(
    env: ImpedanceLearningEnv,
    ll_policy: PPOImpedancePolicy,
    deterministic: bool = True,
    save_trajectory: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single episode.

    Returns:
        Dictionary with episode statistics
    """
    obs, _ = env.reset()
    done = False
    truncated = False

    episode_reward = 0.0
    episode_length = 0
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'ee_poses': [],
        'external_wrenches': []
    }

    while not (done or truncated):
        # Get action from RL policy
        action, _ = ll_policy.predict(obs, deterministic=deterministic)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Record trajectory
        if save_trajectory:
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)

            # Get environment observation
            env_obs = env.base_env.get_obs()
            trajectory['ee_poses'].append(env_obs['ee_poses'].copy())
            trajectory['external_wrenches'].append(env_obs['external_wrenches'].copy())

        episode_reward += reward
        episode_length += 1
        obs = next_obs

    results = {
        'reward': episode_reward,
        'length': episode_length,
        'success': done and not truncated
    }

    if save_trajectory:
        results['trajectory'] = trajectory

    return results


def compute_metrics(episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute evaluation metrics from episode results."""
    rewards = [ep['reward'] for ep in episode_results]
    lengths = [ep['length'] for ep in episode_results]
    successes = [ep['success'] for ep in episode_results]

    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': np.mean(successes) * 100.0,
        'num_episodes': len(episode_results)
    }

    return metrics


def save_results(
    metrics: Dict[str, Any],
    episode_results: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: str
):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # Save episode results
    results_path = os.path.join(output_dir, 'episode_results.json')
    with open(results_path, 'w') as f:
        # Remove trajectory data for JSON serialization
        results_to_save = []
        for ep in episode_results:
            ep_data = {k: v for k, v in ep.items() if k != 'trajectory'}
            results_to_save.append(ep_data)
        json.dump(results_to_save, f, indent=2)
    print(f"✓ Episode results saved to {results_path}")

    # Save trajectories if available
    trajectories = [ep['trajectory'] for ep in episode_results if 'trajectory' in ep]
    if trajectories:
        traj_path = os.path.join(output_dir, 'trajectories.npz')
        np.savez(traj_path, trajectories=trajectories)
        print(f"✓ Trajectories saved to {traj_path}")

    # Save configuration
    config_path = os.path.join(output_dir, 'eval_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✓ Configuration saved to {config_path}")


def evaluate(
    config_path: str = 'scripts/configs/rl_config.yaml',
    hl_policy_type: Optional[str] = None,
    hl_checkpoint: Optional[str] = None,
    ll_checkpoint: Optional[str] = None,
    controller_type: Optional[str] = None,
    num_episodes: Optional[int] = None,
    deterministic: bool = True,
    save_trajectories: bool = False,
    save_metrics: bool = True,
    visualize: bool = False,
    device: str = 'auto',
    output_dir: Optional[str] = None
):
    """
    Evaluate hierarchical control pipeline.

    Args:
        config_path: Path to configuration file
        hl_policy_type: High-level policy type (overrides config)
        hl_checkpoint: Path to HL policy checkpoint (overrides config)
        ll_checkpoint: Path to LL policy checkpoint (required)
        controller_type: Controller type (overrides config)
        num_episodes: Number of evaluation episodes (overrides config)
        deterministic: Use deterministic policy
        save_trajectories: Save trajectory data
        save_metrics: Save metrics
        visualize: Enable visualization
        device: Device for computation
        output_dir: Output directory (overrides config)
    """
    print("=" * 80)
    print("Hierarchical Policy Evaluation")
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
    if num_episodes is None:
        num_episodes = config.get('evaluation', {}).get('num_episodes', 50)
    if output_dir is None:
        output_dir = config.get('evaluation', {}).get('output_dir', './evaluation_results')

    # Check LL checkpoint
    if ll_checkpoint is None:
        raise ValueError("Low-level policy checkpoint is required! Use --ll_checkpoint")
    if not os.path.exists(ll_checkpoint):
        raise FileNotFoundError(f"LL checkpoint not found: {ll_checkpoint}")

    print(f"\nConfiguration:")
    print(f"  High-level policy: {hl_policy_type}")
    print(f"  HL checkpoint: {hl_checkpoint if hl_checkpoint else 'None (random init)'}")
    print(f"  LL checkpoint: {ll_checkpoint}")
    print(f"  Controller type: {controller_type}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Output directory: {output_dir}")

    # Load high-level policy
    hl_policy = load_hl_policy(hl_policy_type, hl_checkpoint, device)

    # Create environment configuration
    print(f"\nCreating environment...")
    env_config = create_env_config(config, controller_type)
    env = ImpedanceLearningEnv(
        config=env_config,
        hl_policy=hl_policy,
        render_mode='rgb_array' if visualize else None
    )
    print(f"✓ Environment created")

    # Load low-level policy
    print(f"\nLoading low-level policy from {ll_checkpoint}...")
    ll_policy = PPOImpedancePolicy.load(ll_checkpoint, env=env)
    print(f"✓ Low-level policy loaded")

    # Evaluate
    print("\n" + "=" * 80)
    print(f"Starting evaluation for {num_episodes} episodes...")
    print("=" * 80)

    episode_results = []
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        results = evaluate_episode(
            env, ll_policy,
            deterministic=deterministic,
            save_trajectory=save_trajectories
        )
        episode_results.append(results)

        if (episode + 1) % 10 == 0:
            temp_metrics = compute_metrics(episode_results)
            print(f"\nProgress [{episode+1}/{num_episodes}]:")
            print(f"  Mean reward: {temp_metrics['mean_reward']:.2f} ± {temp_metrics['std_reward']:.2f}")
            print(f"  Success rate: {temp_metrics['success_rate']:.1f}%")

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)

    # Compute final metrics
    metrics = compute_metrics(episode_results)
    print(f"\nFinal Results:")
    print(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Reward range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print(f"  Mean episode length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"  Success rate: {metrics['success_rate']:.1f}%")

    # Save results
    if save_metrics or save_trajectories:
        print(f"\nSaving results to {output_dir}...")
        save_results(metrics, episode_results, config, output_dir)

    # Close environment
    env.close()

    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hierarchical control pipeline (HL + LL)',
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
        choices=['flow_matching', 'diffusion', 'act'],
        help='Type of high-level policy (overrides config)'
    )
    parser.add_argument(
        '--hl_checkpoint',
        type=str,
        default=None,
        help='Path to high-level policy checkpoint (overrides config)'
    )

    # Low-level policy checkpoint (required)
    parser.add_argument(
        '--ll_checkpoint',
        type=str,
        required=True,
        help='Path to low-level policy checkpoint (REQUIRED)'
    )

    # Controller type
    parser.add_argument(
        '--controller',
        type=str,
        default=None,
        choices=['se2_impedance', 'screw_decomposed'],
        help='Type of low-level controller (overrides config)'
    )

    # Evaluation arguments
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=None,
        help='Number of evaluation episodes (overrides config)'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic policy'
    )

    # Output arguments
    parser.add_argument(
        '--save_trajectories',
        action='store_true',
        help='Save trajectory data'
    )
    parser.add_argument(
        '--save_metrics',
        action='store_true',
        default=True,
        help='Save metrics'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for computation'
    )

    args = parser.parse_args()

    # Evaluate
    evaluate(
        config_path=args.config,
        hl_policy_type=args.hl_policy,
        hl_checkpoint=args.hl_checkpoint,
        ll_checkpoint=args.ll_checkpoint,
        controller_type=args.controller,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        save_trajectories=args.save_trajectories,
        save_metrics=args.save_metrics,
        visualize=args.visualize,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
