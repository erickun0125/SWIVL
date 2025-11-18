# High-Level Policy + RL Impedance Learning Pipeline

This document describes the hierarchical control pipeline combining high-level imitation learning policies with low-level RL-based impedance parameter learning.

## Overview

The pipeline consists of three main components:

```
High-Level Policy → Trajectory Generator → Impedance Controller (with RL-learned parameters)
```

### 1. High-Level Policies

High-level policies generate desired poses for the end-effectors at 10 Hz. Three types of policies are available:

#### Flow Matching Policy
- **Location**: `src/hl_planners/flow_matching.py`
- **Description**: Conditional flow matching for trajectory generation
- **Method**: Learns velocity fields for continuous normalizing flows
- **Advantages**: Fast inference, smooth trajectories

#### Diffusion Policy
- **Location**: `src/hl_planners/diffusion_policy.py`
- **Description**: Diffusion-based imitation learning
- **Method**: Denoising diffusion with DDIM sampling
- **Advantages**: Strong performance on complex tasks, action chunking
- **Reference**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (2023)

#### ACT (Action Chunking with Transformers)
- **Location**: `src/hl_planners/act.py`
- **Description**: Transformer-based imitation learning with CVAE
- **Method**: CVAE + Transformer encoder-decoder
- **Advantages**: Temporal consistency, excellent for bimanual tasks
- **Reference**: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (2023)

### 2. Trajectory Generator

- **Location**: `src/trajectory_generator.py`
- **Description**: Converts desired poses into smooth, continuous trajectories
- **Method**: Cubic spline interpolation or minimum jerk trajectories
- **Output**: Desired pose and twist (velocity) at each timestep

### 3. RL Impedance Learning

#### Overview
The RL policy learns optimal impedance parameters (damping D and stiffness K) for the impedance controller. The inertia matrix M is always set to the task space inertia.

#### Environment
- **Location**: `src/rl_policy/impedance_learning_env.py`
- **Type**: Gymnasium environment

**State Space (30 dimensions)**:
- External wrench (6): [fx_0, fy_0, tau_0, fx_1, fy_1, tau_1]
- Current pose (6): [x_0, y_0, theta_0, x_1, y_1, theta_1]
- Current twist (6): [vx_0, vy_0, omega_0, vx_1, vy_1, omega_1]
- Desired pose (6): [x_d0, y_d0, theta_d0, x_d1, y_d1, theta_d1]
- Desired twist (6): [vx_d0, vy_d0, omega_d0, vx_d1, vy_d1, omega_d1]

**Action Space (12 dimensions)**:
- Arm 0: [D_linear_x, D_linear_y, D_angular, K_linear_x, K_linear_y, K_angular]
- Arm 1: [D_linear_x, D_linear_y, D_angular, K_linear_x, K_linear_y, K_angular]

Actions are normalized to [-1, 1] and scaled to safe ranges.

**Reward Function**:
```python
reward = - tracking_weight * tracking_error
         - wrench_weight * wrench_magnitude
         - smoothness_weight * parameter_change
```

#### PPO Policy
- **Location**: `src/rl_policy/ppo_impedance_policy.py`
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Implementation**: Stable-Baselines3
- **Features**:
  - Custom feature extractor for impedance learning
  - Separate encoders for wrenches, poses, and twists
  - Tensorboard logging of impedance statistics

## Usage

### 1. Training High-Level Policy

First, train a high-level policy using your preferred method (Flow Matching, Diffusion, or ACT).

```python
from src.hl_planners import FlowMatchingPolicy, DiffusionPolicy, ACTPolicy

# Example: Flow Matching
hl_policy = FlowMatchingPolicy(device='cuda')
# ... training code ...
hl_policy.save('checkpoints/flow_matching.pth')
```

### 2. Training RL Impedance Policy

Once you have a trained high-level policy, train the RL impedance policy:

```bash
# With Flow Matching policy
python -m src.rl_policy.train_impedance_policy \
    --hl_policy flow_matching \
    --hl_policy_path checkpoints/flow_matching.pth \
    --total_timesteps 1000000 \
    --output_path checkpoints/impedance_policy.zip

# With Diffusion Policy
python -m src.rl_policy.train_impedance_policy \
    --hl_policy diffusion \
    --hl_policy_path checkpoints/diffusion.pth \
    --total_timesteps 1000000 \
    --output_path checkpoints/impedance_diffusion.zip

# With ACT
python -m src.rl_policy.train_impedance_policy \
    --hl_policy act \
    --hl_policy_path checkpoints/act.pth \
    --total_timesteps 1000000 \
    --output_path checkpoints/impedance_act.zip
```

**Training Arguments**:
- `--hl_policy`: Type of high-level policy ('flow_matching', 'diffusion', 'act', 'none')
- `--hl_policy_path`: Path to pre-trained high-level policy checkpoint
- `--total_timesteps`: Total training timesteps (default: 1,000,000)
- `--learning_rate`: PPO learning rate (default: 3e-4)
- `--n_steps`: Steps per PPO update (default: 2048)
- `--batch_size`: Batch size (default: 64)
- `--output_path`: Where to save trained policy
- `--tensorboard_log`: Path for tensorboard logs
- `--device`: Device for computation ('auto', 'cpu', 'cuda')

### 3. Using the Complete Pipeline

```python
from src.hl_planners import DiffusionPolicy
from src.rl_policy import ImpedanceLearningEnv, PPOImpedancePolicy
from src.envs.biart import BiartEnv
from src.trajectory_generator import MinimumJerkTrajectory
from src.ll_controllers.task_space_impedance import TaskSpaceImpedanceController

# 1. Load high-level policy
hl_policy = DiffusionPolicy(device='cuda')
hl_policy.load('checkpoints/diffusion.pth')

# 2. Load RL impedance policy
env = ImpedanceLearningEnv(hl_policy=hl_policy)
rl_policy = PPOImpedancePolicy.load_from_file('checkpoints/impedance_policy.zip', env)

# 3. Run inference
obs, _ = env.reset()
hl_policy.reset()

for step in range(1000):
    # Get RL observation and predict impedance parameters
    rl_obs = env._get_rl_observation(obs)
    action = rl_policy.predict(rl_obs, deterministic=True)

    # Step environment (RL policy controls impedance parameters)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
        hl_policy.reset()
```

## Pipeline Details

### Control Flow

1. **High-Level Policy** (10 Hz):
   - Input: Current state (EE poses, link poses, external wrenches)
   - Output: Desired poses for both end-effectors

2. **Trajectory Generator** (100 Hz):
   - Input: Desired poses from high-level policy
   - Output: Smooth trajectory with desired pose and twist at each timestep

3. **RL Impedance Policy** (10 Hz):
   - Input: External wrench, current state, desired state
   - Output: Impedance parameters (D, K) for both arms

4. **Impedance Controller** (100 Hz):
   - Input: Current pose, desired pose, desired twist, impedance parameters
   - Output: Wrench command

### Key Features

- **Bimanual Control**: Both arms are controlled independently by the impedance controller, but the RL policy learns impedance parameters for both arms jointly
- **Hierarchical Learning**: High-level policy learns task-level behavior, RL policy learns low-level compliance
- **Adaptive Compliance**: Impedance parameters adapt based on force feedback and task requirements
- **Temporal Consistency**: Action chunking in Diffusion and ACT policies ensures smooth trajectories

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     High-Level Policy                        │
│              (Flow Matching / Diffusion / ACT)               │
│                                                              │
│  Input: Current State                                        │
│  Output: Desired Poses (10 Hz)                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Trajectory Generator                        │
│                 (Cubic Spline / Min Jerk)                    │
│                                                              │
│  Input: Desired Poses                                        │
│  Output: Continuous Trajectory (pose + twist, 100 Hz)       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │  ┌──────────────────────────────────┐
                       │  │   RL Impedance Policy (PPO)      │
                       │  │                                  │
                       │  │  Input: State + Desired State    │
                       │  │  Output: D, K (10 Hz)            │
                       │  └───────────┬──────────────────────┘
                       │              │
                       ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Impedance Controller                        │
│                   (Task Space, 100 Hz)                       │
│                                                              │
│  Input: Current Pose, Desired Pose/Twist, D, K              │
│  Output: Wrench Command                                      │
│                                                              │
│  Control Law: F = K(x_d - x) + D(ẋ_d - ẋ)                  │
│  Note: M (inertia) = Task Space Inertia (fixed)             │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Bimanual Robot                           │
│                      (SE(2) Dynamics)                        │
└─────────────────────────────────────────────────────────────┘
```

## Benefits of This Approach

1. **Modularity**: Each component can be trained and tested independently
2. **Transferability**: High-level policies can be reused with different impedance strategies
3. **Sample Efficiency**: RL only learns low-dimensional impedance parameters
4. **Interpretability**: Impedance parameters have clear physical meaning
5. **Safety**: Impedance bounds ensure safe parameter ranges

## Next Steps

1. **Collect Demonstrations**: Gather expert demonstrations for high-level policy training
2. **Train High-Level Policy**: Train Flow Matching, Diffusion, or ACT policy
3. **Train RL Policy**: Use trained high-level policy to train impedance parameters
4. **Fine-tune**: Optionally fine-tune the complete pipeline end-to-end
5. **Deploy**: Use the trained pipeline for bimanual manipulation tasks

## References

1. Chi, C., et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
2. Zhao, T. Z., et al. (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
3. Lipson, H., et al. (2022). "Conditional Flow Matching for Trajectory Generation"
4. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"

## Troubleshooting

### High-level policy not generating good trajectories
- Check if the policy is properly loaded
- Verify observation preprocessing
- Ensure sufficient training data/timesteps

### RL policy not learning
- Reduce reward weights if learning is unstable
- Adjust PPO hyperparameters (learning rate, batch size)
- Check if impedance bounds are reasonable
- Verify environment observations are correct

### Unstable control
- Reduce maximum impedance parameters
- Increase smoothness penalty in reward
- Check if trajectory generator is producing smooth trajectories

## Contact

For questions or issues, please open an issue on GitHub.
