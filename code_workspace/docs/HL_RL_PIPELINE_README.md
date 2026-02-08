# SWIVL Hierarchical Control Pipeline

This document describes the SWIVL (Screw-Wrench Informed Impedance Variable Learning) hierarchical control pipeline for bimanual manipulation of articulated objects.

## Overview

SWIVL is a four-layer hierarchical control framework:

```
Layer 1: High-Level Policy → desired poses (10 Hz)
Layer 2: Reference Twist Field Generator → reference twists (100 Hz)
Layer 3: Impedance Modulation Policy (RL) → impedance variables (100 Hz)
Layer 4: Screw-Decomposed Impedance Controller → control wrenches (100 Hz)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: HIGH-LEVEL POLICY                                 │
│  (FlowMatching / Diffusion / ACT / Teleoperation)          │
│  Input: Image + Proprioception                              │
│  Output: Sparse waypoints {T_sd[τ]} at 10 Hz               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: REFERENCE TWIST FIELD GENERATOR                   │
│  1. SE(3) Trajectory Smoothing (SLERP + Cubic Spline)       │
│  2. Body Twist Computation                                   │
│  3. Stable Imitation Vector Field:                          │
│     V_ref = Ad_{T_bd} V_des + k_p * E                       │
│  Output: Dense reference twists at 100 Hz                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: IMPEDANCE MODULATION POLICY (PPO)                 │
│  Observation (30D):                                         │
│    [V_ref(6), B(6), F(6), T(6), V(6)]                      │
│  Action (7D):                                               │
│    (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)          │
│  Output: Impedance variables at 100 Hz                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: SCREW-DECOMPOSED IMPEDANCE CONTROLLER             │
│  Damping Matrix: K_d = G(P_∥ d_∥ + P_⊥ d_⊥)                │
│  Control Law: F_cmd = K_d(V_ref - V) + μ_b                 │
│  Output: Control wrenches at 100 Hz                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PHYSICS ENGINE (BiArtEnv + Pymunk)                         │
│  Output: Next state observation                             │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: High-Level Policies

High-level policies generate desired poses for end-effectors at 10 Hz.

### Available Policies

| Policy | Location | Description |
|--------|----------|-------------|
| **Flow Matching** | `src/hl_planners/flow_matching.py` | Continuous normalizing flows, fast inference |
| **Diffusion** | `src/hl_planners/diffusion_policy.py` | Diffusion-based with DDIM sampling |
| **ACT** | `src/hl_planners/act.py` | Transformer-based with CVAE, action chunking |
| **Teleoperation** | `src/hl_planners/teleoperation.py` | Keyboard-based control |

### Training HL Policy

```bash
cd code_workspace

# Train Flow Matching
python scripts/training/train_hl_policy.py --policy flow_matching

# Train ACT
python scripts/training/train_hl_policy.py --policy act

# Train Diffusion
python scripts/training/train_hl_policy.py --policy diffusion
```

## Layer 2: Reference Twist Field Generator

Transforms sparse waypoints into dense, stable reference twists.

### Key Equations

**Body Twist Computation:**
```
V_des = [ω_des, v_des]
[ω_des] = R^T Ṙ
v_des = R^T ṗ
```

**Stable Imitation Vector Field:**
```
V_ref = Ad_{T_bd} V_des + k_p * E
```

Where:
- `Ad_{T_bd}`: Adjoint transformation from desired to body frame
- `E = [α * e_R, e_p]`: Weighted pose error
- `k_p`: Learned pose error correction gain

## Layer 3: Impedance Modulation Policy (PPO)

RL policy that learns to modulate impedance variables based on wrench feedback and object geometry.

### Observation Space (30D)

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Reference twists | 6 | V_l^ref, V_r^ref ∈ R³ × 2 |
| Screw axes | 6 | B_l, B_r ∈ R³ × 2 (object geometry) |
| Wrenches | 6 | F_l, F_r ∈ R³ × 2 (F/T sensor feedback) |
| EE poses | 6 | (x, y, θ) × 2 |
| EE body twists | 6 | (ω, vx, vy) × 2 |

### Action Space (7D)

| Parameter | Description | Range |
|-----------|-------------|-------|
| `d_l_∥, d_r_∥` | Damping for internal motion (per-arm) | [1, 50] |
| `d_l_⊥, d_r_⊥` | Damping for bulk motion (per-arm) | [1, 50] |
| `k_p_l, k_p_r` | Pose correction gains (per-arm) | [0.1, 10] |
| `α` | Characteristic length (metric tensor) | [1, 20] |

### Reward Function

```
r_t = r_track + r_safety + r_reg + r_term
```

| Component | Formula | Description |
|-----------|---------|-------------|
| `r_track` | `-w_track * Σ\|\|V - V_ref\|\|²_G` | G-metric tracking error |
| `r_safety` | `w_safety * exp(-κ * Σ\|\|F_⊥\|\|²_{G⁻¹})` | Exponential safety (alive bonus) |
| `r_reg` | `-w_reg * Σ\|\|V̇\|\|²` | Twist acceleration penalty |
| `r_term` | `-w_term` (on failure) | Termination penalty |

**Key Design Choices:**
- Exponential safety reward provides positive "alive bonus" when fighting forces are low
- Dual metric G⁻¹ for wrenches ensures dimensional consistency: `||F||²_{G⁻¹} = τ²/α² + fx² + fy²`
- Termination penalty prevents learning to intentionally fail

### Termination Conditions

1. **Grasp Drift**: Geodesic distance exceeds threshold (50 px)
2. **Wrench Limit**: External wrench magnitude exceeds limit (200 N)

### Training LL Policy

```bash
cd code_workspace

# Train with HL policy
python scripts/training/train_ll_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --total_timesteps 500000

# Train without HL policy (hold position mode)
python scripts/training/train_ll_policy.py --hl_policy none

# Override settings via command line
python scripts/training/train_ll_policy.py \
    --config scripts/configs/rl_config.yaml \
    --hl_policy flow_matching \
    --total_timesteps 100000
```

## Layer 4: Screw-Decomposed Impedance Controller

Executes compliant control with independent regulation of bulk and internal motions.

### Motion Space Decomposition

Using metric tensor `G = diag(α², 1, 1)`:

**Projection Operators:**
```
P_∥ = B (B^T G B)^{-1} B^T G    (internal motion)
P_⊥ = I - P_∥                    (bulk motion)
```

**Damping Matrix:**
```
K_d = G(P_∥ d_∥ + P_⊥ d_⊥)
```

### Control Law

```
F_cmd = K_d (V_ref - V) + μ_b
```

Where:
- `V_ref = Ad_{T_bd} V_des + k_p * E`: Reference twist from Layer 2
- `μ_b`: Coriolis/centrifugal compensation

## Usage

### Complete Pipeline Evaluation

```bash
cd code_workspace

# Interactive mode with visualization
python scripts/evaluation/eval_hierarchical_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --no_wrench

# Batch evaluation
python scripts/evaluation/eval_hierarchical_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --num_episodes 100 \
    --no_wrench \
    --output_path evaluation_results/results.json
```

### Programmatic Usage

```python
from src.envs.biart import BiArtEnv
from src.rl_policy.impedance_learning_env import ImpedanceLearningEnv
from stable_baselines3 import PPO

# Load environment
env = ImpedanceLearningEnv(config=config, hl_policy=hl_policy)

# Load trained LL policy
ll_policy = PPO.load("checkpoints/impedance_policy.zip")

# Run inference
obs, _ = env.reset()
for step in range(1000):
    action, _ = ll_policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Access impedance parameters
    print(f"α: {info['current_alpha']:.2f}")
```

## Configuration

All settings are in `scripts/configs/rl_config.yaml`:

```yaml
ll_controller:
  type: "screw_decomposed"
  screw_decomposed:
    min_d_parallel: 1.0
    max_d_parallel: 50.0
    min_alpha: 1.0
    max_alpha: 20.0

rl_training:
  total_timesteps: 500000
  reward:
    tracking_weight: 0.0001
    safety_reward_weight: 1.0
    safety_exp_scale: 0.01
    termination_penalty: 10.0

environment:
  max_grasp_drift: 50.0
  max_external_wrench: 200.0
```

## Key Files

| File | Description |
|------|-------------|
| `src/rl_policy/impedance_learning_env.py` | Layer 3 Gym environment |
| `src/rl_policy/ppo_impedance_policy.py` | PPO policy with FiLM conditioning |
| `src/ll_controllers/se2_screw_decomposed_impedance.py` | Layer 4 controller |
| `scripts/training/train_ll_policy.py` | Training script |
| `scripts/evaluation/eval_hierarchical_policy.py` | Evaluation script |
| `scripts/configs/rl_config.yaml` | Configuration file |

## Troubleshooting

### RL policy not learning
- Check reward weights in `rl_config.yaml`
- Verify HL policy is generating reasonable trajectories
- Monitor fighting force in TensorBoard

### Unstable control
- Reduce `max_d_parallel` and `max_d_perp`
- Increase `termination_penalty`
- Check screw axes are computed correctly

### Grasp failures
- Lower `max_grasp_drift` threshold
- Increase `safety_reward_weight`
- Verify initial grasp is stable

## References

1. Lynch & Park - Modern Robotics (twist/wrench conventions)
2. Chi et al. - Diffusion Policy (2023)
3. Zhao et al. - ACT / ALOHA (2023)
4. Schulman et al. - PPO (2017)
