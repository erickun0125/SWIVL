# SWIVL Technical Documentation

Technical documentation for the SWIVL project (Screw-Wrench Informed Impedance Variable Learning).

## Overview

SWIVL is a four-layer hierarchical control framework for bimanual manipulation of articulated objects:

| Layer | Component | Input | Output |
|-------|-----------|-------|--------|
| **Layer 1** | High-Level Policy | Image + Proprio | Sparse waypoints (10 Hz) |
| **Layer 2** | Reference Twist Field Generator | Waypoints + Current pose | Dense reference twists (100 Hz) |
| **Layer 3** | Impedance Modulation Policy (RL) | Obs (30D) | Impedance variables (7D) |
| **Layer 4** | Screw-Decomposed Controller | Reference + Impedance | Control wrenches |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: HIGH-LEVEL POLICY                                 │
│  (FlowMatching / Diffusion / ACT / Teleoperation)          │
│  Output: Desired poses {T_sd}[τ] at 10 Hz                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: REFERENCE TWIST FIELD GENERATOR                   │
│  V_ref = Ad_{T_bd} V_des + k_p * E                         │
│  Output: Reference twists V_ref at 100 Hz                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: IMPEDANCE MODULATION POLICY (PPO)                 │
│  Observation: [V_ref, B, F, T, V] ∈ R^30                   │
│  Action: (d_∥, d_⊥, k_p, α) ∈ R^7                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: SCREW-DECOMPOSED IMPEDANCE CONTROLLER             │
│  F_cmd = K_d (V_ref - V) + μ_b                             │
│  K_d = G(P_∥ d_∥ + P_⊥ d_⊥)                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHYSICS ENGINE (BiArtEnv + Pymunk)                         │
│  Output: Observation dict                                   │
└─────────────────────────────────────────────────────────────┘
```

## Modern Robotics Conventions

This codebase follows **Modern Robotics (Lynch & Park)** conventions:

| Quantity | Format | Units |
|----------|--------|-------|
| Twist | `V = [ω, vx, vy]ᵀ` | [rad/s, px/s, px/s] |
| Wrench | `F = [τ, fx, fy]ᵀ` | [N⋅px, N, N] |
| Screw axis | `S = [sω, sx, sy]ᵀ` | normalized |
| Metric tensor | `G = diag(α², 1, 1)` | for twists |
| Dual metric | `G⁻¹ = diag(1/α², 1, 1)` | for wrenches |

## Key Source Files

### Core Modules

| File | Description |
|------|-------------|
| `src/se2_math.py` | SE(2) Lie group operations (exp, log, adjoint) |
| `src/se2_dynamics.py` | Robot dynamics (mass matrix, Coriolis) |
| `src/trajectory_generator.py` | CubicSpline, MinimumJerk trajectories |

### Controllers (Layer 4)

| File | Description |
|------|-------------|
| `src/ll_controllers/se2_impedance_controller.py` | Classical SE(2) impedance control |
| `src/ll_controllers/se2_screw_decomposed_impedance.py` | **SWIVL screw-decomposed impedance** |

### RL Policy (Layer 3)

| File | Description |
|------|-------------|
| `src/rl_policy/impedance_learning_env.py` | Gym environment for impedance learning |
| `src/rl_policy/ppo_impedance_policy.py` | PPO policy with FiLM conditioning |

### High-Level Planners (Layer 1)

| File | Description |
|------|-------------|
| `src/hl_planners/act.py` | Action Chunking Transformer |
| `src/hl_planners/diffusion_policy.py` | Diffusion Policy |
| `src/hl_planners/flow_matching.py` | Flow Matching Policy |

### Environment

| File | Description |
|------|-------------|
| `src/envs/biart.py` | Main BiArt environment |
| `src/envs/object_manager.py` | Articulated object management |
| `src/envs/end_effector_manager.py` | Gripper management |

## Documentation Files

### Controller Documentation

- [SE2 Impedance Verification](SE2_IMPEDANCE_VERIFICATION.md) - Mathematical verification of SE(2) impedance control
- [Impedance Controller Implementation](IMPEDANCE_CONTROLLER_IMPLEMENTATION.md) - Implementation guide
- [Impedance Controller Analysis](IMPEDANCE_CONTROLLER_ANALYSIS.md) - Analysis of impedance design

### System Architecture

- [SE2 Frame Conventions](SE2_FRAME_CONVENTIONS.md) - Frame conventions and transformations
- [Pipeline Flow Analysis](PIPELINE_FLOW_ANALYSIS.md) - Complete data flow from HL planner to wrench
- [HL-RL Pipeline](HL_RL_PIPELINE_README.md) - Hierarchical RL pipeline architecture

## SWIVL Layer 3: Observation and Action Spaces

### Observation Space (30D)

```python
observation = [
    ref_twists,       # 6D: V_l^ref, V_r^ref ∈ R^3 × 2
    screw_axes,       # 6D: B_l, B_r ∈ R^3 × 2
    wrenches,         # 6D: F_l, F_r ∈ R^3 × 2
    ee_poses,         # 6D: (x, y, θ) × 2
    ee_body_twists,   # 6D: (ω, vx, vy) × 2
]
```

### Action Space (7D)

```python
action = (
    d_l_parallel,  # Left arm internal motion damping
    d_r_parallel,  # Right arm internal motion damping
    d_l_perp,      # Left arm bulk motion damping
    d_r_perp,      # Right arm bulk motion damping
    k_p_l,         # Left arm pose correction gain
    k_p_r,         # Right arm pose correction gain
    alpha,         # Characteristic length (metric tensor)
)
```

## Reward Design

```
r_t = r_track + r_safety + r_reg + r_term
```

| Component | Formula | Description |
|-----------|---------|-------------|
| `r_track` | `-w_track * Σ\|\|V - V_ref\|\|²_G` | G-metric tracking error |
| `r_safety` | `w_safety * exp(-κ * Σ\|\|F_⊥\|\|²_{G⁻¹})` | Exponential safety (alive bonus) |
| `r_reg` | `-w_reg * Σ\|\|V̇\|\|²` | Twist acceleration penalty |
| `r_term` | `-w_term` (on failure) | Termination penalty |

**Dual Metric for Wrenches**: `||F||²_{G⁻¹} = τ²/α² + fx² + fy²`

## Configuration

All RL settings are in `scripts/configs/rl_config.yaml`:

```yaml
ll_controller:
  type: "screw_decomposed"
  screw_decomposed:
    min_d_parallel: 1.0
    max_d_parallel: 50.0
    min_alpha: 1.0
    max_alpha: 20.0

rl_training:
  reward:
    tracking_weight: 0.0001
    safety_reward_weight: 1.0
    safety_exp_scale: 0.01
    termination_penalty: 10.0
```

## Quick Start

```bash
# Setup
cd code_workspace
bash setup_conda.sh
conda activate swivl

# Train LL policy
python scripts/training/train_ll_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth

# Evaluate
python scripts/evaluation/eval_hierarchical_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --no_wrench
```

## Updates

- **2024-12**: Added exponential safety reward and termination penalty
- **2024-12**: Added wrench limit termination condition
- **2024-12**: Implemented dual metric G⁻¹ for wrench norms
- **2024-11**: Updated to Modern Robotics conventions
