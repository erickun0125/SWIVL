# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SWIVL (Screw-Wrench Informed Impedance Variable Learning) is a four-layer hierarchical control framework for bimanual manipulation of articulated objects. It combines imitation learning (Layer 1), reference generation (Layer 2), RL-based impedance modulation (Layer 3), and screw-decomposed impedance control (Layer 4).

## Environment Setup

```bash
cd code_workspace
bash setup_conda.sh
conda activate swivl
```

Python >=3.10 (3.11 used in conda setup). PyTorch with CUDA support for training.

## Common Commands

All commands run from `code_workspace/`:

```bash
# Tests
pytest scripts/tests/                          # All tests
pytest scripts/tests/test_core.py              # Core math/env/controller tests
pytest scripts/tests/test_controllers.py       # Controller-specific tests
python scripts/tests/test_core.py              # Direct execution also works

# Train HL policy (Layer 1)
python scripts/training/train_hl_policy.py --policy flow_matching
python scripts/training/train_hl_policy.py --policy act
python scripts/training/train_hl_policy.py --policy diffusion

# Train LL impedance policy (Layer 3)
python scripts/training/train_ll_policy.py --hl_policy act --hl_checkpoint checkpoints/act_best.pth --total_timesteps 500000
python scripts/training/train_ll_policy.py --hl_policy none   # Hold position mode

# Evaluate full pipeline
python scripts/evaluation/eval_hierarchical_policy.py --hl_policy act --hl_checkpoint checkpoints/act_best.pth --ll_checkpoint checkpoints/impedance_policy.zip --no_wrench

# Demos
python scripts/demos/demo_screw_impedance.py
python scripts/demos/demo_keyboard_teleoperation.py

# Formatting
black --line-length 100 src/
isort --profile black --line-length 100 src/
```

## Architecture

```
Layer 1: HL Policy (ACT/Diffusion/FlowMatching) → sparse waypoints @ 10Hz
    ↓
Layer 2: Reference Twist Field Generator → dense reference twists @ 100Hz
    ↓
Layer 3: Impedance Modulation Policy (PPO+FiLM) → 7D impedance variables
    ↓
Layer 4: Screw-Decomposed Impedance Controller → control wrenches
    ↓
BiArtEnv (Pymunk 2D physics) → next state
```

### Source Layout (`code_workspace/src/`)

- **Core math**: `se2_math.py` (SE(2) Lie group ops), `se2_dynamics.py` (task-space dynamics), `trajectory_generator.py`
- **Environments** (`envs/`): `biart.py` (main env, 512x512 workspace, Pymunk physics), `object_manager.py` (articulated objects with revolute/prismatic/fixed joints), `end_effector_manager.py`, `reward_manager.py`, `goal_manager.py`
- **Low-level controllers** (`ll_controllers/`): `se2_screw_decomposed_impedance.py` (SWIVL Layer 4 — core contribution), `se2_impedance_controller.py` (classical baseline)
- **HL planners** (`hl_planners/`): `act.py`, `diffusion_policy.py`, `flow_matching.py`, `image_encoder.py` (shared ResNet18), `teleoperation.py`, `keyboard_teleoperation.py`
- **RL policy** (`rl_policy/`): `ppo_impedance_policy.py` (PPO with FiLM conditioning), `impedance_learning_env.py` (Gym wrapper, 30D obs / 7D action)
- **Data utils** (`data_utils.py`): Shared `DataCollector` for HDF5 demo data

### Key Config

`scripts/configs/rl_config.yaml` is the **single source of truth** for RL training — covers HL policy, LL controller, environment, and reward settings.

## Mathematical Conventions

Follows **Modern Robotics (Lynch & Park)** — angular component always first:
- **Twist**: `V = [ω, vx, vy]ᵀ`
- **Wrench**: `F = [τ, fx, fy]ᵀ`
- **Metric tensor**: `G = diag(α², 1, 1)` for twists; `G⁻¹ = diag(1/α², 1, 1)` for wrenches

### Layer 3 Action Space (7D)

`(d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)` — per-arm damping for internal (∥) and bulk (⊥) motion, pose correction gains, and shared characteristic length.

### Layer 3 Observation Space (30D)

`[ref_twists(6), screw_axes(6), wrenches(6), ee_poses(6), ee_body_twists(6)]`

## Codebase Patterns

- **Config-driven**: YAML configs drive all training/eval parameters; dataclasses for type-safe internal config
- **Gymnasium interface**: `BiArtEnv` and `ImpedanceLearningEnv` follow standard Gym API
- **Stable-Baselines3**: PPO implementation uses SB3; checkpoints saved as `.zip`
- **HL policy checkpoints**: PyTorch `.pth` files
- **Physics**: Pymunk (2D rigid body physics); no custom C/C++
- **Screw decomposition**: G-orthogonal projectors `P_∥` and `P_⊥` decompose twist space into bulk and internal motion — this is the core mathematical contribution

## Paper

`paper_workspace/` contains the NeurIPS 2024 submission (LaTeX sources + compiled PDF).
