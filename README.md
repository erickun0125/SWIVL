# SWIVL - Screw-Wrench Informed Impedance Variable Learning

Hierarchical control framework for bimanual manipulation of articulated objects with wrench-adaptive impedance modulation.

## 🎯 Project Overview

SWIVL is a four-layer hierarchical control framework that bridges high-level cognitive planning with physically grounded bimanual execution:

1. **Layer 1 (High-Level Policy)**: VLA, behavior cloning (ACT/Diffusion/Flow Matching), or teleoperation → sparse waypoints
2. **Layer 2 (Reference Twist Field Generator)**: Transforms waypoints into dense, stable reference twists
3. **Layer 3 (Impedance Modulation Policy)**: RL policy that modulates impedance variables based on wrench feedback
4. **Layer 4 (Screw-Decomposed Impedance Controller)**: Executes compliant control with independent bulk/internal motion regulation

## 📁 Repository Structure

```
SWIVL/
├── code_workspace/                    # Implementation code
│   ├── src/                           # Core source code
│   │   ├── envs/                      # Environments
│   │   │   ├── biart.py              # Main BiArt environment
│   │   │   ├── object_manager.py     # Articulated object management
│   │   │   └── end_effector_manager.py
│   │   ├── ll_controllers/            # Low-level controllers
│   │   │   ├── se2_impedance_controller.py
│   │   │   └── se2_screw_decomposed_impedance.py
│   │   ├── hl_planners/               # High-level planners
│   │   │   ├── act.py
│   │   │   ├── diffusion_policy.py
│   │   │   └── flow_matching.py
│   │   ├── rl_policy/                 # RL policy (Layer 3)
│   │   │   ├── impedance_learning_env.py
│   │   │   └── ppo_impedance_policy.py
│   │   ├── se2_math.py               # SE(2) Lie group operations
│   │   ├── se2_dynamics.py           # Robot dynamics
│   │   └── trajectory_generator.py
│   ├── scripts/                       # Scripts
│   │   ├── configs/                   # Configuration files
│   │   │   ├── rl_config.yaml        # RL training config (single source of truth)
│   │   │   └── hl_policy_config.yaml
│   │   ├── training/                  # Training scripts
│   │   │   ├── train_hl_policy.py
│   │   │   └── train_ll_policy.py
│   │   ├── evaluation/                # Evaluation scripts
│   │   │   └── eval_hierarchical_policy.py
│   │   ├── demos/                     # Demo scripts
│   │   └── data_collection/           # Data collection
│   ├── docs/                          # Technical documentation
│   ├── checkpoints/                   # Trained model checkpoints
│   ├── data/                          # Training data
│   └── logs/                          # Training logs
│
└── paper_workspace/                   # Paper materials
    ├── main_contents/                 # Main paper sections
    ├── appendix/                      # Appendix sections
    ├── figures/                       # Paper figures
    └── neurips_2024_main.tex          # Main LaTeX file
```

## 🚀 Quick Start

### Installation

```bash
cd code_workspace

# Setup conda environment
bash setup_conda.sh

# Activate environment
conda activate swivl
```

### Train High-Level Policy

```bash
cd code_workspace

# Train Flow Matching policy
python scripts/training/train_hl_policy.py --policy flow_matching

# Train ACT policy
python scripts/training/train_hl_policy.py --policy act

# Train Diffusion policy
python scripts/training/train_hl_policy.py --policy diffusion
```

### Train Low-Level (Impedance Modulation) Policy

```bash
cd code_workspace

# Train with HL policy (recommended)
python scripts/training/train_ll_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --total_timesteps 500000

# Train without HL policy (hold position mode)
python scripts/training/train_ll_policy.py --hl_policy none
```

**SWIVL Layer 3 Action Space (7D):**
- `d_l_∥, d_r_∥`: Per-arm damping for internal motion (parallel to screw axis)
- `d_l_⊥, d_r_⊥`: Per-arm damping for bulk motion (perpendicular to screw axis)
- `k_p_l, k_p_r`: Per-arm pose error correction gains
- `α`: Characteristic length (metric tensor G = diag(α², 1, 1))

### Evaluate Hierarchical Pipeline

```bash
cd code_workspace

# Interactive evaluation with visualization
python scripts/evaluation/eval_hierarchical_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --no_wrench

# Batch evaluation (N episodes)
python scripts/evaluation/eval_hierarchical_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --num_episodes 100 \
    --no_wrench
```

**Evaluation Controls:**
- `SPACE` - Pause/Resume
- `R` - Reset episode
- `I` - Toggle impedance info display
- `ESC` - Exit

### Run Demos

```bash
cd code_workspace

# Screw-decomposed impedance control demo
python scripts/demos/demo_screw_impedance.py

# SE(2) impedance control demo
python scripts/demos/demo_se2_impedance.py

# Keyboard teleoperation
python scripts/demos/demo_keyboard_teleoperation.py
```

## 📐 SWIVL Framework

### Reward Design

The RL policy is trained with the following reward structure:

```
r_t = r_track + r_safety + r_reg + r_term
```

| Component | Formula | Description |
|-----------|---------|-------------|
| `r_track` | `-w_track * Σ\|\|V_i - V_ref_i\|\|²_G` | G-metric velocity tracking |
| `r_safety` | `w_safety * exp(-κ * Σ\|\|F_⊥\|\|²_{G⁻¹})` | Exponential safety reward (alive bonus) |
| `r_reg` | `-w_reg * Σ\|\|V̇_i\|\|²` | Twist acceleration regularization |
| `r_term` | `-w_term` | Termination penalty for failures |

**Key Design Choices:**
- **Exponential safety reward** provides positive "alive bonus" when fighting forces are low
- **Dual metric G⁻¹** for wrench norms ensures dimensional consistency
- **Termination penalty** discourages learning to intentionally fail

### Termination Conditions

1. **Grasp Drift**: Geodesic distance exceeds threshold
2. **Wrench Limit**: External wrench magnitude (G⁻¹-weighted) exceeds limit

### Modern Robotics Convention

This codebase follows **Modern Robotics (Lynch & Park)** conventions:
- **Twist:** `V = [ω, vx, vy]ᵀ` (angular velocity first)
- **Wrench:** `F = [τ, fx, fy]ᵀ` (torque first)
- **Metric Tensor:** `G = diag(α², 1, 1)` for twists, `G⁻¹ = diag(1/α², 1, 1)` for wrenches

## 📚 Documentation

See `code_workspace/docs/` for detailed technical documentation:
- [Documentation Index](code_workspace/docs/README.md)
- [SE(2) Impedance Verification](code_workspace/docs/SE2_IMPEDANCE_VERIFICATION.md)
- [Impedance Controller Implementation](code_workspace/docs/IMPEDANCE_CONTROLLER_IMPLEMENTATION.md)
- [Pipeline Flow Analysis](code_workspace/docs/PIPELINE_FLOW_ANALYSIS.md)
- [HL-RL Pipeline](code_workspace/docs/HL_RL_PIPELINE_README.md)

## ⚙️ Configuration

All RL training settings are in `code_workspace/scripts/configs/rl_config.yaml` (single source of truth):

```yaml
# Key configurations
ll_controller:
  type: "screw_decomposed"  # or "se2_impedance"
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
```

## 🧪 Testing

```bash
cd code_workspace

# Core functionality tests
python scripts/tests/test_core.py

# Controller tests
python scripts/tests/test_controllers.py
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{swivl2024,
  title={SWIVL: Screw-Wrench Informed Impedance Variable Learning for Bimanual Manipulation},
  author={Park, Kyungseo},
  year={2024}
}
```

## 📄 License

MIT License

## 📧 Contact

Kyungseo Park - erickun0125@snu.ac.kr
