# SWIVL: Screw-Wrench Informed Impedance Variable Learning

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**A hierarchical control framework for bimanual manipulation of articulated objects with wrench-adaptive impedance modulation.**

Bimanual manipulation of articulated objects requires simultaneous satisfaction of kinematic constraints and regulation of inter-arm contact forces -- capabilities that current imitation-based approaches lack due to high-stiffness control without adaptive force modulation. SWIVL bridges cognitive planning with physical execution through three core contributions:

1. **Twist-driven impedance control** via stable imitation vector fields that bypass nonlinear pose-error Jacobians
2. **Screw-axes decomposition** enabling independent compliance tuning for bulk (object transport) and internal (joint articulation) motions
3. **Wrench-adaptive impedance variable learning** via RL that suppresses excessive inter-arm fighting forces

## Architecture

SWIVL decomposes the bimanual manipulation problem into a four-layer hierarchy, where high-level cognitive planning at 10 Hz is translated into physically grounded impedance-controlled execution at 100 Hz.

<p align="center">
  <img src="figures/architecture_overview.jpg" alt="SWIVL 4-Layer Architecture" width="700"/>
</p>

- **Layer 1 -- High-Level Policy** (ACT / Diffusion / Flow Matching): Generates sparse SE(3) pose action chunks from visual observations at ~10 Hz.
- **Layer 2 -- Reference Twist Field Generator**: Smooths sparse waypoints into dense reference twist fields via trajectory interpolation and stable imitation vector fields at ~100 Hz.
- **Layer 3 -- Impedance Modulation Policy** (PPO + FiLM): Modulates 7D impedance variables conditioned on real-time wrench feedback and object screw axes.
- **Layer 4 -- Screw-Decomposed Impedance Controller**: Decomposes twist/wrench space into bulk and internal subspaces via G-orthogonal projectors, applying independently tuned compliance to each.

### Screw-Decomposed Impedance Control

The core mathematical idea: given an object's screw axis, G-orthogonal projectors partition each end-effector's twist space into **internal motion** (joint articulation) and **bulk motion** (object transport):

$$P_{i,\parallel} = J_i (J_i^\top G\, J_i)^{-1} J_i^\top G, \qquad P_{i,\perp} = I - P_{i,\parallel}$$

This enables independent damping for each subspace -- high compliance along internal motion to avoid fighting forces, firm tracking along bulk motion to maintain coordination:

$$\mathcal{F}_{\text{cmd},i} = d_{i,\parallel}\, G\, (\mathcal{V}_{i,\parallel}^{\text{ref}} - \mathcal{V}_{i,\parallel}) + d_{i,\perp}\, G\, (\mathcal{V}_{i,\perp}^{\text{ref}} - \mathcal{V}_{i,\perp}) + \mu_{b,i} + \gamma_{b,i}$$

where $d_{i,\parallel}, d_{i,\perp}$ are per-subspace damping coefficients learned by the RL policy (Layer 3), and $G = \text{diag}(\alpha^2, 1, 1)$ is a metric tensor with learnable characteristic length $\alpha$.

## Reference Twist Field

A key challenge in impedance-based imitation is converting sparse pose waypoints from a high-level policy into dense, stable reference twists for the impedance controller. SWIVL's Reference Twist Field Generator combines feedforward imitation terms with pose-error correction to create a stable vector field that ensures robust tracking even under large deviations.

<p align="center">
  <img src="figures/reference_twist_field.jpg" alt="Reference Twist Field Generator" width="600"/>
</p>

## Inference

The full SWIVL pipeline in action: two end-effectors cooperatively manipulate an articulated object while the RL policy (Layer 3) adaptively modulates impedance parameters in real time. The overlay shows learned impedance variables (damping coefficients, stiffness gains, characteristic length) evolving throughout the task, and red arrows indicate wrench feedback from force/torque sensors.

<p align="center">
  <img src="figures/swivl_inference_snapshots.jpg" alt="SWIVL Inference Sequence" width="800"/>
</p>

## BiarT Benchmark & Results

SWIVL is evaluated on **BiarT**, an SE(2) planar benchmark for bimanual articulated manipulation built on Pymunk physics. The environment features a 512x512 workspace with dual 3-DoF end-effectors controlling articulated objects with revolute (angular) or prismatic (linear) joints at 100 Hz.

| Method | Revolute | Prismatic | Avg. Success | Wrench Limit Failures |
|:-------|:--------:|:---------:|:------------:|:---------------------:|
| Position Control | 10% | 30% | 20% | 55% |
| Impedance Control | 10% | 60% | 35% | 10% |
| **SWIVL (Ours)** | **40%** | **80%** | **60%** | **0%** |

SWIVL achieves 3x higher average success rate over position control while **completely eliminating wrench limit violations** -- the dangerous inter-arm fighting forces that cause task failure and potential hardware damage. Classical impedance control reduces but cannot eliminate these forces; SWIVL's learned screw-decomposed compliance suppresses them entirely.

## Quick Start

### Installation

```bash
cd code_workspace
bash setup_conda.sh
conda activate swivl
```

### Run Demos

```bash
cd code_workspace

python scripts/demos/demo_screw_impedance.py        # SWIVL screw-decomposed impedance
python scripts/demos/demo_se2_impedance.py           # Classical SE(2) impedance
python scripts/demos/demo_keyboard_teleoperation.py  # Interactive keyboard control
```

### Train

```bash
cd code_workspace

# High-level policy (Layer 1)
python scripts/training/train_hl_policy.py --policy flow_matching
python scripts/training/train_hl_policy.py --policy act
python scripts/training/train_hl_policy.py --policy diffusion

# Low-level impedance modulation policy (Layer 3)
python scripts/training/train_ll_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --total_timesteps 500000
```

### Evaluate

```bash
cd code_workspace

python scripts/evaluation/eval_hierarchical_policy.py \
    --hl_policy act \
    --hl_checkpoint checkpoints/act_best.pth \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --no_wrench
```

### Test

```bash
cd code_workspace
pytest scripts/tests/ -v
```

## Repository Structure

```
code_workspace/
  src/
    se2_math.py                              # SE(2) Lie group operations
    se2_dynamics.py                          # Task-space dynamics
    envs/biart.py                            # BiarT environment (Pymunk)
    ll_controllers/
      se2_screw_decomposed_impedance.py      # SWIVL Layer 4 (core contribution)
      se2_impedance_controller.py            # Classical SE(2) impedance baseline
    hl_planners/
      act.py | diffusion_policy.py | flow_matching.py
    rl_policy/
      impedance_learning_env.py              # Gym wrapper (30D obs / 7D action)
      ppo_impedance_policy.py                # PPO with FiLM conditioning
  scripts/
    configs/rl_config.yaml                   # Single source of truth for RL training
    training/   | evaluation/   | demos/   | data_collection/

paper_workspace/                             # NeurIPS 2025 submission (LaTeX)
```

## Documentation

See [`code_workspace/docs/`](code_workspace/docs/README.md) for detailed technical documentation:

- [Documentation Index](code_workspace/docs/README.md) -- Architecture, conventions, observation/action spaces
- [SE(2) Impedance Verification](code_workspace/docs/SE2_IMPEDANCE_VERIFICATION.md)
- [Pipeline Flow Analysis](code_workspace/docs/PIPELINE_FLOW_ANALYSIS.md)
- [HL-RL Pipeline](code_workspace/docs/HL_RL_PIPELINE_README.md)
- [SE(2) Frame Conventions](code_workspace/docs/SE2_FRAME_CONVENTIONS.md)

## Citation

```bibtex
@article{swivl2025,
  title={SWIVL: Screw-Wrench Informed Impedance Variable Learning for Bimanual Manipulation},
  author={Park, Kyungseo},
  year={2025}
}
```

## License

MIT License

## Contact

Kyungseo Park - erickun0125@snu.ac.kr
