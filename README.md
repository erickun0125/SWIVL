# SWIVL - Screw and Wrench informed Impedance Variable Learning

Bimanual manipulation of articulated objects with inter-force interaction using reinforcement learning.

## 🎯 Project Overview

This repository contains the implementation of SWIVL, a framework for learning impedance parameters for bimanual manipulation of articulated objects. The system uses:

- **Task-space impedance control** with proper SE(2) dynamics
- **Screw-decomposed impedance control** for directional compliance
- **Low-level policy** that learns impedance variables (stiffness, damping) via RL
- **High-level policy** that provides desired trajectories
- **Dual-arm coordination** for manipulating shared 1-DOF linkage objects

## 🚀 Quick Start

### Installation

```bash
# 1. Setup conda environment and install dependencies
bash setup_conda.sh

# 2. Activate environment
conda activate swivl
```

### Run Demos

```bash
# Visualization demo
python scripts/run_visualization.py

# Teleoperation demo
python -m scripts.demos.demo_teleoperation

# Static hold demo
python -m scripts.demos.demo_static_hold
```

### Run Tests

```bash
# Run all tests
python -m pytest scripts/tests/

# Run specific test
python -m scripts.tests.test_biart_simple
```

### Training

```bash
# Train high-level policy (Flow Matching, Diffusion, or ACT)
python scripts/training/train_hl_policy.py --policy flow_matching

# Train low-level impedance learning policy
# Works with ANY HL policy and BOTH controller types
python scripts/training/train_ll_policy.py \
    --hl_policy flow_matching \
    --controller se2_impedance

# Train with screw-decomposed controller
python scripts/training/train_ll_policy.py \
    --hl_policy diffusion \
    --controller screw_decomposed
```

### Evaluation

```bash
# Evaluate hierarchical pipeline (HL + LL)
python scripts/evaluation/evaluate_hierarchical.py \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --num_episodes 50
```

## 📁 Repository Structure

```
SWIVL/
├── src/                           # Core implementation
│   ├── envs/                      # Environments
│   │   ├── biart.py              # Main BiArt environment
│   │   ├── object_manager.py     # Articulated object management
│   │   └── end_effector_manager.py
│   ├── ll_controllers/            # Low-level controllers
│   │   ├── se2_impedance_controller.py              # Standard impedance
│   │   ├── se2_screw_decomposed_impedance.py       # Screw decomposition
│   │   └── task_space_impedance.py                  # Backward compatibility
│   ├── hl_planners/               # High-level planners
│   │   ├── diffusion_policy.py   # Diffusion policy
│   │   ├── act_policy.py         # ACT policy
│   │   └── flow_matching_policy.py
│   ├── rl_policy/                 # RL policy
│   │   └── impedance_learning_env.py
│   ├── se2_math.py                # SE(2) math utilities
│   ├── se2_dynamics.py            # Robot dynamics
│   └── trajectory_generator.py    # Trajectory generation
├── scripts/                       # Scripts and utilities
│   ├── demos/                     # Demo scripts
│   │   ├── demo_teleoperation.py
│   │   └── demo_static_hold.py
│   ├── tests/                     # Test scripts
│   │   ├── test_biart_simple.py
│   │   ├── test_controllers.py
│   │   └── test_integrated_system.py
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Evaluation scripts
│   └── run_visualization.py       # Visualization runner
├── examples/                      # Usage examples
│   └── screw_decomposed_bimanual_control.py
├── docs/                          # Documentation
│   ├── SE2_IMPEDANCE_VERIFICATION.md       # Controller verification
│   ├── IMPEDANCE_CONTROLLER_IMPLEMENTATION.md
│   ├── PIPELINE_FLOW_ANALYSIS.md
│   └── SE2_FRAME_CONVENTIONS.md
└── README.md                      # This file
```

## 🤖 Key Features

### SE(2) Impedance Control

- **Proper robot dynamics:** Task-space inertia, Coriolis, gravity compensation
- **Model matching mode:** M_d = Lambda_b for guaranteed passivity
- **Acceleration feedforward:** Lambda_b * dV_d for improved tracking
- **Frame conventions:** Consistent spatial/body frame transformations

See [docs/SE2_IMPEDANCE_VERIFICATION.md](docs/SE2_IMPEDANCE_VERIFICATION.md) for mathematical verification.

### Screw-Decomposed Impedance Control

- **Directional compliance:** Independent impedance along/perpendicular to screw axis
- **Natural constraints:** Uses object's joint axis as screw
- **1D + 2D decomposition:** Parallel (compliant) + Perpendicular (stiff)
- **Coordinated bimanual control:** Both EEs respect kinematic constraints

Example:
```python
# Get joint axis in each EE frame
B_left, B_right = env.get_joint_axis_screws()

# Create screw-decomposed controller
controller = SE2ScrewDecomposedImpedanceController(
    screw_axis=B_left,
    params=ScrewImpedanceParams(
        K_parallel=10.0,      # Compliant along joint
        K_perpendicular=100.0 # Stiff to maintain grasp
    )
)
```

See [examples/screw_decomposed_bimanual_control.py](examples/screw_decomposed_bimanual_control.py) for complete example.

### High-Level Planners

- **Diffusion Policy:** Conditional diffusion for trajectory generation
- **ACT (Action Chunking Transformer):** Transformer-based policy
- **Flow Matching Policy:** Continuous normalizing flows

### RL-Based Impedance Learning

- **PPO for impedance parameters:** Learns optimal stiffness/damping
- **Separate HL/LL policies:** Trajectory planning + impedance control
- **Proper dynamics:** Full control pipeline with acceleration feedforward

## 📚 Documentation

- [SE(2) Impedance Verification](docs/SE2_IMPEDANCE_VERIFICATION.md) - Mathematical verification and comparison with SE(3)
- [Impedance Controller Implementation](docs/IMPEDANCE_CONTROLLER_IMPLEMENTATION.md) - Implementation guide
- [Pipeline Flow Analysis](docs/PIPELINE_FLOW_ANALYSIS.md) - Complete data flow from planner to physics
- [SE(2) Frame Conventions](docs/SE2_FRAME_CONVENTIONS.md) - Frame conventions used throughout

## 🧪 Testing

Run the test suite:

```bash
# All tests
python -m pytest scripts/tests/

# Controller tests
python -m scripts.tests.test_controllers

# Integration tests
python -m scripts.tests.test_integrated_system

# Stability tests
python -m scripts.tests.test_stability
```

## 🎮 Environment Details

### BiArt Environment

SE(2) bimanual manipulation environment with:
- Dual parallel-jaw grippers with wrench sensing
- Articulated objects (revolute, prismatic, fixed joints)
- Proper SE(2) dynamics and frame transformations
- External wrench sensing via Pymunk collision handlers

**Observation Space:**
```python
{
    'ee_poses': (2, 3),          # [x, y, theta] in spatial frame
    'ee_twists': (2, 3),         # [vx, vy, omega] in spatial frame
    'link_poses': (2, 3),        # Object link poses
    'external_wrenches': (2, 3)  # [fx, fy, tau] in body frame
}
```

**Action Space:**
```python
# Wrenches in body frame for both grippers
[left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
```

## 🔬 Research

This project implements:
- **Proper SE(2) impedance control** with full robot dynamics
- **Screw-axis based impedance decomposition** for directional compliance
- **RL for impedance learning** with separation of trajectory and compliance
- **Bimanual coordination** via kinematic constraints

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{swivl2024,
  title={SWIVL: Screw and Wrench informed Impedance Variable Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SWIVL}
}
```

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Your Contact Information]
