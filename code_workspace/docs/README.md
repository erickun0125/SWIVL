# SWIVL Documentation

This directory contains technical documentation for the SWIVL project (Screw-Wrench informed Impedance Variable Learning).

## Overview

SWIVL is a hierarchical control framework for bimanual manipulation of articulated objects that bridges cognitive planning with physical execution. The system operates in SE(2) (planar robotics) and features:

- **Twist-Driven Impedance Control** via stable imitation vector fields
- **Screw Axes-Decomposed** control for independent compliance
- **Wrench-adaptive Impedance Variable Learning** via RL

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL PLANNERS                      │
│   (FlowMatching / Diffusion / ACT / Teleoperation)         │
│   Output: Desired poses (2, 3) in spatial frame            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRAJECTORY GENERATOR                       │
│   (CubicSpline / MinimumJerk)                               │
│   Output: TrajectoryPoint (pose, twist, acceleration)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LOW-LEVEL CONTROLLERS                      │
│   (SE2 Impedance / Screw-Decomposed Impedance / PD)        │
│   Output: Wrench [τ, fx, fy] in body frame (MR convention) │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     PHYSICS ENGINE                          │
│   (BiArtEnv with Pymunk)                                   │
│   Output: Observation dict                                  │
└─────────────────────────────────────────────────────────────┘
```

## Modern Robotics Conventions

This codebase follows **Modern Robotics (Lynch & Park)** conventions:

- **Twist:** `V = [ω, vx, vy]ᵀ` (angular velocity first)
- **Wrench:** `F = [τ, fx, fy]ᵀ` (torque first)
- **Screw axis:** `S = [sω, sx, sy]ᵀ` (angular component first)

## Documentation Files

### Controller Documentation

- [SE2 Impedance Verification](SE2_IMPEDANCE_VERIFICATION.md) - Mathematical verification of SE(2) impedance control and screw decomposition
- [Impedance Controller Implementation](IMPEDANCE_CONTROLLER_IMPLEMENTATION.md) - Implementation guide for proper SE(2) impedance control
- [Impedance Controller Analysis](IMPEDANCE_CONTROLLER_ANALYSIS.md) - Analysis of impedance controller design

### System Architecture

- [SE2 Frame Conventions](SE2_FRAME_CONVENTIONS.md) - Frame conventions and transformations used throughout the system
- [Pipeline Flow Analysis](PIPELINE_FLOW_ANALYSIS.md) - Complete data flow from HL planner to wrench command
- [HL-RL Pipeline](HL_RL_PIPELINE_README.md) - High-level and reinforcement learning pipeline architecture

## Key Source Files

| File | Description |
|------|-------------|
| `src/se2_math.py` | SE(2) Lie group operations |
| `src/se2_dynamics.py` | Robot dynamics (mass, Coriolis, gravity) |
| `src/ll_controllers/se2_impedance_controller.py` | SE(2) task-space impedance control |
| `src/ll_controllers/se2_screw_decomposed_impedance.py` | Screw-axis decomposed impedance control |
| `src/envs/biart.py` | Main bimanual manipulation environment |
| `src/rl_policy/impedance_learning_env.py` | RL environment for impedance learning |

## Quick Start

```bash
# Setup conda environment
./setup_conda.sh
conda activate swivl

# Run teleoperation demo
python scripts/demos/demo_teleoperation.py revolute

# Test basic functionality
python -c "from src.envs import BiArtEnv; env = BiArtEnv(); env.reset(); print('Success!')"
```

## Additional Resources

- See `../examples/` for usage examples
- See `../scripts/tests/` for test files
- See `../scripts/demos/` for demonstration scripts

## Updates

- **2025-11-27**: Updated all documentation to reflect Modern Robotics conventions and current code structure
