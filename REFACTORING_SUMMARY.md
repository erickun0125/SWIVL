# Repository Refactoring Summary

## Overview

This document describes the major refactoring of the SWIVL repository to a more organized, modular structure suitable for research and development in bimanual manipulation.

## New Directory Structure

```
SWIVL/
├── src/                          # Source code
│   ├── envs/                     # SE(2) BiArt environments
│   │   ├── biart.py             # Main bimanual manipulation environment
│   │   ├── linkage_manager.py   # Articulated object management
│   │   └── pymunk_override.py   # Physics engine utilities
│   │
│   ├── ll_controllers/           # Low-level controllers (wrench output)
│   │   ├── pd_controller.py     # Position & Orientation decomposed PD
│   │   ├── task_space_impedance.py  # Task space impedance control
│   │   └── screw_impedance.py   # Screw-aware impedance control
│   │
│   ├── hl_planners/              # High-level planners (desired pose output @ 10Hz)
│   │   ├── flow_matching.py     # Flow matching policy
│   │   └── teleoperation.py     # Teleoperation interface
│   │
│   ├── se2_math.py               # SE(2) Lie group/algebra mathematics
│   └── trajectory_generator.py   # Cubic spline trajectory generation
│
├── scripts/                      # Scripts for experiments
│   ├── configs/                  # Configuration files
│   │   └── default_config.yaml  # Default experiment configuration
│   │
│   ├── tests/                    # Test scripts
│   │   ├── test_controllers.py  # Controller unit tests
│   │   ├── test_demo_smoke.py   # Smoke tests
│   │   └── test_stability.py    # Stability tests
│   │
│   ├── training/                 # Training scripts (placeholder)
│   └── evaluation/               # Evaluation scripts (placeholder)
│
└── gym_biart/                    # Legacy code (kept for compatibility)
```

## Key Components

### 1. SE(2) Mathematics (`src/se2_math.py`)

Comprehensive SE(2) Lie group and Lie algebra utilities:

- **Exponential/Logarithm maps**: `se2_exp()`, `se2_log()`
- **Adjoint representations**: `se2_adjoint()`, `se2_adjoint_algebra()`
- **Pose operations**: `se2_compose()`, `se2_inverse()`, `se2_interpolate()`
- **Frame transformations**: `body_to_world_velocity()`, `world_to_body_velocity()`
- **Screw axis computation**: `compute_screw_axis()`
- **Distance metrics**: `se2_distance()`

### 2. Trajectory Generation (`src/trajectory_generator.py`)

Smooth trajectory generation using cubic splines:

- **CubicSplineTrajectory**: Cubic spline interpolation with separate position/orientation handling
- **MinimumJerkTrajectory**: Minimum jerk trajectory for natural motion
- **TrajectoryManager**: Multi-segment trajectory chaining
- **Predefined trajectories**: Circular, lemniscate (figure-eight) patterns

Key Features:
- Proper angle unwrapping to prevent discontinuities
- Velocity and acceleration computation via derivatives
- Configurable boundary conditions
- Arc length parametrization

### 3. Low-Level Controllers (`src/ll_controllers/`)

All controllers share a common interface: `compute_wrench(current_pose, desired_pose, ...) -> wrench`

#### PD Controller (`pd_controller.py`)
- Position & orientation decomposed control
- Body frame error computation
- Velocity feedback or finite difference estimation
- Force/torque saturation

#### Task Space Impedance Controller (`task_space_impedance.py`)
- Adjustable stiffness and damping
- Force-based compliance activation
- Threshold-based impedance modulation
- Suitable for contact-rich tasks

#### Screw-Aware Impedance Controller (`screw_impedance.py`)
- Decomposes motion along/perpendicular to screw axis
- Different impedance properties in each direction
- Better handling of constrained motions
- Improved force control for manipulation

### 4. High-Level Planners (`src/hl_planners/`)

All planners output desired poses at 10 Hz for the low-level controller.

#### Flow Matching Policy (`flow_matching.py`)
- Conditional flow matching for trajectory generation
- Neural network-based policy
- Context-aware planning with state history
- ODE integration for sampling

Architecture:
- Time embedding MLP
- State/Action embedding MLPs
- Multi-layer main network
- Velocity field prediction

#### Teleoperation Planner (`teleoperation.py`)
- Keyboard/joystick interface
- Real-time pose command generation
- Multi-EE coordination
- Copied from existing keyboard_planner.py

### 5. Environment (`src/envs/`)

BiArt environment with articulated objects:
- **Joint types**: Revolute, Prismatic, Fixed
- **Parallel grippers**: 1-DOF with constant closing force
- **External wrench sensing**: Body frame force/torque measurement
- **Pymunk physics**: 2D rigid body simulation

## Design Principles

### 1. **Modular Architecture**
- Clear separation of concerns
- Each component has a specific responsibility
- Easy to swap implementations

### 2. **Common Interfaces**
- Low-level controllers: `compute_wrench()`
- High-level planners: `get_action()`
- Consistent input/output formats

### 3. **Proper SE(2) Mathematics**
- Lie group operations for poses
- Screw theory for motion description
- Proper angle handling and interpolation

### 4. **Configurability**
- YAML configuration files
- Dataclass-based parameter management
- Easy hyperparameter tuning

### 5. **Extensibility**
- Easy to add new controllers
- Easy to add new planners
- Plugin architecture potential

## Control Flow

```
High-Level Planner (10 Hz)
    ↓ desired_pose
Low-Level Controller (10 Hz)
    ↓ wrench command
BiArt Environment
    ↓ observation
[feedback loop]
```

## Key Improvements Over Previous Structure

1. **Better Organization**
   - Clear hierarchy: envs → ll_controllers → hl_planners
   - Separated concerns: math, trajectory, control, planning

2. **Proper Mathematics**
   - SE(2) Lie group operations
   - Screw-aware control
   - Geodesic interpolation

3. **Multiple Controller Options**
   - PD for simple tasks
   - Impedance for contact tasks
   - Screw-aware for constrained motions

4. **Learning-Ready Structure**
   - Flow matching policy template
   - Training/evaluation script placeholders
   - Configuration management

5. **Testing Infrastructure**
   - Unit tests for controllers
   - Integration tests for full system
   - Stability tests for physics

## Migration Notes

### Old → New Mapping

- `gym_biart/envs/biart.py` → `src/envs/biart.py`
- `gym_biart/envs/pd_controller.py` → `src/ll_controllers/pd_controller.py`
- `gym_biart/envs/keyboard_planner.py` → `src/hl_planners/teleoperation.py`
- Test scripts → `scripts/tests/`

### Breaking Changes

- Import paths changed: `from src.ll_controllers import PDController`
- Controller class renamed: `PoseController` → `PDController`
- Multi-gripper controller: `MultiGripperController` → `MultiGripperPDController`

### Backward Compatibility

- Legacy `gym_biart/` kept for compatibility
- Old demo scripts still work
- Gradual migration recommended

## Usage Examples

### Using PD Controller

```python
from src.ll_controllers import PDController, PDGains

controller = PDController(
    gains=PDGains(kp_linear=50.0, kd_linear=10.0)
)
controller.set_timestep(0.01)

wrench = controller.compute_wrench(
    current_pose=np.array([0, 0, 0]),
    desired_pose=np.array([10, 5, 0.5])
)
```

### Using Impedance Controller

```python
from src.ll_controllers import TaskSpaceImpedanceController, ImpedanceGains

controller = TaskSpaceImpedanceController(
    gains=ImpedanceGains(force_threshold=20.0, compliance_factor=0.5)
)

wrench = controller.compute_wrench(
    current_pose, desired_pose, measured_wrench
)
```

### Generating Trajectories

```python
from src.trajectory_generator import CubicSplineTrajectory

waypoints = [
    [0, 0, 0],
    [10, 5, 0.5],
    [20, 10, 1.0]
]

traj = CubicSplineTrajectory(waypoints)
traj.set_duration(5.0)  # 5 seconds

# Sample at specific time
point = traj.evaluate(2.5)
print(point.pose, point.velocity, point.acceleration)
```

### Using SE(2) Math

```python
from src.se2_math import se2_interpolate, se2_exp, se2_log

# Interpolate between poses
pose_mid = se2_interpolate(pose_start, pose_end, alpha=0.5)

# Exponential map
T = se2_exp(np.array([vx, vy, omega]))

# Logarithm map
xi = se2_log(T)
```

## Future Work

### Training Pipeline
- Implement imitation learning for flow matching
- Implement RL for variable impedance learning
- Data collection utilities

### Evaluation Pipeline
- Benchmarking scripts
- Metrics computation
- Visualization tools

### Additional Controllers
- Adaptive impedance
- Learning-based controllers
- Hybrid position/force control

### Additional Planners
- Diffusion Policy
- Action Chunking Transformer
- MPC-based planning

## Testing

Run all tests:
```bash
python scripts/tests/test_controllers.py
python scripts/tests/test_demo_smoke.py
python scripts/tests/test_stability.py
```

## Configuration

Edit `scripts/configs/default_config.yaml` to change:
- Environment parameters
- Controller gains
- Planner settings
- Training hyperparameters

## Conclusion

This refactoring provides a solid foundation for research in bimanual manipulation with:
- Clean, modular architecture
- Proper mathematical foundations
- Multiple control strategies
- Learning-ready structure
- Comprehensive testing

The structure is designed to be easily extended with new controllers, planners, and learning algorithms while maintaining a consistent interface throughout the system.
