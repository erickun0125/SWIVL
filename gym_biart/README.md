# BiArt Environment

**BiArt** (Bimanual Articulated object manipulation) is a Gymnasium-based SE(2) environment for bimanual manipulation of articulated objects using wrench control.

## Overview

This environment is designed for the SWIVL (Screw and Wrench informed Impedance Variable Learning) research project. It simulates dual-arm manipulation of articulated objects in a 2D plane (SE(2) space) with:

- **Dual robot arms** with dynamic end-effectors
- **Wrench command control** (force and moment in body frame)
- **U-shaped (ㄷ) grippers** with parallel grip mechanism
- **Articulated objects** with revolute, prismatic, or fixed joints
- **External wrench sensing** in body frame

## Features

### 1. SE(2) State Space
- Position (x, y) and orientation (θ) for all objects
- Full 2D rigid body dynamics using Pymunk

### 2. Wrench Command Control
- Actions are wrench commands: `[left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]`
- Forces and moments are specified in the body frame of each gripper
- Dynamic bodies respond to applied wrenches according to rigid body dynamics

### 3. Dynamic Grippers
- End-effectors are dynamic bodies (not kinematic)
- U-shaped (ㄷ) gripper design for stable grasping
- Constant grip force to hold objects

### 4. Articulated Objects
Three types of joints supported:
- **Revolute**: Rotational joint (e.g., door hinge)
- **Prismatic**: Sliding joint (e.g., drawer)
- **Fixed**: No relative motion between links

### 5. External Wrench Sensor
- Measures contact forces and moments in body frame
- Useful for impedance control and force feedback
- Included in observation space

## Installation

```bash
# Install dependencies
pip install gymnasium numpy pymunk pygame opencv-python shapely

# The environment is ready to use locally
```

## Usage

### Basic Example

```python
import gymnasium as gym
import gym_biart

# Create environment
env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type="revolute")

# Reset
observation, info = env.reset()

# Run episode
for _ in range(1000):
    # Random wrench commands
    action = env.action_space.sample()

    # Step
    observation, reward, terminated, truncated, info = env.step(action)

    # Render
    env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Running Example Script

```bash
cd gym_biart
python example.py
```

## Environment Details

### Observation Space

**State observation** (default):
```python
observation = [
    # Left gripper (x, y, θ)
    left_x, left_y, left_theta,

    # Right gripper (x, y, θ)
    right_x, right_y, right_theta,

    # Link 1 (x, y, θ)
    link1_x, link1_y, link1_theta,

    # Link 2 (x, y, θ)
    link2_x, link2_y, link2_theta,

    # External wrench left (fx, fy, τ) in body frame
    left_ext_fx, left_ext_fy, left_ext_tau,

    # External wrench right (fx, fy, τ) in body frame
    right_ext_fx, right_ext_fy, right_ext_tau,
]
# Total: 18 dimensions
```

### Action Space

Wrench commands in body frame:
```python
action = [
    left_fx,     # Left gripper force x (body frame)
    left_fy,     # Left gripper force y (body frame)
    left_tau,    # Left gripper moment (torque)
    right_fx,    # Right gripper force x (body frame)
    right_fy,    # Right gripper force y (body frame)
    right_tau,   # Right gripper moment (torque)
]
# Shape: (6,)
# Range: forces ∈ [-100, 100], torques ∈ [-50, 50]
```

### Reward Function

The reward consists of two components:

1. **Tracking Reward**: Exponential reward based on position and orientation error from goal
   ```python
   tracking_reward = exp(-0.1 * pos_error) * exp(-angle_error)
   ```

2. **Safety Penalty**: Penalizes excessive contact forces
   ```python
   safety_penalty = -0.01 * max_external_wrench
   ```

Total reward:
```python
reward = tracking_reward + safety_penalty
```

### Environment Parameters

```python
env = gym.make(
    "gym_biart/BiArt-v0",
    obs_type="state",           # "state" or "pixels"
    render_mode="rgb_array",    # "rgb_array" or "human"
    joint_type="revolute",      # "revolute", "prismatic", or "fixed"
    observation_width=96,
    observation_height=96,
    visualization_width=680,
    visualization_height=680,
)
```

## Code Structure

```
gym_biart/
├── __init__.py              # Environment registration
├── README.md               # This file
├── example.py              # Example usage script
└── envs/
    ├── __init__.py         # Exports BiArtEnv
    ├── biart.py            # Main environment implementation
    └── pymunk_override.py  # Custom rendering utilities
```

## Key Implementation Details

### Gripper Design
- **Shape**: U-shaped (ㄷ) with three rectangles (left arm, bottom, right arm)
- **Body type**: Dynamic (responds to forces)
- **Control**: Wrench commands in body frame
- **Grip**: Constant attractive force when near grasping part

### Articulated Object
- **Links**: Two rectangular links with grasping parts
- **Joints**: Revolute, prismatic, or fixed constraint
- **Grasping parts**: Small protrusions that fit inside gripper opening

### Physics Simulation
- **Engine**: Pymunk (2D rigid body physics)
- **Time step**: 0.01s (100 Hz)
- **Control frequency**: 10 Hz (10 physics steps per action)
- **Damping**: 0.95 for stability

### Wrench Sensing
- Contact forces measured during collision callbacks
- Transformed to body frame using rotation matrix
- Torques computed from contact points using cross product
- Updated every physics step

## Future Improvements

- [ ] Add more sophisticated gripping mechanism (contact-based)
- [ ] Implement joint limits for articulated objects
- [ ] Add joint angle/velocity to observations
- [ ] Implement high-level policy interface for desired trajectories
- [ ] Add visualization of goal pose and forces
- [ ] Support for more complex articulated objects (>2 links)

## Research Context

This environment is part of the SWIVL (Screw and Wrench informed Impedance Variable Learning) project, which aims to learn impedance parameters for bimanual manipulation of articulated objects using reinforcement learning.

Key research aspects:
- **Task-space impedance control**: Control framework for robot manipulation
- **Low-level policy**: Learns impedance variables (stiffness, damping)
- **High-level policy**: Provides desired trajectories
- **Rewards**: Tracking performance and safety constraints

## License

This environment is developed for research purposes.

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{biart2024,
  title={BiArt: Bimanual Articulated Object Manipulation Environment},
  author={SWIVL Research Team},
  year={2024},
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
