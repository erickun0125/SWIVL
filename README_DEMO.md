# Bimanual Manipulation System - Demo Guide

## Overview

This system provides a complete bimanual manipulation framework with:
- **Linkage Object Management**: Support for R (Revolute), P (Prismatic), and Fixed joints
- **PD Controller**: Low-level pose control with wrench output
- **Keyboard Teleoperation**: High-level planning with automatic grasp maintenance

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Interactive Demo

The main demo provides real-time teleoperation with full visualization:

```bash
# Revolute joint (default)
python demo_teleoperation.py

# Prismatic joint
python demo_teleoperation.py prismatic

# Fixed joint
python demo_teleoperation.py fixed
```

## Demo Features

### Real-Time Visualization

The demo displays:
- **Episode Information**: Episode number, step count, total reward
- **Controlled EE**: Currently controlled end-effector (highlighted with ►)
- **Velocity Command**: Current linear (vx, vy) and angular (ω) velocities
- **EE Poses**: Position and orientation of both end-effectors
- **External Wrenches**: Real-time force and torque measurements
  - Force magnitude displayed in red if exceeds 50N
- **Object State**: Joint configuration and link poses
- **Control Help**: Keyboard shortcuts (toggle with H)

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Arrow Keys** | Move controlled end-effector (linear velocity) |
| **↑** | Forward (+x in body frame) |
| **↓** | Backward (-x in body frame) |
| **←** | Left (+y in body frame) |
| **→** | Right (-y in body frame) |
| **Q** | Rotate counterclockwise |
| **W** | Rotate clockwise |
| **1** | Control left end-effector (EE 0) |
| **2** | Control right end-effector (EE 1) |
| **Space** | Reset velocity to zero |
| **R** | Reset environment |
| **H** | Toggle help display |
| **ESC** | Exit demo |

### Automatic Grasp Maintenance

When you control one end-effector:
- The controlled EE follows your keyboard commands
- The other EE automatically maintains its grasp on the object
- This is achieved by computing desired poses based on linkage configuration

## Component Architecture

### 1. Linkage Manager (`gym_biart/envs/linkage_manager.py`)

Manages articulated objects with different joint types:

```python
from gym_biart.envs.linkage_manager import create_two_link_object, JointType

# Create a two-link object
linkage = create_two_link_object(JointType.REVOLUTE)

# Set pymunk bodies
linkage.set_link_body(0, link1_body)
linkage.set_link_body(1, link2_body)

# Update and query state
linkage.update_joint_states()
config = linkage.get_configuration()
poses = linkage.get_all_link_poses()
```

**Features:**
- Forward kinematics
- Joint state tracking
- Automatic grasp pose computation

### 2. PD Controller (`gym_biart/envs/pd_controller.py`)

Converts desired poses to wrench commands:

```python
from gym_biart.envs.pd_controller import MultiGripperController, PDGains

# Create controller
gains = PDGains(kp_linear=30.0, kd_linear=8.0,
                kp_angular=15.0, kd_angular=3.0)
controller = MultiGripperController(num_grippers=2, gains=gains)

# Compute wrenches
wrenches = controller.compute_wrenches(
    current_poses,    # (2, 3) array
    desired_poses,    # (2, 3) array
    current_velocities # (2, 3) array, optional
)
# Returns (2, 3) array with [fx, fy, tau] for each gripper
```

**Features:**
- Proportional-Derivative control
- Body frame operation
- Multi-gripper support

### 3. Keyboard Planner (`gym_biart/envs/keyboard_planner.py`)

High-level teleoperation interface:

```python
from gym_biart.envs.keyboard_planner import MultiEEPlanner

# Create planner
planner = MultiEEPlanner(
    num_end_effectors=2,
    linkage_object=linkage,
    control_dt=0.1
)

# In main loop
desired_poses, actions = planner.update(pygame_events, current_ee_poses)

if actions['quit']:
    break
```

**Features:**
- Keyboard input processing
- Leader-follower control
- Automatic grasp maintenance

## Testing

### Run Unit Tests

```bash
python test_integrated_system.py
```

This tests:
- Linkage manager with all joint types
- PD controller wrench computation
- Keyboard planner state updates

### Run Simple Environment Test

```bash
python test_biart_simple.py
```

## Advanced Usage

### Custom Joint Limits

```python
from gym_biart.envs.linkage_manager import LinkageObject, JointType

linkage = LinkageObject(num_links=2)
linkage.add_joint(
    joint_type=JointType.REVOLUTE,
    parent_link=0,
    child_link=1,
    joint_limits=(-np.pi/2, np.pi/2)  # Custom limits
)
```

### Custom PD Gains

```python
from gym_biart.envs.pd_controller import PDGains

# Stiff control
stiff_gains = PDGains(kp_linear=100.0, kd_linear=20.0,
                      kp_angular=40.0, kd_angular=10.0)

# Soft/compliant control
soft_gains = PDGains(kp_linear=10.0, kd_linear=2.0,
                     kp_angular=5.0, kd_angular=1.0)
```

### Impedance Control

```python
from gym_biart.envs.pd_controller import ImpedanceController

controller = ImpedanceController(
    gains=gains,
    force_threshold=20.0,      # Activate compliance at 20N
    compliance_factor=0.5      # Reduce stiffness by 50%
)

wrench = controller.compute_wrench_with_compliance(
    current_pose,
    desired_pose,
    measured_wrench,
    current_velocity
)
```

## Troubleshooting

### Demo window not showing

Make sure you're not running in a headless environment. If running remotely:
```bash
export MUJOCO_GL=osmesa  # or egl
python demo_teleoperation.py
```

### Import errors

Ensure the package is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/SWIVL"
```

Or install in development mode:
```bash
pip install -e .
```

### Pygame errors

Make sure pygame is properly installed:
```bash
pip install --upgrade pygame
```

## Example Output

When running the demo, you'll see output like:

```
============================================================
BIMANUAL MANIPULATION TELEOPERATION DEMO
============================================================
Joint Type: REVOLUTE
Render Mode: human
============================================================

Creating BiArt environment with revolute joint...
Resetting environment (Episode 1)...
Environment reset complete!

Starting Interactive Demo
...

[Step   50] EE0 | Vel:[ 10.0,  0.0, 0.10] | Wrench:|F|= 15.2 | Reward: 0.856 | Success:False
[Step  100] EE1 | Vel:[  0.0, 10.0, 0.00] | Wrench:|F|= 23.4 | Reward: 0.782 | Success:False
...
```

## Tips for Better Control

1. **Start Slow**: Begin with small velocity commands to get a feel for the system
2. **Switch EEs**: Practice switching between end-effectors to see automatic grasp maintenance
3. **Watch Wrenches**: Monitor external forces to avoid excessive contact
4. **Different Joints**: Try all three joint types to see different behaviors:
   - **Revolute**: Allows rotation between links
   - **Prismatic**: Allows sliding/translation
   - **Fixed**: Rigid connection (testing edge case)

## Contributing

For detailed API documentation, see `USAGE_EXAMPLES.md`.

For issues or improvements, please submit a GitHub issue or pull request.
