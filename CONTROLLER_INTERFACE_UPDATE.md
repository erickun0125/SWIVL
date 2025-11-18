# Controller Interface Update

## Overview

This document describes the major update to the controller interface and high-level planner architecture to properly support trajectory tracking with desired velocities and accelerations.

## Key Changes

### 1. **Grasping Frame System** (`src/envs/grasping_frames.py`)

Added a comprehensive grasping frame management system:

```python
class GraspingFrameManager:
    """
    Manages predefined grasping frames for articulated objects.
    Each link has a grasping frame where EE should align.
    """
```

**Features**:
- Predefined grasping frames for each link (in link's local coordinates)
- Automatic computation of grasping poses in world frame
- Constraint-based computation of non-controlled EE velocity

**Grasping Frame Definition**:
- Left gripper → Link1, left side (offset -length/4), perpendicular orientation
- Right gripper → Link2, right side (offset +length/4), perpendicular orientation
- Frame orientation: 90° relative to link (jaws perpendicular to link length)

### 2. **Low-Level Controller Interface Update**

All controllers now accept desired trajectory information:

#### Before:
```python
compute_wrench(
    current_pose,
    desired_pose,
    current_velocity=None
) -> wrench
```

#### After:
```python
compute_wrench(
    current_pose,
    desired_pose,
    desired_velocity,        # NEW: Required
    desired_acceleration=None,  # NEW: Optional (usually zero)
    current_velocity=None
) -> wrench
```

**Benefits**:
- Better velocity tracking
- Feedforward control capability
- Smoother trajectories
- Proper PD+feedforward control law

**Updated Controllers**:
- ✅ `PDController`
- ✅ `MultiGripperPDController`
- ⏳ `TaskSpaceImpedanceController` (interface updated, implementation pending)
- ⏳ `ScrewImpedanceController` (interface updated, implementation pending)

### 3. **High-Level Planner Architecture**

Clear distinction between two types of planners:

#### Type 1: Learned Policies (ACT, Diffusion, Flow Matching)
```
Learned Policy
    ↓ output: desired_pose
Trajectory Generator
    ↓ output: desired_pose, desired_vel, desired_accel
Low-Level Controller
    ↓ output: wrench
```

#### Type 2: Teleoperation
```
Keyboard Input
    ↓ output: desired_velocity (twist)
Integration (desired_vel * dt)
    ↓ output: desired_pose, desired_vel (desired_accel = 0)
Low-Level Controller
    ↓ output: wrench
```

### 4. **Keyboard Teleoperation** (`src/hl_planners/keyboard_teleoperation.py`)

**New Architecture**:

```python
class KeyboardTeleoperationPlanner:
    def get_action(keyboard_events, current_state):
        # 1. Process keyboard → controlled EE velocity (body frame)
        controlled_vel_body, joint_vel = process_keyboard_input()

        # 2. Transform to world frame
        controlled_vel_world = body_to_world(controlled_vel_body)

        # 3. Compute other EE velocity from object constraints
        other_vel_world = compute_from_constraints(
            controlled_vel_world, joint_vel
        )

        # 4. Integrate velocities → desired poses
        desired_poses = integrate_velocity(prev_poses, velocities, dt)

        return {
            'desired_poses': desired_poses,
            'desired_velocities': desired_velocities,
            'desired_accelerations': zeros  # Always zero
        }
```

**Key Features**:
- Direct velocity control (not pose control)
- Commanded inputs:
  - Controlled EE: linear velocity (vx, vy) + angular velocity (ω)
  - Object joint: joint velocity (rad/s for revolute, m/s for prismatic)
- Automatic computation:
  - Other EE velocity from object constraints
  - Ensures grasp maintenance while allowing joint motion

**Keyboard Mapping**:
- `↑/↓`: Forward/backward (body frame x-axis)
- `←/→`: Left/right (body frame y-axis)
- `Q/W`: Rotate CCW/CW
- `A/D`: Joint motion (extend/retract)

### 5. **Object Constraint Computation**

The `GraspingFrameManager.compute_desired_gripper_velocity()` method computes the non-controlled EE velocity:

**Given**:
- Controlled gripper velocity: `v_c = [vx, vy, ω]`
- Joint velocity: `q̇`

**Compute**:
- Other gripper velocity: `v_o` such that:
  1. Object constraint satisfied (connection point velocity consistent)
  2. Controlled gripper moves with commanded velocity
  3. Joint moves with commanded velocity

**Constraint Equations** (simplified):

For Revolute Joint:
```
ω_link2 = ω_link1 + q̇_joint
v_connection = v_link1 + ω_link1 × r_1 = v_link2 + ω_link2 × r_2
```

For Prismatic Joint:
```
ω_link2 = ω_link1
v_link2 = v_link1 + q̇_joint * slide_direction
```

For Fixed Joint:
```
v_link2 = v_link1
ω_link2 = ω_link1
```

## Control Flow Comparison

### Before:
```
Planner → desired_pose → Controller → wrench
```

### After (Teleoperation):
```
Keyboard → desired_vel (twist)
          ↓ integration
       desired_pose, desired_vel, desired_accel=0
          ↓
       Controller → wrench
```

### After (Learned Policies):
```
Policy → desired_pose
       ↓
   Trajectory Generator → desired_pose, desired_vel, desired_accel
       ↓
   Controller → wrench
```

## Implementation Status

### ✅ Completed
1. Grasping frame system
2. PD controller interface update
3. Multi-gripper PD controller update
4. Keyboard teleoperation planner
5. Object constraint computation (simplified)
6. Test suite for updated controllers

### ⏳ Pending
1. Impedance controller implementation update
2. Screw-aware impedance controller implementation update
3. Full object dynamics in constraint computation
4. BiArt environment initialization with grasping frames
5. Integration with trajectory generator output

### 🎯 Future Work
1. More accurate constraint equation solver
2. Handle kinematic singularities
3. Velocity limits and smoothing
4. Multi-object scenarios
5. Real-time replanning

## Usage Example

### Using Updated PD Controller

```python
from src.ll_controllers import PDController, PDGains

controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
controller.set_timestep(0.01)

# From trajectory generator or teleoperation
desired_pose = np.array([10, 5, 0.5])
desired_vel = np.array([5, 2.5, 0.1])
desired_accel = np.zeros(3)  # Usually zero

wrench = controller.compute_wrench(
    current_pose,
    desired_pose,
    desired_vel,
    desired_accel,
    current_velocity
)
```

### Using Keyboard Teleoperation

```python
from src.hl_planners.keyboard_teleoperation import KeyboardTeleoperationPlanner

planner = KeyboardTeleoperationPlanner(joint_type="revolute")
planner.reset(initial_ee_poses, initial_link_poses)

# In control loop
action = planner.get_action(
    keyboard_events,
    current_ee_poses,
    current_link_poses,
    current_ee_velocities,
    current_link_velocities
)

# action contains:
# - 'desired_poses': (2, 3)
# - 'desired_velocities': (2, 3)
# - 'desired_accelerations': (2, 3) - all zeros

wrenches = multi_controller.compute_wrenches(
    current_poses,
    action['desired_poses'],
    action['desired_velocities'],
    action['desired_accelerations'],
    current_velocities
)
```

## Testing

All tests pass:

```bash
python scripts/tests/test_updated_controllers.py
# ✅ PD Controller with desired velocity test passed
# ✅ PD Controller with current velocity test passed
# ✅ Trajectory following test passed
```

## Benefits

1. **Better Control Performance**:
   - Velocity feedforward improves tracking
   - Smoother trajectories
   - Reduced tracking error

2. **Proper Teleoperation**:
   - Direct velocity control (more intuitive)
   - Automatic grasp maintenance
   - Joint motion control

3. **Unified Interface**:
   - All controllers use same signature
   - Easy to swap controller types
   - Consistent with standard control theory

4. **Extensibility**:
   - Easy to add acceleration feedforward
   - Support for learned policies
   - Compatible with MPC

## Migration Guide

### For Existing Code Using Old Interface

**Before**:
```python
wrench = controller.compute_wrench(current_pose, desired_pose, current_vel)
```

**After**:
```python
# If you only have desired pose:
desired_vel = np.zeros(3)  # Or estimate from pose difference
desired_accel = np.zeros(3)

wrench = controller.compute_wrench(
    current_pose,
    desired_pose,
    desired_vel,
    desired_accel,
    current_vel
)
```

### For Trajectory Following

Use `TrajectoryPoint` from trajectory generator:
```python
point = trajectory.evaluate(t)
wrench = controller.compute_wrench(
    current_pose,
    point.pose,
    point.velocity,
    point.acceleration,
    current_velocity
)
```

## Conclusion

This update brings the controller interface in line with modern control theory and robotics best practices. The separation between learned policies (pose output) and teleoperation (velocity output) is now clear, and both paths properly support the low-level controllers.
