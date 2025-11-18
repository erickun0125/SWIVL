# SE(2) Reference Frame Conventions and Specifications

This document specifies the reference frame conventions used throughout the SWIVL codebase for SE(2) bimanual manipulation.

## Table of Contents
1. [Overview](#overview)
2. [Frame Definitions](#frame-definitions)
3. [Notation](#notation)
4. [Data Flow Through Pipeline](#data-flow-through-pipeline)
5. [Implementation Details](#implementation-details)
6. [Common Pitfalls](#common-pitfalls)

## Overview

The SWIVL framework operates in SE(2), the Special Euclidean group representing 2D rigid body transformations. Proper handling of reference frames is critical for correct control behavior.

### Key Principles
- **Poses** are always expressed in the **spatial (world) frame**
- **Twists (velocities)** can be expressed in either the **spatial frame** or **body frame**
- **Wrenches (forces)** are always expressed in the **body frame**

## Frame Definitions

### Spatial Frame (World Frame) `{s}`
- Fixed inertial reference frame
- Origin at world coordinates (0, 0)
- Does not move with the robot

### Body Frame `{i}` (where i ∈ {l, r} for left/right)
- Attached to each end-effector
- Origin at end-effector center
- Moves with the end-effector

## Notation

Following Modern Robotics notation:

### Poses (Transformations)
- `T_si`: Transformation from body frame {i} to spatial frame {s}
- Represents pose of end-effector i in spatial frame
- In code: `np.ndarray([x, y, theta])` where `(x, y)` is position in spatial frame, `theta` is orientation

### Velocities (Twists)
- `s^V_i`: **Spatial velocity** - velocity of frame {i} expressed in spatial frame {s}
  - Time derivative of pose: `[ẋ, ẏ, θ̇]`
  - In code: often denoted as `velocity_spatial`

- `i^V_i`: **Body twist** - velocity of frame {i} expressed in body frame {i}
  - Related to spatial velocity via: `i^V_i = Ad_{T_si}^{-1} * s^V_i`
  - In code: denoted as `velocity_body` or `twist_body`

### Conversion
```python
# Spatial velocity to body twist
velocity_body = world_to_body_velocity(pose, velocity_spatial)

# Body twist to spatial velocity
velocity_spatial = body_to_world_velocity(pose, velocity_body)
```

### Wrenches (Forces)
- `F_i`: Wrench applied to body frame {i}, expressed in body frame
- Format: `[fx, fy, τ]` where `fx, fy` are forces, `τ` is torque
- **Always in body frame** for impedance control

## Data Flow Through Pipeline

### Complete Pipeline

```
High-Level Policy → Trajectory Generator → Impedance Controller → Robot
```

### Detailed Frame Flow

#### 1. High-Level Policy Output
```python
# Output: Desired poses in SPATIAL frame
desired_poses = hl_policy.get_action(observation)
# Shape: (2, 3) for bimanual
# Frame: T_si^des (spatial frame)
# Format: [[x_left, y_left, theta_left],
#          [x_right, y_right, theta_right]]
```

#### 2. Trajectory Generator

**Input:**
- Current pose: `T_si` (spatial frame)
- Desired pose: `T_si^des` (spatial frame)

**Processing:**
```python
# Interpolate in SE(2)
traj = MinimumJerkTrajectory(start_pose, end_pose, duration)
traj_point = traj.evaluate(t)

# traj_point contains:
# - pose: [x, y, theta] in SPATIAL frame (T_si^des(t))
# - velocity_spatial: [vx, vy, omega] in SPATIAL frame (time derivative)
# - velocity_body: [vx_b, vy_b, omega] in BODY frame (i^V_i^des(t))
```

**Conversion:**
```python
# Spatial velocity (time derivative of pose)
velocity_spatial = np.array([dx/dt, dy/dt, dθ/dt])

# Convert to body twist
velocity_body = world_to_body_velocity(pose, velocity_spatial)
# This implements: i^V_i = Ad_{T_si}^{-1} * s^V_i
```

**Output:**
- Desired pose: `T_si^des(t)` (spatial frame)
- Desired twist: `i^V_i^des(t)` (body frame)

#### 3. Impedance Controller

**Input:**
```python
def compute_wrench(
    current_pose,        # T_si: Current pose in SPATIAL frame [x, y, theta]
    desired_pose,        # T_si^des: Desired pose in SPATIAL frame [x, y, theta]
    measured_wrench,     # External wrench in BODY frame [fx, fy, tau]
    current_velocity,    # s^V_i: Current velocity in SPATIAL frame [vx, vy, omega]
    desired_velocity     # i^V_i^des: Desired velocity in BODY frame [vx_b, vy_b, omega]
)
```

**Processing:**
```python
# 1. Compute pose error in body frame
error_pos_spatial = desired_pose[:2] - current_pose[:2]
R_si = rotation_matrix(current_pose[2])
error_pos_body = R_si.T @ error_pos_spatial

error_angle = normalize_angle(desired_pose[2] - current_pose[2])

# 2. Convert current spatial velocity to body frame
current_twist_body = world_to_body_velocity(current_pose, current_velocity)

# 3. Compute twist error in body frame
error_twist_body = desired_velocity - current_twist_body
# (both are in body frame now)

# 4. Impedance control law in body frame
force_body = K * error_pos_body + D * error_twist_body[:2]
torque = K_angular * error_angle + D_angular * error_twist_body[2]
```

**Output:**
- Wrench command in BODY frame: `[fx, fy, τ]`

#### 4. Robot/Environment

**Input:**
- Wrench in body frame: `[fx, fy, τ]`

**Processing:**
- Robot dynamics integrate the wrench
- Updates pose and velocity

**Output:**
- New pose: `T_si` (spatial frame)
- New velocity: typically in spatial frame

## Implementation Details

### In `trajectory_generator.py`

```python
class TrajectoryPoint:
    pose: np.ndarray              # [x, y, theta] in spatial frame
    velocity_spatial: np.ndarray  # [vx, vy, omega] in spatial frame
    velocity_body: np.ndarray     # [vx_b, vy_b, omega] in body frame
    acceleration: np.ndarray      # [ax, ay, alpha] in spatial frame
```

Key implementation:
```python
# Compute spatial velocity from spline derivatives
vx_spatial = self.spline_x(t_norm, 1) / self.duration
vy_spatial = self.spline_y(t_norm, 1) / self.duration
omega = self.spline_theta(t_norm, 1) / self.duration

velocity_spatial = np.array([vx_spatial, vy_spatial, omega])

# Convert to body twist
velocity_body = world_to_body_velocity(pose, velocity_spatial)
```

### In `task_space_impedance.py`

```python
def compute_wrench(
    current_pose,        # Spatial frame
    desired_pose,        # Spatial frame
    measured_wrench,     # Body frame
    current_velocity,    # Spatial frame (optional)
    desired_velocity     # Body frame (optional)
):
    # Convert current spatial velocity to body frame
    current_twist_body = world_to_body_velocity(current_pose, current_velocity)

    # Compute errors in body frame
    error_twist_body = desired_velocity - current_twist_body

    # Apply impedance law
    wrench_body = K * error_pose + D * error_twist

    return wrench_body  # Body frame
```

### In `impedance_learning_env.py`

```python
# Get trajectory targets
desired_poses, desired_twists = self._get_trajectory_targets()
# desired_poses: (2, 3) in spatial frame
# desired_twists: (2, 3) in body frame

# Compute wrench
wrench = controller.compute_wrench(
    current_pose=obs['ee_poses'][i],        # Spatial frame
    desired_pose=desired_poses[i],          # Spatial frame
    measured_wrench=obs['external_wrenches'][i],  # Body frame
    current_velocity=obs['ee_twists'][i],   # Spatial frame
    desired_velocity=desired_twists[i]      # Body frame
)
```

## Mathematical Background

### SE(2) Lie Group
- Element: `T = [[R, p], [0, 1]]` where `R ∈ SO(2)`, `p ∈ ℝ²`
- Represents transformation from body to spatial frame

### se(2) Lie Algebra
- Element: `ξ = [v, ω]` where `v ∈ ℝ²`, `ω ∈ ℝ`
- Represents velocity/twist

### Adjoint Transformation
Transforms velocities between frames:
```
s^V = Ad_{T_si} * i^V
i^V = Ad_{T_si}^{-1} * s^V
```

Where:
```python
Ad_{T_si} = [[R, J*p],
             [0, 1]]

R: Rotation matrix
p: Position vector
J: 2D perpendicular matrix [[0, -1], [1, 0]]
```

In code:
```python
def world_to_body_velocity(pose, vel_world):
    theta = pose[2]
    R = rotation_matrix(theta)  # 2x2 rotation matrix

    vel_body = np.zeros(3)
    vel_body[:2] = R.T @ vel_world[:2]  # Rotate linear velocity
    vel_body[2] = vel_world[2]          # Angular velocity unchanged

    return vel_body
```

### Body Twist from Pose Trajectory

Given a pose trajectory `T_si(t)`, the body twist is:

```
i^V_i = log(T_si^{-1} * dT_si/dt)
```

For small time steps:
```
i^V_i ≈ log(T_si^{-1} * T_si(t + dt)) / dt
```

Or using the adjoint:
```
i^V_i = Ad_{T_si}^{-1} * s^V_i
```

where `s^V_i = [ẋ, ẏ, θ̇]` is the time derivative of pose.

## Common Pitfalls

### ❌ WRONG: Using spatial velocity for impedance control
```python
# WRONG - mixing frames!
error_vel = desired_velocity_spatial - current_velocity_spatial
force = K * error_pos + D * error_vel  # ← WRONG if error_pos is in body frame
```

### ✅ CORRECT: Convert to body frame first
```python
# CORRECT
current_twist_body = world_to_body_velocity(pose, current_vel_spatial)
desired_twist_body = world_to_body_velocity(desired_pose, desired_vel_spatial)
# OR desired_twist_body is already provided in body frame from trajectory generator

error_twist_body = desired_twist_body - current_twist_body
force_body = K * error_pos_body + D * error_twist_body[:2]
```

### ❌ WRONG: Using time derivative of cubic spline as body twist
```python
# WRONG
vx = spline_x.derivative(t)  # This is spatial velocity!
twist_body = [vx, vy, omega]  # ← WRONG! This is spatial, not body
```

### ✅ CORRECT: Convert spatial velocity to body twist
```python
# CORRECT
vx_spatial = spline_x.derivative(t)
vy_spatial = spline_y.derivative(t)
omega = spline_theta.derivative(t)

velocity_spatial = [vx_spatial, vy_spatial, omega]
pose = [x, y, theta]

velocity_body = world_to_body_velocity(pose, velocity_spatial)
```

### ❌ WRONG: Applying wrench in wrong frame
```python
# WRONG
wrench_spatial = K * error  # Computed in spatial frame
robot.apply_wrench(wrench_spatial)  # ← WRONG if robot expects body frame
```

### ✅ CORRECT: Compute wrench in body frame
```python
# CORRECT
error_body = world_to_body_velocity(pose, error_spatial)
wrench_body = K * error_body
robot.apply_wrench(wrench_body)  # Correct!
```

## Verification Checklist

When implementing or debugging:

- [ ] Poses are in spatial frame
- [ ] Desired twists from trajectory are in body frame
- [ ] Current velocities are converted from spatial to body frame
- [ ] Impedance control computes errors in body frame
- [ ] Output wrenches are in body frame
- [ ] All frame conversions use `world_to_body_velocity` / `body_to_world_velocity`

## References

1. Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press.
   - Chapter 3: Rigid-Body Motions
   - Chapter 8: Open-Chain Dynamics
   - Chapter 11: Force Control

2. Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
   - Chapter 2: Lie Groups and Lie Algebras
   - Chapter 4: Manipulator Kinematics

## Contact

For questions about frame conventions or SE(2) mathematics, please refer to the source code documentation in:
- `src/se2_math.py`: SE(2) mathematics utilities
- `src/trajectory_generator.py`: Trajectory generation with proper frame handling
- `src/ll_controllers/task_space_impedance.py`: Impedance controller implementation
