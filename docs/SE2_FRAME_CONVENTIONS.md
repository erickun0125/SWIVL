# SE(2) Frame Conventions

## Overview

This document describes the frame conventions used throughout the SWIVL codebase. We follow **Modern Robotics (Lynch & Park)** conventions for all twist, wrench, and screw representations.

## Modern Robotics Convention (Critical!)

### Twist (se(2) element)
```
V = [ω, vx, vy]ᵀ
```
- **Angular velocity first!**
- `ω`: Angular velocity (rad/s)
- `vx`, `vy`: Linear velocity components (m/s or pixels/s)

### Wrench (se(2)* element)
```
F = [τ, fx, fy]ᵀ
```
- **Torque first!**
- `τ`: Torque (N·m)
- `fx`, `fy`: Force components (N)

### Screw Axis
```
S = [sω, sx, sy]ᵀ
```
- **Angular component first!**
- For revolute: `[1, ry, -rx]` (unit angular velocity, r = position to joint)
- For prismatic: `[0, vx, vy]` (unit linear velocity along axis)

---

## Coordinate Frames

### Spatial (World) Frame {s}
- Fixed inertial frame
- Origin at (0, 0) of the workspace
- X-axis pointing right, Y-axis pointing up
- All **poses** are expressed in this frame: `[x, y, θ]`

### Body Frame {b}
- Attached to the rigid body (end-effector or link)
- Origin at the body's reference point (e.g., gripper center)
- X-axis aligned with body's forward direction
- All **twists** and **wrenches** in control are expressed in this frame

### Object Frame {o}
- Attached to the articulated object
- Used for defining grasping frames and joint axes

---

## Pose Representation

### SE(2) Pose
```python
pose = [x, y, θ]
```
- `x`, `y`: Position in spatial frame (pixels)
- `θ`: Orientation angle (radians)

### Homogeneous Transformation Matrix
```python
T = [[cos(θ), -sin(θ), x],
     [sin(θ),  cos(θ), y],
     [0,       0,      1]]
```

### SE2Pose Dataclass
```python
from src.se2_math import SE2Pose

pose = SE2Pose(x=100.0, y=200.0, theta=np.pi/4)
T = pose.to_matrix()  # 3x3 transformation matrix
arr = pose.to_array()  # [x, y, theta] array
```

---

## Velocity Conventions

### Spatial Frame Velocity (Time Derivative)
```python
velocity_spatial = [vx_s, vy_s, ω]
```
- Direct time derivative of pose: `d[x, y, θ]/dt`
- Note: This is **NOT** in MR order (angular last)

### Body Frame Twist (MR Convention)
```python
body_twist = [ω, vx_b, vy_b]
```
- **MR convention: angular first!**
- Velocity of body frame origin expressed in body frame
- Related to spatial velocity via Adjoint transformation

### Conversion Functions
```python
from src.se2_math import spatial_to_body_twist, body_to_spatial_twist

# Spatial twist (MR order: [ω, vx_s, vy_s]) to body twist
body_twist = spatial_to_body_twist(pose, spatial_twist_mr)

# Body twist to spatial twist (MR order)
spatial_twist_mr = body_to_spatial_twist(pose, body_twist)
```

---

## Wrench Conventions

### Body Frame Wrench (MR Convention)
```python
wrench = [τ, fx, fy]
```
- **MR convention: torque first!**
- `τ`: Torque about body frame z-axis (N·m)
- `fx`, `fy`: Force in body frame (N)

### Wrench Transformation
```python
from src.se2_math import wrench_body_to_spatial, wrench_spatial_to_body

# Transform wrench between frames
wrench_spatial = wrench_body_to_spatial(pose, wrench_body)
wrench_body = wrench_spatial_to_body(pose, wrench_spatial)
```

---

## Adjoint Representation

### Adjoint Matrix Ad_T
For T ∈ SE(2) with rotation R and translation p = [px, py]:
```
Ad_T = [ 1     0      0   ]
       [ py   R11    R12  ]
       [-px   R21    R22  ]
```

Transforms twists from body to spatial frame:
```
V_spatial = Ad_T · V_body
```

### Inverse Adjoint
```
V_body = Ad_{T^{-1}} · V_spatial
```

### Implementation
```python
from src.se2_math import se2_adjoint

T = SE2Pose(x, y, theta).to_matrix()
Ad = se2_adjoint(T)  # 3x3 Adjoint matrix
```

---

## Lie Algebra Operations

### Exponential Map (se(2) → SE(2))
```python
from src.se2_math import se2_exp

xi = np.array([omega, vx, vy])  # se(2) element (MR order)
T = se2_exp(xi)  # SE(2) transformation matrix
```

### Logarithm Map (SE(2) → se(2))
```python
from src.se2_math import se2_log

T = SE2Pose(x, y, theta).to_matrix()
xi = se2_log(T)  # se(2) element [omega, vx, vy] (MR order)
```

### Pose Error Computation
```python
# Error in body frame via log map
T_current = SE2Pose.from_array(current_pose).to_matrix()
T_desired = SE2Pose.from_array(desired_pose).to_matrix()
T_error = se2_inverse(T_current) @ T_desired
error = se2_log(T_error)  # [omega_e, vx_e, vy_e] in body frame
```

---

## Screw Decomposition

### Screw Axis Definition
For an articulated joint, the screw axis describes the allowed motion:

**Revolute Joint:**
```python
# Unit screw for revolute joint
# ω = 1 (unit angular velocity)
# v = ω × r where r is position from body to joint axis
B_revolute = np.array([1.0, ry, -rx])  # [sω, sx, sy] MR order
```

**Prismatic Joint:**
```python
# Unit screw for prismatic joint
# ω = 0 (no rotation)
# v = unit direction of sliding
B_prismatic = np.array([0.0, vx, vy])  # [sω, sx, sy] MR order
```

### Projection Operators
```python
# Parallel projection
P_parallel = (S @ S.T) / (S.T @ S)

# Perpendicular projection
P_perpendicular = I - P_parallel
```

---

## Integration

### Twist Integration on SE(2)
```python
from src.se2_math import integrate_twist

# Proper SE(2) integration using exponential map
new_pose = integrate_twist(current_pose, body_twist, dt)
```

### Velocity Integration (Simple)
```python
from src.se2_math import integrate_velocity

# Integrate body twist over timestep
new_pose = integrate_velocity(current_pose, body_twist, dt)
```

---

## Common Pitfalls

### 1. Wrong Order
```python
# WRONG (old convention)
twist = [vx, vy, omega]

# CORRECT (Modern Robotics)
twist = [omega, vx, vy]
```

### 2. Wrong Frame
```python
# WRONG: Using spatial velocity as body velocity
wrench = D @ spatial_velocity  # Error!

# CORRECT: Convert first, then use body velocity
body_velocity = spatial_to_body_twist(pose, spatial_velocity_mr)
wrench = D @ body_velocity
```

### 3. Missing Adjoint Transform
```python
# WRONG: Simple rotation
v_body = R.T @ v_spatial  # Ignores p × ω term!

# CORRECT: Full Adjoint transformation
v_body = spatial_to_body_twist(pose, v_spatial_mr)
```

---

## Summary Table

| Quantity | Order | Frame | Example |
|----------|-------|-------|---------|
| Pose | [x, y, θ] | Spatial | `[100, 200, 0.5]` |
| Body Twist | [ω, vx, vy] | Body | `[0.1, 10, 5]` |
| Spatial Velocity | [vx, vy, ω] | Spatial | `[10, 5, 0.1]` (NOT MR order) |
| Wrench | [τ, fx, fy] | Body | `[5.0, 20, 10]` |
| Screw (revolute) | [1, ry, -rx] | Body | `[1.0, 5, -10]` |
| Screw (prismatic) | [0, vx, vy] | Body | `[0.0, 1, 0]` |

---

## References

1. Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press.
2. Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.

---

**Last Updated:** 2025-11-27  
**Convention:** Modern Robotics (Lynch & Park)
