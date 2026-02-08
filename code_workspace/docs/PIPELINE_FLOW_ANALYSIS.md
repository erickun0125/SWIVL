# Complete Pipeline Flow Analysis

## Executive Summary

전체 pipeline을 검사한 결과, **High-Level Planner → Trajectory Generator → Low-Level Controller → Physics Engine**의 데이터 흐름이 올바르게 구현되어 있습니다.

**Status:** ✅ All interfaces properly connected  
**Frame Conventions:** ✅ Consistent throughout pipeline (Modern Robotics)  
**Data Flow:** ✅ Complete and verified

---

## Modern Robotics Convention (중요!)

이 코드베이스는 **Modern Robotics (Lynch & Park)** 규약을 따릅니다:

- **Twist:** `V = [ω, vx, vy]ᵀ` (angular velocity **first!**)
- **Wrench:** `F = [τ, fx, fy]ᵀ` (torque **first!**)
- **Screw axis:** `S = [sω, sx, sy]ᵀ` (angular component **first!**)

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. HIGH-LEVEL PLANNER                                           │
│    ┌──────────────────┐                                         │
│    │ FlowMatching /   │                                         │
│    │ Diffusion /      │  Input: observation dict                │
│    │ ACT /            │  Output: desired_poses (2, 3)          │
│    │ Teleoperation    │         [spatial frame]                 │
│    └──────────────────┘                                         │
│           │                                                      │
│           │ desired_poses: (2, 3) - [x, y, θ] for 2 EEs        │
│           │ Frame: Spatial (T_si^des)                           │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAJECTORY GENERATOR                                         │
│    ┌──────────────────┐                                         │
│    │ CubicSpline /    │  Input: start_pose, end_pose, duration │
│    │ MinimumJerk      │  Output: TrajectoryPoint(t)            │
│    └──────────────────┘                                         │
│           │                                                      │
│           │ TrajectoryPoint:                                    │
│           │   - pose: (3,) [x, y, θ] spatial frame             │
│           │   - velocity_spatial: (3,) [vx, vy, ω] spatial     │
│           │   - velocity_body: (3,) [ω, vx_b, vy_b] body ★MR   │
│           │   - acceleration: (3,) [ax, ay, α] spatial          │
│           │   - time: float                                     │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. LOW-LEVEL CONTROLLER                                         │
│    ┌──────────────────┐                                         │
│    │ SE2 Impedance /  │  Input:                                 │
│    │ Screw-Decomposed │   - current_pose: (3,) spatial         │
│    │ Controller       │   - desired_pose: (3,) spatial         │
│    └──────────────────┘   - body_twist_current: (3,) [ω,vx,vy] │
│           │                - body_twist_desired: (3,) [ω,vx,vy]│
│           │                - F_ext: (3,) [τ, fx, fy] body      │
│           │                                                      │
│           │ Output: wrench (3,) [τ, fx, fy] body frame ★MR     │
│           │                                                      │
│           │ Control Law:                                        │
│           │ F = Λ_b·dV_d + C_b·V + η_b + D_d·V_e + K_d·e      │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. PHYSICS ENGINE (Pymunk)                                      │
│    ┌──────────────────┐                                         │
│    │ BiArtEnv         │  Input: wrenches (2, 3) body frame     │
│    │ + EndEffector    │  Apply: F = ma (dynamics)              │
│    │   Manager        │  Output: next state                     │
│    └──────────────────┘                                         │
│           │                                                      │
│           │ State (observation dict):                           │
│           │   - ee_poses: (2, 3) spatial frame                 │
│           │   - ee_velocities: (2,3) point vels [vx,vy,ω]     │
│           │   - ee_body_twists: (2, 3) body [ω, vx_b, vy_b] ★ │
│           │   - link_poses: (2, 3) spatial frame               │
│           │   - external_wrenches: (2, 3) body [τ, fx, fy] ★   │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘

                      [Loop back to HL Planner]
```

---

## Detailed Interface Specifications

### 1. High-Level Planner Interface

#### **Input: Observation Dictionary**

```python
observation: Dict[str, np.ndarray] = {
    'ee_poses': np.ndarray,         # (2, 3) - EE poses [x, y, θ] in spatial frame
    'ee_velocities': np.ndarray,    # (2, 3) - EE point velocities [vx, vy, ω] (NOT twists!)
    'ee_body_twists': np.ndarray,   # (2, 3) - EE body twists [ω, vx_b, vy_b] ★MR
    'link_poses': np.ndarray,       # (2, 3) - Object link poses in spatial frame
    'external_wrenches': np.ndarray # (2, 3) - External wrenches [τ, fx, fy] ★MR
}
```

**Source:** `BiArtEnv.get_obs()` (src/envs/biart.py)

#### **Output: Desired Poses**

```python
desired_poses: np.ndarray  # (2, 3) - Desired EE poses [x, y, θ] in spatial frame
```

---

### 2. Trajectory Generator Interface

#### **Input: Waypoints**

```python
# Constructor
CubicSplineTrajectory(
    waypoints: List[np.ndarray],  # List of [x, y, θ] poses
    times: Optional[List[float]] = None,
    boundary_conditions: str = 'natural'
)

# Or for point-to-point
MinimumJerkTrajectory(
    start_pose: np.ndarray,  # (3,) [x, y, θ] in spatial frame
    end_pose: np.ndarray,    # (3,) [x, y, θ] in spatial frame
    duration: float          # seconds
)
```

**Source:** src/trajectory_generator.py

#### **Output: TrajectoryPoint**

```python
@dataclass
class TrajectoryPoint:
    pose: np.ndarray              # (3,) [x, y, θ] in SPATIAL frame
    velocity_spatial: np.ndarray  # (3,) [vx, vy, ω] in SPATIAL frame (time derivative)
    velocity_body: np.ndarray     # (3,) [ω, vx_b, vy_b] in BODY frame ★MR convention!
    acceleration: np.ndarray      # (3,) [ax, ay, α] in SPATIAL frame
    time: float
```

**Key Implementation:**
```python
# src/trajectory_generator.py - Convert spatial velocity to body twist
velocity_spatial_mr = np.array([omega, vx_spatial, vy_spatial])  # MR convention
velocity_body = spatial_to_body_twist(pose, velocity_spatial_mr)
```

---

### 3. Low-Level Controller Interface

#### **SE2ImpedanceController.compute_control()**

```python
def compute_control(
    self,
    current_pose: np.ndarray,         # (3,) [x, y, θ] SPATIAL frame
    desired_pose: np.ndarray,         # (3,) [x, y, θ] SPATIAL frame
    body_twist_current: np.ndarray,   # (3,) [ω, vx, vy] BODY frame ★MR
    body_twist_desired: np.ndarray,   # (3,) [ω, vx, vy] BODY frame ★MR
    body_accel_desired: Optional[np.ndarray] = None,  # (3,) [α, ax, ay] BODY
    F_ext: Optional[np.ndarray] = None  # (3,) [τ, fx, fy] BODY ★MR
) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
        F_cmd: Control wrench (3,) [τ, fx, fy] in BODY frame ★MR
        info: Dictionary with debugging information
    """
```

#### **SE2ScrewDecomposedImpedanceController.compute_control()**

```python
def compute_control(
    self,
    current_pose: np.ndarray,       # (3,) [x, y, θ] SPATIAL
    desired_pose: np.ndarray,       # (3,) [x, y, θ] SPATIAL
    body_twist_current: np.ndarray, # (3,) [ω, vx, vy] BODY ★MR
    body_twist_desired: np.ndarray, # (3,) [ω, vx, vy] BODY ★MR
    body_accel_desired: Optional[np.ndarray] = None,
    F_ext: Optional[np.ndarray] = None  # (3,) [τ, fx, fy] BODY ★MR
) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
        F_cmd: Control wrench (3,) [τ, fx, fy] in BODY frame ★MR
        info: Dictionary with decomposition info (theta, e_parallel, e_perp, etc.)
    """
```

**Control Law:**
```
F_cmd = Λ_b · dV_d + C_b · V + η_b + D_d · V_e + K_d · e
```

Where:
- `Λ_b`: Task-space inertia matrix (3×3)
- `C_b`: Coriolis matrix (3×3)
- `η_b`: Gravity wrench (typically zero for planar)
- `D_d`: Desired damping matrix (3×3)
- `K_d`: Desired stiffness matrix (3×3)
- `e`: Pose error via log map
- `V_e`: Velocity error

---

### 4. Physics Engine Interface

#### **Input: Wrench Commands**

```python
def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
    """
    Args:
        action: (6,) or (2, 3)
                [left_τ, left_fx, left_fy, right_τ, right_fx, right_fy] ★MR
                All in BODY frame
    """
```

**Source:** BiArtEnv.step() (src/envs/biart.py)

**Internal Processing:**
```python
# Wrench in MR convention: [τ, fx, fy]
wrenches = np.array([action[:3], action[3:]])  # (2, 3)
self.ee_manager.apply_wrenches(wrenches)

# Inside EndEffectorManager.apply_wrench():
tau, fx_body, fy_body = wrench  # MR convention: torque first!
# Transform body forces to world frame, apply
```

#### **Output: Next State**

```python
observation: Dict[str, np.ndarray] = {
    'ee_poses': (2, 3),            # [x, y, θ] Spatial frame
    'ee_velocities': (2, 3),       # [vx, vy, ω] Point velocities (NOT twists!)
    'ee_body_twists': (2, 3),      # [ω, vx_b, vy_b] Body frame ★MR
    'link_poses': (2, 3),          # [x, y, θ] Spatial frame
    'external_wrenches': (2, 3)    # [τ, fx, fy] Body frame ★MR
}
reward: float
terminated: bool
truncated: bool
info: Dict
```

---

## Frame Convention Summary

| Quantity | Frame | Convention |
|----------|-------|------------|
| **Poses** (T_si) | Spatial | [x, y, θ] |
| **Point Velocities** | Spatial | [vx, vy, ω] (NOT twists!) |
| **Body Twists** | Body | [ω, vx, vy] ★MR |
| **Wrenches** (F) | Body | [τ, fx, fy] ★MR |
| **Screw Axes** | Body | [sω, sx, sy] ★MR |

### Key Transformations

**Spatial → Body Twist:**
```python
# src/se2_math.py: spatial_to_body_twist()
def spatial_to_body_twist(pose: np.ndarray, spatial_twist: np.ndarray) -> np.ndarray:
    """
    Args:
        pose: [x, y, theta]
        spatial_twist: [omega, vx_s, vy_s] (MR convention)
    Returns:
        body_twist: [omega, vx_b, vy_b] (MR convention)
    """
    x, y, theta = pose
    omega_s, vx_s, vy_s = spatial_twist
    R = rotation_matrix(theta)
    v_s = np.array([vx_s, vy_s])
    v_b = R.T @ (v_s - np.array([y, -x]) * omega_s)
    return np.array([omega_s, v_b[0], v_b[1]])
```

**Body → World Force Application:**
```python
# src/envs/end_effector_manager.py: apply_wrench()
def apply_wrench(self, wrench: np.ndarray):
    tau, fx_body, fy_body = wrench  # MR convention
    cos_theta = np.cos(self.base_body.angle)
    sin_theta = np.sin(self.base_body.angle)
    fx_world = cos_theta * fx_body - sin_theta * fy_body
    fy_world = sin_theta * fx_body + cos_theta * fy_body
    self.base_body.apply_force_at_world_point((fx_world, fy_world), ...)
    self.base_body.torque += tau
```

---

## Complete Pipeline Example (RL Integration)

### Step-by-Step Execution

**File:** `src/rl_policy/impedance_learning_env.py`

#### **Step 1: Get Observation from Environment**

```python
obs = self.base_env.get_obs()
# obs = {
#     'ee_poses': (2, 3) spatial,
#     'ee_velocities': (2, 3) point velocities [vx, vy, ω] (NOT twists!),
#     'ee_body_twists': (2, 3) body [ω, vx_b, vy_b] ★MR,
#     'link_poses': (2, 3) spatial,
#     'external_wrenches': (2, 3) body [τ, fx, fy] ★MR
# }
```

#### **Step 2: Update Trajectories (Periodic)**

```python
def _update_trajectories(self, obs):
    # Get desired pose chunk from HL policy or create holding trajectory
    for i in range(2):
        self.trajectories[i] = MinimumJerkTrajectory(
            start_pose=obs['ee_poses'][i],    # Current spatial pose
            end_pose=desired_poses[i],        # Desired spatial pose
            duration=self.hl_period           # 1.0s
        )
```

#### **Step 3: Get Trajectory Targets**

```python
def _get_trajectory_targets(self, t):
    desired_poses = []
    desired_twists = []      # Body frame [ω, vx_b, vy_b] ★MR
    desired_accelerations = []  # Body frame

    for i in range(2):
        traj_point = self.trajectories[i].evaluate(t)
        desired_poses.append(traj_point.pose)            # Spatial
        desired_twists.append(traj_point.velocity_body)  # Body ★MR
        # Transform acceleration to body frame
        accel_body = spatial_to_body_acceleration(...)
        desired_accelerations.append(accel_body)

    return np.array(desired_poses), np.array(desired_twists), np.array(desired_accelerations)
```

#### **Step 4: Compute Control Wrenches**

```python
desired_poses, desired_twists, desired_accels = self._get_trajectory_targets(t)

wrenches = []
for i in range(2):
    # Use body twist from observation
    current_body_twist = obs['ee_body_twists'][i]  # [ω, vx_b, vy_b] ★MR
    
    wrench, _ = self.controllers[i].compute_control(
        current_pose=obs['ee_poses'][i],              # (3,) spatial
        desired_pose=desired_poses[i],                # (3,) spatial
        body_twist_current=current_body_twist,        # (3,) body ★MR
        body_twist_desired=desired_twists[i],         # (3,) body ★MR
        body_accel_desired=desired_accels[i],         # (3,) body
        F_ext=obs['external_wrenches'][i]             # (3,) body [τ, fx, fy] ★MR
    )
    wrenches.append(wrench)  # (3,) [τ, fx, fy] body frame ★MR
```

#### **Step 5: Execute in Physics**

```python
wrenches = np.array(wrenches)  # (2, 3) body frame [τ, fx, fy] ★MR
obs, _, terminated, truncated, info = self.base_env.step(wrenches)
```

---

## Verification Checklist

### ✅ Interface Consistency

| Connection | Expected | Actual | Status |
|------------|----------|--------|--------|
| HL Policy → Trajectory | (2, 3) spatial poses | ✓ | ✅ |
| Trajectory → Controller | pose: spatial, twist: body MR | ✓ | ✅ |
| Controller → Physics | (2, 3) body wrenches [τ, fx, fy] | ✓ | ✅ |
| Physics → HL Policy | obs dict with body_twists | ✓ | ✅ |

### ✅ Frame Convention Consistency

| Data | Expected Frame | Convention | Status |
|------|----------------|------------|--------|
| Poses | Spatial | [x, y, θ] | ✅ |
| Body twists | Body | [ω, vx, vy] MR | ✅ |
| Wrenches | Body | [τ, fx, fy] MR | ✅ |
| Screw axes | Body | [sω, sx, sy] MR | ✅ |

---

## Interface Documentation Summary

### Quick Reference Table

| Module | Input | Output | Frame/Convention |
|--------|-------|--------|------------------|
| **HL Planners** | obs dict | (2,3) poses | Spatial |
| **Trajectory** | waypoints | TrajectoryPoint | Mixed (body twist in MR) |
| **SE2ImpedanceController** | poses, body twists | (3,) wrench | Body [τ, fx, fy] MR |
| **ScrewDecomposedController** | poses, body twists, screw | (3,) wrench | Body [τ, fx, fy] MR |
| **BiArtEnv** | (2,3) wrenches [τ, fx, fy] | obs dict | Body wrenches, Mixed obs |

### TrajectoryPoint Field Reference

```python
TrajectoryPoint(
    pose=(3,),              # [x, y, θ] SPATIAL frame
    velocity_spatial=(3,),  # [vx, vy, ω] SPATIAL frame (NOT MR order)
    velocity_body=(3,),     # [ω, vx_b, vy_b] BODY frame ★MR convention
    acceleration=(3,),      # [ax, ay, α] SPATIAL frame
    time=float
)
```

---

## Conclusion

### Summary

**Overall Status: ✅ GOOD**

The pipeline is correctly implemented with Modern Robotics conventions:
- ✅ All interfaces properly connected
- ✅ Frame conventions consistent (MR: [ω, vx, vy] for twist, [τ, fx, fy] for wrench)
- ✅ Data types correct
- ✅ Body twists properly computed via Adjoint map
- ✅ Acceleration feedforward implemented

### Verified Data Flow

```
Observation Dict (BiArtEnv)
    ↓ ee_body_twists: [ω, vx_b, vy_b] ★MR
HL Policy (spatial poses)
    ↓
Trajectory Generator (pose spatial, twist body MR)
    ↓ velocity_body: [ω, vx_b, vy_b] ★MR
Controller (wrench body MR)
    ↓ F_cmd: [τ, fx, fy] ★MR
Physics Engine (applies dynamics)
    ↓
[Loop]
```

**All connections verified and working correctly with Modern Robotics conventions.**

---

**Analysis Date:** 2025-11-27  
**Status:** Complete and Updated for Modern Robotics Convention
