# Impedance Controller 상세 분석 리포트

## Executive Summary

현재 SWIVL 프로젝트의 impedance controller들은 **simplified PD-style controllers**로 구현되어 있으며,
**robot dynamics (inertia matrix, Coriolis/centrifugal forces)를 전혀 고려하지 않습니다**.

이는 제대로 된 task-space impedance control이 아니며, 다음과 같은 문제를 야기할 수 있습니다:

- ❌ **비선형 dynamics 보상 없음** → 속도/자세에 따라 성능 변화
- ❌ **Task space inertia 미고려** → Decoupling 안됨
- ❌ **Desired acceleration feedforward 없음** → Trajectory tracking 성능 저하
- ❌ **Model-based compliance 불가능** → 정확한 impedance 달성 어려움

---

## 1. 현재 구현 상태

### 1.1 발견된 Impedance Controllers

현재 코드베이스에 2개의 impedance controller가 있습니다:

#### A. `TaskSpaceImpedanceController` (src/ll_controllers/task_space_impedance.py)

**Control Law:**
```python
F = K_p * error_pose + K_d * error_twist
```

**특징:**
- 단순 PD controller with force feedback
- Robot dynamics 전혀 고려 안함
- Compliance: external force threshold 기반으로 gain scaling

**코드:**
```python
# Line 139-148
force_body = (
    self.gains.kp_linear * error_pos_body +
    self.gains.kd_linear * error_vel_linear
)
torque = (
    self.gains.kp_angular * error_angle +
    self.gains.kd_angular * error_vel_angular
)
```

#### B. `ScrewImpedanceController` (src/ll_controllers/screw_impedance.py)

**Control Law:**
```python
F = K_along * error_along * screw_axis + K_perp * error_perp
```

**특징:**
- Screw axis 따라 분해된 PD control
- Robot dynamics 전혀 고려 안함
- Screw-aware compliance

---

## 2. 제대로 된 Task-Space Impedance Control

### 2.1 이론적 Control Law

**정확한 Impedance Control Law:**

```
F_cmd = Λ(q) * [ẍ_d + K_p * e_x + K_d * ė_x] + μ(q, q̇) + p(q) - F_ext
```

여기서:
- **Λ(q)**: Task space inertia matrix
  ```
  Λ(q) = [J(q) * M(q)^-1 * J(q)^T]^-1
  ```
- **μ(q, q̇)**: Task space Coriolis/centrifugal terms
  ```
  μ = Λ * J̇ * q̇ - Λ * J * M^-1 * C(q, q̇) * q̇
  ```
- **p(q)**: Task space gravity (SE(2)는 0)
- **F_ext**: External wrench
- **M(q)**: Joint space inertia matrix
- **C(q, q̇)**: Joint space Coriolis matrix
- **J(q)**: Jacobian

### 2.2 SE(2)의 특수성

SE(2) planar motion의 경우:

1. **Gravity = 0**: z축 방향이므로 planar motion에 영향 없음
2. **Simple kinematics**: Mobile robot나 planar end-effector
3. **Coriolis 있음**: 회전 중 linear motion이 있으면 발생
4. **Inertia coupling**: Translation-rotation coupling

**SE(2)의 Inertia Matrix (예시 - rigid body):**
```
M = [m    0     0  ]
    [0    m     0  ]
    [0    0     I  ]
```

**Task Space Inertia (동일한 frame):**
```
Λ = M  (SE(2) direct control 경우)
```

---

## 3. 현재 시스템 구조 분석

### 3.1 Robot Model (BiartEnv)

```python
# src/envs/end_effector_manager.py:50-98
base_mass: float = 0.8          # kg
jaw_mass: float = 0.1           # kg (each)
base_inertia = pymunk.moment_for_poly(base_mass, base_verts)
```

**Physical Properties:**
- Base body: 0.8 kg, rectangular (25×8 units)
- Jaws: 0.1 kg each, rectangular (4×20 units)
- Total gripper mass: ~1.0 kg
- Inertia: Pymunk automatically computed

### 3.2 Control Architecture

```
[High-Level Policy]
    ↓ (desired poses)
[Trajectory Generator]
    ↓ (desired pose + twist)
[Impedance Controller] ← ❌ No robot dynamics!
    ↓ (wrench commands)
[Pymunk Physics Engine]  ← ✅ Actual dynamics here
    ↓
[Robot Motion]
```

**문제점:**
- Controller는 "outer-loop" control
- Pymunk이 F = Ma로 실제 dynamics 계산
- Controller가 dynamics 모르면 performance degradation

---

## 4. 심각한 문제점 및 영향

### 4.1 Critical Issues

#### Issue 1: No Task Space Inertia Matrix (Λ)

**문제:**
```python
# 현재 (WRONG):
F = Kp * e + Kd * de

# 올바른 (CORRECT):
F = Λ * (Kp * e + Kd * de) + μ
```

**영향:**
- Gripper의 mass와 inertia가 달라도 같은 gain 사용
- 실제 impedance가 desired impedance와 다름
- Decoupling 안됨 (translation-rotation coupling)

**예시:**
- Desired stiffness: K = 50 N/m
- Actual stiffness: K_actual = K / (m + coupling) ≠ K

#### Issue 2: No Coriolis/Centrifugal Compensation

**문제:**
```python
# 현재: Coriolis term 없음
F = Kp * e + Kd * de

# 올바른:
F = Λ * ẍ_d + μ(q, q̇) + ...
```

**영향:**
- 빠른 회전 중 linear motion 시 drift 발생
- Trajectory tracking error 증가
- Velocity-dependent behavior

**SE(2) Coriolis 예시:**
```
회전하면서 전진 시: F_coriolis ⊥ velocity
→ 원하지 않는 lateral force 발생
```

#### Issue 3: No Desired Acceleration Feedforward

**문제:**
```python
# 현재: ẍ_d feedforward 없음
F = Kp * (x_d - x) + Kd * (ẋ_d - ẋ)

# 올바른:
F = Λ * ẍ_d + Kp * e + Kd * de + ...
```

**영향:**
- Dynamic trajectory tracking 성능 저하
- Phase lag 발생
- Tracking error proportional to ẍ_d

#### Issue 4: Gain Tuning 불가능

**문제:**
- Robot dynamics를 모르므로 이론적 gain 설정 불가
- Trial-and-error로만 tuning 가능
- Robot parameter 변경 시 re-tuning 필요

**영향:**
- Impedance parameters (M_d, D_d, K_d)가 desired와 다름
- RL policy가 학습한 impedance가 부정확

---

### 4.2 RL Impedance Learning에 미치는 영향

현재 RL policy는 impedance parameters (D, K)를 학습합니다:

```python
# src/rl_policy/impedance_learning_env.py
action: [D_linear_x, D_linear_y, D_angular, K_linear_x, K_linear_y, K_angular]
```

**문제:**
1. **Learned parameters ≠ Actual impedance**
   - Controller가 dynamics 모르므로 실제 impedance 다름
   - RL이 compensation 위해 왜곡된 값 학습할 수 있음

2. **Generalization 어려움**
   - Object mass/inertia 변경 시 policy 재학습 필요
   - Dynamics가 controller에 없으므로 adaptation 불가

3. **Sample efficiency 저하**
   - RL이 implicit dynamics compensation까지 학습해야 함
   - 본질적 impedance learning에 집중 못함

---

## 5. 현재 코드의 누락된 구성 요소

### 5.1 Robot Dynamics Model

**없는 것들:**
- ❌ Mass matrix M(q)
- ❌ Coriolis matrix C(q, q̇)
- ❌ Jacobian J(q)
- ❌ Task space inertia Λ(q)
- ❌ Task space Coriolis μ(q, q̇)

**Search 결과:**
```bash
$ grep -r "jacobian\|inertia.*matrix\|Lambda" src/
# No results (robot dynamics functions 없음)
```

### 5.2 SE(2) Kinematics/Dynamics Library

**se2_math.py에 없는 것:**
```python
# 없음:
def compute_jacobian(pose)
def compute_inertia_matrix(robot_params)
def compute_task_space_inertia(M, J)
def compute_coriolis_matrix(pose, velocity)
def transform_wrench_to_task_space(...)
```

**있는 것:**
```python
# 기본적인 SE(2) math만:
- SE2Pose class
- se2_log, se2_exp
- world_to_body_velocity
- body_to_world_velocity
```

---

## 6. SE(2) Direct Control의 특수 케이스

### 6.1 왜 Simplified Controller가 작동할 수 있는가?

현재 시스템은 **mobile robot과 유사**합니다:

```
Mobile Robot:
- Direct SE(2) control (no arm kinematics)
- Simple dynamics: M = diag(m, m, I)
- No joint space → Task space = Configuration space
```

**이 경우:**
1. Jacobian = Identity (J = I)
2. Task space inertia = Joint space inertia (Λ = M)
3. Simple Coriolis (여전히 있음!)

**따라서:**
- Λ를 상수로 근사 가능 → Gain에 흡수
- 하지만 여전히 Coriolis 보상 필요!

### 6.2 현재 구현이 "작동"하는 이유

1. **Low velocity regime**
   - Coriolis forces 작음
   - Dynamics effects 무시 가능

2. **Pymunk damping**
   ```python
   self.space.damping = 0.99  # High damping
   ```
   - Implicit stability
   - Dynamics effects 감쇠

3. **Simple trajectories**
   - Slow motion
   - Low acceleration

4. **Stiff control**
   - High gains → Dynamics effects 상대적으로 작음

**하지만:**
- Fast motion에서는 실패 가능
- Precise impedance control 불가능
- RL learning efficiency 저하

---

## 7. 권장 수정 사항

### 7.1 Priority 1: Add Task Space Inertia

**구현:**
```python
class TaskSpaceImpedanceController:
    def __init__(self, robot_params):
        self.mass = robot_params.mass          # m (base)
        self.inertia = robot_params.inertia    # I (base)

    def compute_wrench(self, ...):
        # Task space inertia (SE(2) direct control)
        Lambda = np.diag([self.mass, self.mass, self.inertia])

        # Impedance control law
        ddx_d = ...  # desired acceleration
        wrench = Lambda @ (ddx_d + Kp * e + Kd * de)

        return wrench
```

**난이도:** Medium
**영향:** High - Proper impedance scaling

### 7.2 Priority 2: Add Coriolis Compensation

**SE(2) Coriolis for rigid body:**
```python
def compute_coriolis_se2(velocity, mass, inertia):
    """
    Coriolis matrix for SE(2) rigid body.

    For SE(2): C(q, q̇) accounts for rotation-translation coupling
    """
    vx, vy, omega = velocity

    # Coriolis/centrifugal forces
    C = np.array([
        [0,           -mass * omega,  0],
        [mass * omega, 0,             0],
        [0,            0,             0]
    ])

    mu = C @ velocity  # Coriolis wrench in body frame
    return mu
```

**난이도:** Medium
**영향:** High - Velocity-dependent accuracy

### 7.3 Priority 3: Add Desired Acceleration Feedforward

**구현:**
```python
def compute_wrench(self, current_pose, desired_pose,
                   current_velocity, desired_velocity,
                   desired_acceleration,  # ← ADD THIS
                   measured_wrench):
    # ...

    # Feedforward term
    ddx_d = desired_acceleration  # From trajectory generator

    wrench = Lambda @ (ddx_d + Kp * e + Kd * de) + mu - measured_wrench

    return wrench
```

**난이도:** Low (Trajectory generator already computes acceleration)
**영향:** Medium - Better tracking

### 7.4 Priority 4: Complete Robot Dynamics Module

**새 파일 생성:** `src/se2_dynamics.py`

```python
"""
SE(2) Robot Dynamics

Provides robot dynamics computation for SE(2) systems:
- Inertia matrices
- Coriolis/centrifugal forces
- Task space transformations
"""

@dataclass
class SE2RobotParams:
    """SE(2) robot physical parameters."""
    mass: float              # Total mass [kg]
    inertia: float           # Rotational inertia [kg⋅m²]
    base_width: float        # For distributed mass
    base_height: float

class SE2Dynamics:
    """SE(2) robot dynamics computations."""

    def __init__(self, params: SE2RobotParams):
        self.params = params

    def get_mass_matrix(self) -> np.ndarray:
        """Get configuration space mass matrix."""
        return np.diag([
            self.params.mass,
            self.params.mass,
            self.params.inertia
        ])

    def get_task_space_inertia(self, pose: np.ndarray) -> np.ndarray:
        """Get task space inertia (SE(2) direct control: same as M)."""
        return self.get_mass_matrix()

    def compute_coriolis_wrench(self,
                                velocity: np.ndarray) -> np.ndarray:
        """Compute Coriolis/centrifugal wrench in body frame."""
        vx, vy, omega = velocity
        m = self.params.mass

        # SE(2) Coriolis forces
        return np.array([
            -m * omega * vy,   # Centrifugal in x
            m * omega * vx,    # Centrifugal in y
            0.0                # No torque from linear velocity
        ])

    def gravity_wrench(self, pose: np.ndarray) -> np.ndarray:
        """Gravity wrench (zero for SE(2) planar)."""
        return np.zeros(3)
```

**난이도:** Medium
**영향:** Very High - Foundation for proper control

---

## 8. 검증 방법

### 8.1 Unit Tests

**새 파일:** `test_impedance_dynamics.py`

```python
def test_inertia_matrix_scaling():
    """Test that different masses result in different control."""
    controller_light = TaskSpaceImpedanceController(
        robot_params=SE2RobotParams(mass=0.5, inertia=0.01)
    )
    controller_heavy = TaskSpaceImpedanceController(
        robot_params=SE2RobotParams(mass=2.0, inertia=0.1)
    )

    # Same error, different wrenches due to Lambda scaling
    wrench_light = controller_light.compute_wrench(...)
    wrench_heavy = controller_heavy.compute_wrench(...)

    assert np.linalg.norm(wrench_heavy) > np.linalg.norm(wrench_light)

def test_coriolis_compensation():
    """Test Coriolis compensation during rotation."""
    # Rotate gripper while moving forward
    velocity = np.array([1.0, 0.0, 1.0])  # Forward + rotating

    wrench_with_comp = controller.compute_wrench(..., compensate=True)
    wrench_without = controller.compute_wrench(..., compensate=False)

    # Should see difference in lateral force
    assert abs(wrench_with_comp[1]) < abs(wrench_without[1])
```

### 8.2 Integration Tests

```python
def test_trajectory_tracking_accuracy():
    """Test tracking of dynamic trajectories."""
    # Generate fast circular trajectory
    trajectory = generate_circular_trajectory(radius=50, speed=100)

    # Run with dynamics-aware controller
    errors_with_dynamics = run_tracking(controller_v2, trajectory)

    # Run with simplified controller
    errors_simplified = run_tracking(controller_v1, trajectory)

    # Dynamics-aware should be better
    assert np.mean(errors_with_dynamics) < np.mean(errors_simplified)
```

---

## 9. Migration Path

### Phase 1: Add Robot Parameters (1-2 hours)

1. Create `SE2RobotParams` dataclass
2. Extract params from EndEffectorManager
3. Pass params to controllers

### Phase 2: Add Inertia Matrix (2-3 hours)

1. Implement `SE2Dynamics.get_task_space_inertia()`
2. Update TaskSpaceImpedanceController
3. Test with different masses

### Phase 3: Add Coriolis Compensation (3-4 hours)

1. Implement `SE2Dynamics.compute_coriolis_wrench()`
2. Add to control law
3. Test with rotating trajectories

### Phase 4: Add Acceleration Feedforward (1-2 hours)

1. Update TrajectoryGenerator to provide acceleration
2. Add ddx_d parameter to controller
3. Update all call sites

### Phase 5: Integration & Testing (4-6 hours)

1. End-to-end tests
2. RL training comparison
3. Performance benchmarks

**Total effort:** 11-17 hours

---

## 10. 결론

### 10.1 현재 상태

현재 SWIVL의 impedance controller는:
- ✅ **구현됨**: Basic PD control with force feedback
- ✅ **작동함**: Low-speed, simple trajectories
- ❌ **정확하지 않음**: No robot dynamics
- ❌ **이론적 근거 부족**: Not true impedance control
- ❌ **확장성 낮음**: Robot params 변경 시 문제

### 10.2 핵심 문제

1. **No Task Space Inertia Matrix (Λ)**
   - 가장 심각한 문제
   - Impedance scaling 틀림

2. **No Coriolis Compensation (μ)**
   - Fast motion에서 drift
   - Velocity-dependent errors

3. **No Acceleration Feedforward (ẍ_d)**
   - Trajectory tracking lag
   - Dynamic performance 저하

### 10.3 권장 조치

**즉시 수정 필요:**
- Priority 1-2 (Inertia + Coriolis) 구현
- SE2Dynamics module 생성
- Unit tests 추가

**개선 효과:**
- ✅ Accurate impedance control
- ✅ Better trajectory tracking
- ✅ RL learning efficiency
- ✅ Generalization to different robots
- ✅ Theoretical foundation

**투자 대비 효과:**
- Effort: ~15 hours
- Impact: 현재 controller의 근본적 한계 해결
- ROI: Very High (특히 RL impedance learning)

---

## Appendix A: SE(2) Dynamics 수식 정리

### A.1 Configuration Space Dynamics

```
M(q) ⋅ q̈ + C(q, q̇) ⋅ q̇ + G(q) = τ
```

For SE(2):
```
q = [x, y, θ]ᵀ
M = diag(m, m, I)
G = [0, 0, 0]ᵀ  (planar)
```

### A.2 Task Space Dynamics

```
Λ(q) ⋅ ẍ + μ(q, q̇) + p(q) = F
```

Where:
```
Λ = [J ⋅ M⁻¹ ⋅ Jᵀ]⁻¹
μ = Λ ⋅ J̇ ⋅ q̇ - Λ ⋅ J ⋅ M⁻¹ ⋅ C ⋅ q̇
p = Λ ⋅ J ⋅ M⁻¹ ⋅ G
```

For SE(2) direct control (J = I):
```
Λ = M
μ = -M ⋅ M⁻¹ ⋅ C ⋅ q̇ = -C ⋅ q̇
p = 0
```

### A.3 Impedance Control Law

```
F = Λ ⋅ (ẍd + Kp⋅e + Kd⋅ė) + μ + p - Fext
```

Simplifies to:
```
F = M ⋅ (ẍd + Kp⋅e + Kd⋅ė) - C⋅q̇ - Fext
```

### A.4 Coriolis Matrix for SE(2) Rigid Body

```
C(q, q̇) = [0      -m⋅ω   0]
          [m⋅ω    0      0]
          [0      0      0]

C ⋅ q̇ = [−m⋅ω⋅vy]
        [m⋅ω⋅vx ]
        [0      ]
```

Where ω = dθ/dt (angular velocity).

---

## Appendix B: 관련 파일 목록

**Controller Files:**
- `src/ll_controllers/task_space_impedance.py` - 수정 필요
- `src/ll_controllers/screw_impedance.py` - 수정 필요
- `src/ll_controllers/pd_controller.py` - 참고용

**Math Library:**
- `src/se2_math.py` - Dynamics functions 추가 필요

**Robot Model:**
- `src/envs/end_effector_manager.py` - Physical params 추출

**RL Integration:**
- `src/rl_policy/impedance_learning_env.py` - Controller 사용

**Documentation:**
- `SE2_FRAME_CONVENTIONS.md` - Frame 규약
- `HL_RL_PIPELINE_README.md` - Pipeline 설명

---

**Report Generated:** 2025-11-18
**Analysis Focus:** SE(2) Impedance Controller & Robot Dynamics
**Severity:** HIGH - Fundamental control law issues
