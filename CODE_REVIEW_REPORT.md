# SWIVL Codebase Comprehensive Review Report

**Date**: 2025-11-18
**Reviewer**: Claude (Automated Code Analysis)
**Status**: Critical Issues Found - Immediate Action Required

---

## Executive Summary

코드베이스 전체 검사 결과, **여러 심각한 문제와 불일치**가 발견되었습니다. 주요 문제는 다음과 같습니다:

1. ✅ **Frame Convention 불일치** - 다양한 controller 간 frame convention 충돌
2. 🚨 **Environment 미구현 기능** - BiartEnv가 RL에 필요한 velocity 정보 미제공
3. 🚨 **External Wrench Sensing 미구현** - 항상 zeros 반환
4. ❌ **Missing Dependencies** - teleoperation.py가 존재하지 않는 linkage_manager import
5. ⚠️ **Controller 중복 및 불일치** - root level controller.py와 src/ll_controllers/ 간 불일치

---

## 🚨 Critical Issues (즉시 수정 필요)

### 1. BiartEnv: EE Velocity 정보 미제공

**File**: `src/envs/biart.py`

**문제**:
```python
# Line 390-411: get_obs() 메서드
def get_obs(self):
    if self.obs_type == "state":
        ee_poses = self.ee_manager.get_poses()  # (2, 3)
        link_poses = self.object_manager.get_link_poses()  # (2, 3)
        external_wrenches = self.ee_manager.get_external_wrenches()  # (2, 3)

        obs = np.concatenate([
            ee_poses[0],
            ee_poses[1],
            link_poses[0],
            link_poses[1],
            external_wrenches[0],
            external_wrenches[1],
        ], dtype=np.float32)

        return obs  # ❌ ee_twists가 없음!
```

**영향**:
- RL environment (`impedance_learning_env.py`)가 `obs['ee_twists']`를 요구함
- Impedance controller가 `current_velocity` 필요
- **현재 코드는 RuntimeError 발생할 것임!**

**해결책**:
```python
def get_obs(self):
    """Get observation dictionary."""
    ee_poses = self.ee_manager.get_poses()
    ee_velocities = self.ee_manager.get_velocities()  # ← 추가!
    link_poses = self.object_manager.get_link_poses()
    external_wrenches = self.ee_manager.get_external_wrenches()

    # Return as dictionary
    return {
        'ee_poses': ee_poses,
        'ee_twists': ee_velocities,  # ← 추가!
        'link_poses': link_poses,
        'external_wrenches': external_wrenches
    }
```

**또한**: observation_space도 수정 필요 (현재 18차원 → 30차원으로 증가)

---

### 2. External Wrench Sensing 완전 미구현

**File**: `src/envs/end_effector_manager.py:242-256`

**문제**:
```python
def compute_external_wrench(self, link_body: Optional[pymunk.Body] = None) -> np.ndarray:
    """
    Compute external wrench from contact forces.

    This is simplified - proper implementation would use collision callbacks.
    """
    # Simplified: return zero for now
    # Proper implementation would accumulate forces from contact callbacks
    return np.zeros(3)  # ❌ 항상 0 반환!

def get_external_wrench(self) -> np.ndarray:
    """Get most recent external wrench measurement."""
    return self.external_wrench.copy()  # ← 이것도 항상 zeros
```

**영향**:
- Impedance control의 핵심 기능 중 하나인 force feedback이 동작하지 않음
- RL policy가 external wrench를 observation으로 사용하는데 항상 0
- Safety reward가 제대로 계산되지 않음

**해결책**:
Pymunk collision callbacks를 사용하여 실제 contact force 축적:
```python
class ParallelGripper:
    def __init__(self, ...):
        # ... 기존 코드 ...

        # Add collision handler for force sensing
        self.contact_forces = []
        handler = space.add_collision_handler(1, 2)  # Gripper vs Object
        handler.begin = self._on_contact_begin
        handler.post_solve = self._on_contact_post_solve
        handler.separate = self._on_contact_separate

    def _on_contact_post_solve(self, arbiter, space, data):
        """Accumulate contact forces."""
        for contact in arbiter.contact_point_set.points:
            # Transform to body frame
            force_world = contact.normal * arbiter.total_impulse.length
            force_body = self._transform_to_body_frame(force_world)
            self.contact_forces.append(force_body)
        return True

    def compute_external_wrench(self) -> np.ndarray:
        """Compute external wrench from accumulated contact forces."""
        if not self.contact_forces:
            return np.zeros(3)

        # Sum all contact forces
        total_force = sum(self.contact_forces)
        wrench = self._force_to_wrench(total_force)

        # Clear accumulated forces
        self.contact_forces = []

        return wrench
```

---

### 3. Missing Dependency: linkage_manager

**File**: `src/hl_planners/teleoperation.py:23`

**문제**:
```python
from .linkage_manager import LinkageObject  # ❌ 파일이 존재하지 않음!
```

**영향**:
- `teleoperation.py` import 시 ModuleNotFoundError 발생
- 모든 teleoperation 기능 동작하지 않음

**해결책**:
1. `linkage_manager.py` 파일 생성 필요
2. 또는 `object_manager.py`의 ArticulatedObject 사용하도록 수정

---

## ⚠️ Major Issues (중요 수정 필요)

### 4. Controller Frame Convention 불일치

**문제**: 3가지 controller가 서로 다른 frame convention 사용

#### A. `pd_controller.py` (Line 71-82)
```python
def compute_wrench(
    current_pose: np.ndarray,        # World frame
    desired_pose: np.ndarray,        # World frame
    desired_velocity: np.ndarray,    # World frame ← spatial!
    desired_acceleration: Optional[np.ndarray] = None,
    current_velocity: Optional[np.ndarray] = None
):
    # ... desired velocity를 body frame으로 변환
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    desired_vel_body = np.array([
        cos_theta * vx_d + sin_theta * vy_d,
        -sin_theta * vx_d + cos_theta * vy_d
    ])
```

#### B. `task_space_impedance.py` (Line 64-87) - ✅ 올바름!
```python
def compute_wrench(
    current_pose: np.ndarray,        # Spatial frame (T_si)
    desired_pose: np.ndarray,        # Spatial frame (T_si^des)
    measured_wrench: np.ndarray,     # Body frame
    current_velocity: Optional[np.ndarray] = None,    # Spatial frame
    desired_velocity: Optional[np.ndarray] = None     # Body frame ← 올바름!
):
```

#### C. `screw_impedance.py` (Line 61-79) - ❌ 문제!
```python
def compute_wrench(
    current_pose: np.ndarray,
    desired_pose: np.ndarray,
    measured_wrench: np.ndarray,
    current_velocity: Optional[np.ndarray] = None  # ← desired_velocity 없음!
):
```

**영향**:
- 코드를 사용하는 사람이 어떤 controller 사용하느냐에 따라 다른 frame으로 데이터 전달해야 함
- 혼란과 버그 발생 가능성

**해결책**:
**모든 controller를 `task_space_impedance.py`의 convention으로 통일**:
- `current_pose`, `desired_pose`: spatial frame
- `current_velocity`: spatial frame
- `desired_velocity`: **body frame**
- `measured_wrench`: body frame
- Output `wrench`: body frame

---

### 5. Root Level controller.py vs src/ll_controllers/ 불일치

**File**: `controller.py` (root level)

**문제**:
- Root level에 `controller.py` 파일이 있는데 이는 이전 버전
- `from se2_math import SE2Math` import하는데 실제로는 `src.se2_math` 사용
- Frame convention이 src/ll_controllers/와 다름
- 혼란 유발

**해결책**:
1. Root level `controller.py` 삭제 또는 deprecated 표시
2. 모든 코드가 `src/ll_controllers/`만 사용하도록 통일

---

## ⚡ Medium Issues (개선 필요)

### 6. BiartEnv: step()에서 velocity tracking 미구현

**File**: `src/envs/biart.py:313-318`

```python
reward_info = self.reward_manager.compute_reward(
    current_ee_poses=current_ee_poses,
    desired_ee_poses=desired_ee_poses,
    current_ee_velocities=np.zeros((2, 3)),  # ❌ Not tracked yet
    desired_ee_velocities=desired_ee_velocities,
    applied_wrenches=applied_wrenches,
    external_wrenches=external_wrenches
)
```

**해결책**:
```python
current_ee_velocities = self.ee_manager.get_velocities()  # ← 실제 velocity 사용
```

---

### 7. Duplicate SE(2) Math Libraries

**문제**:
- `se2_math.py` (root level)
- `src/se2_math.py`
- 두 파일이 같은지 다른지 불명확

**해결책**:
- Root level `se2_math.py` 삭제
- 모든 코드가 `src.se2_math` 사용

---

### 8. RL Environment Observation Frame 불명확

**File**: `src/rl_policy/impedance_learning_env.py`

**문제**:
Documentation에 twist가 어떤 frame인지 명시되지 않음:
```python
# State space (per arm):
# - External wrench (3): [fx, fy, tau]
# - Current pose (3): [x, y, theta]
# - Current twist (3): [vx, vy, omega]  # ← Spatial? Body?
# - Desired pose (3): [x_d, y_d, theta_d]
# - Desired twist (3): [vx_d, vy_d, omega_d]  # ← Spatial? Body?
```

**실제 구현**:
```python
def _get_rl_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
    desired_poses, desired_twists = self._get_trajectory_targets()
    current_twists = obs.get('ee_twists', np.zeros((2, 3)))  # ← Spatial from env

    rl_obs = np.concatenate([
        obs['external_wrenches'].flatten(),  # 6
        obs['ee_poses'].flatten(),  # 6
        current_twists.flatten(),  # 6 ← Spatial!
        desired_poses.flatten(),  # 6
        desired_twists.flatten()  # 6 ← Body!
    ])
```

**해결책**:
Documentation 업데이트:
```python
# State space (per arm):
# - External wrench (3): [fx, fy, tau] in body frame
# - Current pose (3): [x, y, theta] in spatial frame
# - Current twist (3): [vx, vy, omega] in SPATIAL frame  # ← 명시!
# - Desired pose (3): [x_d, y_d, theta_d] in spatial frame
# - Desired twist (3): [vx_d, vy_d, omega_d] in BODY frame  # ← 명시!
```

---

## 📝 Minor Issues (개선 권장)

### 9. PD Controller - Unnecessary Frame Conversion

**File**: `src/ll_controllers/pd_controller.py`

PD controller가 desired_velocity를 spatial frame으로 받아서 매번 body frame으로 변환하는데, 이는 비효율적입니다. trajectory_generator가 이미 body twist를 제공하므로 이를 직접 사용하는 것이 좋습니다.

**현재**:
```python
# Trajectory generator
velocity_body = traj_point.velocity_body  # Body frame

# PD controller에 전달할 때
velocity_spatial = body_to_world_velocity(pose, velocity_body)  # 다시 spatial로 변환

# PD controller 내부
velocity_body = world_to_body_velocity(pose, velocity_spatial)  # 다시 body로 변환
```

**개선**:
PD controller도 task_space_impedance처럼 desired_velocity를 body frame으로 받도록 수정

---

### 10. High-Level Policies Return Format 불일치

**문제**:
- `flow_matching.py`, `diffusion_policy.py`, `act.py` 모두 `get_action()` 메서드가 있음
- 그런데 return format이 명시되지 않음
- Docstring에 frame 정보 없음

**현재**:
```python
def get_action(self, observation: Dict[str, np.ndarray], goal: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns:
        Desired poses (2, 3) for both end-effectors  # ← Frame?
    """
```

**개선**:
```python
def get_action(self, observation: Dict[str, np.ndarray], goal: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns:
        Desired poses (2, 3) for both end-effectors in SPATIAL frame (T_si^des)
    """
```

---

### 11. Trajectory Generator - Acceleration Frame 미명시

**File**: `src/trajectory_generator.py:32`

```python
@dataclass
class TrajectoryPoint:
    pose: np.ndarray                # [x, y, theta] in spatial frame
    velocity_spatial: np.ndarray    # [vx, vy, omega] in spatial frame
    velocity_body: np.ndarray       # [vx_b, vy_b, omega] in body frame
    acceleration: np.ndarray        # [ax, ay, alpha] in spatial frame  # ← 좋음!
    time: float
```

이것은 잘 되어 있지만, acceleration도 body frame 버전이 있으면 좋을 것입니다 (impedance controller에서 feedforward 사용 시).

---

## 🔍 Unimplemented Features (미구현 기능)

### 12. Model-based Dynamics Computation

**File**: `controller.py:294-370`

```python
class RobotDynamics:
    def compute_task_inertia(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """TODO: Implement for specific robot"""
        raise NotImplementedError("Implement for specific robot")

    def compute_coriolis(self, q: np.ndarray, q_dot: np.ndarray, body_twist: np.ndarray) -> np.ndarray:
        """TODO: Implement for specific robot"""
        raise NotImplementedError("Implement for specific robot")

    def compute_gravity(self, q: np.ndarray) -> np.ndarray:
        """TODO: Implement for specific robot"""
        raise NotImplementedError("Implement for specific robot")
```

**상태**: Placeholder만 존재, 실제 구현 없음

**영향**: Model-matching impedance control을 제대로 사용하려면 이것들이 필요함

---

### 13. Screw Impedance Controller - desired_velocity 지원 없음

**File**: `src/ll_controllers/screw_impedance.py:61-67`

```python
def compute_wrench(
    self,
    current_pose: np.ndarray,
    desired_pose: np.ndarray,
    measured_wrench: np.ndarray,
    current_velocity: Optional[np.ndarray] = None  # ← desired_velocity 없음!
) -> np.ndarray:
```

**문제**: Trajectory tracking을 제대로 할 수 없음 (desired velocity 정보 없음)

---

## 📊 Summary Statistics

| Category | Count | Priority |
|----------|-------|----------|
| Critical Issues | 3 | 🚨 Immediate |
| Major Issues | 2 | ⚠️ High |
| Medium Issues | 3 | ⚡ Medium |
| Minor Issues | 4 | 📝 Low |
| Unimplemented | 2 | 🔍 Future |
| **Total** | **14** | |

---

## 🎯 Recommended Action Plan

### Phase 1: Critical Fixes (1-2 days)

1. **Fix BiartEnv observation**
   - Add `ee_twists` to observation
   - Update observation_space dimensions
   - Return dictionary instead of concatenated array

2. **Implement External Wrench Sensing**
   - Add pymunk collision callbacks
   - Accumulate contact forces
   - Transform to body frame

3. **Fix or Remove teleoperation.py**
   - Create linkage_manager.py
   - OR remove dependency on it

### Phase 2: Frame Convention Unification (2-3 days)

4. **Unify all controllers to same convention**
   - Update pd_controller.py
   - Update screw_impedance.py
   - All use: desired_velocity in body frame

5. **Remove duplicate files**
   - Delete root level controller.py
   - Delete root level se2_math.py
   - Update all imports

### Phase 3: Documentation & Polish (1-2 days)

6. **Document all frame conventions**
   - Add frame annotations to all function signatures
   - Update all docstrings
   - Create frame convention guide (already done: SE2_FRAME_CONVENTIONS.md)

7. **Improve observation documentation**
   - Specify frames for all observation components
   - Update RL environment docs

### Phase 4: Feature Implementation (Future)

8. **Implement missing features**
   - External wrench sensing
   - Model-based dynamics
   - Screw impedance desired_velocity support

---

## 📋 Detailed File-by-File Issues

### src/envs/biart.py
- ❌ Missing `ee_twists` in observation
- ❌ Velocity tracking not implemented in step()
- ⚠️ observation_space dimension mismatch
- ✅ Well-structured otherwise

### src/envs/end_effector_manager.py
- ❌ External wrench sensing returns zeros
- ❌ No collision callbacks
- ✅ Gripper mechanics well-implemented

### src/envs/object_manager.py
- ✅ Well-implemented
- ✅ Good SE(2) frame usage
- ✅ Grasping frames properly defined

### src/ll_controllers/task_space_impedance.py
- ✅ Frame conventions correct!
- ✅ Well-documented
- ✅ Proper body twist handling

### src/ll_controllers/pd_controller.py
- ⚠️ Frame convention different from task_space_impedance
- ⚠️ desired_velocity in spatial frame (should be body)
- ✅ Implementation correct otherwise

### src/ll_controllers/screw_impedance.py
- ❌ Missing desired_velocity parameter
- ⚠️ Cannot do trajectory tracking properly
- ⚠️ Frame convention unclear

### src/hl_planners/*.py
- ✅ Flow matching well-implemented
- ✅ Diffusion policy well-implemented
- ✅ ACT well-implemented
- ⚠️ Return frame not documented

### src/hl_planners/teleoperation.py
- ❌ Missing dependency: linkage_manager
- ❌ Will not run

### src/rl_policy/impedance_learning_env.py
- ✅ Well-structured
- ✅ Frame handling correct (after our fixes)
- ⚠️ Observation frame documentation unclear

### src/rl_policy/ppo_impedance_policy.py
- ✅ Well-implemented
- ✅ Good SB3 integration
- ✅ Custom feature extractor

### src/trajectory_generator.py
- ✅ Excellent! (after our fixes)
- ✅ Body twist properly computed
- ✅ Both spatial and body velocities provided

### controller.py (root level)
- ❌ Duplicate/legacy code
- ❌ Import errors (se2_math)
- ⚠️ Should be removed

### se2_math.py (root level)
- ⚠️ Duplicate of src/se2_math.py
- ⚠️ Should be removed

---

## ✅ What's Working Well

1. **SE(2) Math Library** (`src/se2_math.py`)
   - Comprehensive Lie group/algebra operations
   - Well-documented
   - Proper frame transformations

2. **Trajectory Generator** (after our fixes)
   - Correct body twist computation
   - Both spatial and body velocities
   - Good spline interpolation

3. **Task Space Impedance Controller**
   - Correct frame conventions
   - Well-documented
   - Proper implementation

4. **High-Level Policies**
   - Flow Matching: Good implementation
   - Diffusion Policy: Proper DDIM sampling
   - ACT: Good CVAE + Transformer

5. **RL Policy Infrastructure**
   - Good SB3 integration
   - Custom feature extractor
   - Proper reward design

6. **Object Manager**
   - Good articulated object modeling
   - Proper joint constraints
   - Grasping frames well-defined

---

## 📚 Additional Recommendations

### Testing
- Add unit tests for frame conversions
- Add integration tests for full pipeline
- Test external wrench sensing when implemented

### Documentation
- Add architecture diagram
- Document data flow through pipeline
- Create troubleshooting guide

### Code Organization
- Remove all root-level duplicates
- Consolidate under src/
- Clear module hierarchy

---

## 🎓 Conclusion

코드베이스는 **전반적으로 잘 구조화**되어 있지만, **몇 가지 critical한 문제**가 있어서 현재 상태로는 RL 학습이 제대로 동작하지 않을 것입니다.

**가장 시급한 문제**:
1. BiartEnv의 ee_twists 누락
2. External wrench sensing 미구현
3. teleoperation.py의 missing dependency

이 3가지를 먼저 수정하면 기본적인 동작은 가능할 것입니다.

**장기적으로**는 모든 controller의 frame convention을 통일하고, documentation을 개선하는 것이 중요합니다.

---

**Report Generated**: 2025-11-18
**Total Issues Found**: 14
**Critical**: 3 | **Major**: 2 | **Medium**: 3 | **Minor**: 4 | **Unimplemented**: 2
