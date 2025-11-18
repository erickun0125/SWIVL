# Code-to-Paper Mapping for SWIVL

This document maps the mathematical formulations in the Method section to the actual code implementation.

## Section 3.1: Problem Formulation

### SE(2) Representation
- **Paper**: $T_{si} \in \text{SE}(2)$ with $(x, y, \theta)$ parametrization
- **Code**: `src/se2_math.py`
  - Pose stored as `np.array([x, y, theta])`
  - Transformation matrices constructed via `se2_math.pose_to_matrix()`

### Body and Spatial Frames
- **Paper**: Body twist ${}^i V_i$, Spatial twist ${}^s V_i$
- **Code**: `src/se2_math.py`
  - `body_to_world_velocity()`: Converts body frame to spatial frame
  - `world_to_body_velocity()`: Converts spatial frame to body frame
  - `world_to_body_acceleration()`: Frame transformation for accelerations

### Wrench Representation
- **Paper**: ${}^i F_i = [f_x^b, f_y^b, \tau]^T$
- **Code**: Environment observation
  - `obs['external_wrenches']`: Shape (2, 3) for bimanual
  - Measured in body frame from F/T sensors

### Screw Axis
- **Paper**: Unit screw $B_i \in \mathfrak{se}(2)$
- **Code**: `src/envs/object_manager.py`
  - `get_joint_axis_screws()`: Returns $(B_L, B_R)$
  - Configuration-invariant representation in body frame

## Section 3.2: SE(2) Task Space Impedance Control

### Pose Error (Eq. 1)
```python
# src/ll_controllers/se2_impedance_controller.py: line ~180
e = self._compute_pose_error(current_pose, desired_pose)
# Computes log(T_current^-1 @ T_desired)
```

### Model Matching (Eq. 2)
```python
# src/se2_dynamics.py: line ~150
Lambda_b = self.compute_task_space_inertia(current_pose)
# Lambda_b = (J_b M^-1 J_b^T)^-1
```

### Impedance Control Law (Eq. 3)
```python
# src/ll_controllers/se2_impedance_controller.py: line ~200-220
wrench = (Lambda_b @ desired_accel_body +
          C_b @ current_twist_body +
          eta_b +
          D_d @ velocity_error +
          K_d @ pose_error -
          measured_wrench)
```

**File**: `src/ll_controllers/se2_impedance_controller.py`
- Class: `SE2ImpedanceController`
- Method: `compute_wrench()`

### Diagonal Impedance Matrices
```python
# src/ll_controllers/task_space_impedance.py: line ~50-70
class ImpedanceGains:
    kp_linear: float = 50.0   # K_x, K_y
    kd_linear: float = 10.0   # D_x, D_y
    kp_angular: float = 20.0  # K_theta
    kd_angular: float = 5.0   # D_theta
```

## Section 3.3: Screw-Decomposed Impedance Control

### Screw Subspace Decomposition (Eq. 4-5)
```python
# src/ll_controllers/se2_screw_decomposed_impedance.py: line ~250-270
def _decompose_twist(self, twist):
    # Parallel component: (B · V) B
    parallel_magnitude = np.dot(self.screw_axis, twist)
    V_parallel = parallel_magnitude * self.screw_axis

    # Perpendicular component: V - V_parallel
    V_perpendicular = twist - V_parallel

    return V_parallel, V_perpendicular
```

### Directional Impedance Parameters (Eq. 6)
```python
# src/ll_controllers/se2_screw_decomposed_impedance.py: line ~30-50
@dataclass
class ScrewImpedanceParams:
    M_parallel: float        # Inertia parallel
    D_parallel: float        # Damping parallel (compliant)
    K_parallel: float        # Stiffness parallel (compliant)

    M_perpendicular: float   # Inertia perpendicular
    D_perpendicular: float   # Damping perpendicular (stiff)
    K_perpendicular: float   # Stiffness perpendicular (stiff)
```

### Screw-Decomposed Control Law (Eq. 7)
```python
# src/ll_controllers/se2_screw_decomposed_impedance.py: line ~300-330
# Parallel impedance wrench
wrench_parallel = (D_parallel * V_error_parallel +
                   K_parallel * e_parallel)

# Perpendicular impedance wrench
wrench_perpendicular = (D_perpendicular * V_error_perpendicular +
                        K_perpendicular * e_perpendicular)

# Total wrench with dynamics compensation
wrench_cmd = (Lambda_b @ desired_accel +
              C_b @ current_twist +
              eta_b +
              wrench_parallel +
              wrench_perpendicular -
              measured_wrench)
```

**File**: `src/ll_controllers/se2_screw_decomposed_impedance.py`
- Class: `SE2ScrewDecomposedImpedanceController`
- Method: `compute_control()`

## Section 3.4: Reference Motion Field

### Motion Field Equation (Eq. 8)
The desired acceleration field is implicitly defined by the impedance control law.

```python
# Desired acceleration is computed from trajectory generator
# src/trajectory_generator.py: line ~100-120
class TrajectoryPoint:
    pose: np.ndarray              # T_si^d
    velocity_body: np.ndarray     # i^V_i^d
    velocity_spatial: np.ndarray  # s^V_i^d
    acceleration: np.ndarray      # Acceleration in spatial frame
```

### Stability via Passivity
Model matching (M_d = Lambda_b) ensures passivity. This is guaranteed in:
```python
# src/ll_controllers/se2_impedance_controller.py: line ~150
# Always uses Lambda_b from robot dynamics
M_d = Lambda_b  # Model matching for passivity
```

## Section 3.5: RL Impedance Variable Learning

### State Space (Eq. 9)
```python
# src/rl_policy/impedance_learning_env.py: line ~430-440
def _get_rl_observation(self, obs):
    rl_obs = np.concatenate([
        obs['external_wrenches'].flatten(),  # 6D (L+R wrenches)
        obs['ee_poses'].flatten(),            # 6D (L+R poses)
        current_twists.flatten(),             # 6D (L+R twists)
        desired_poses.flatten(),              # 6D (L+R desired poses)
        desired_twists.flatten()              # 6D (L+R desired twists)
    ])
    return rl_obs.astype(np.float32)  # Total: 30D
```

### Action Space

**SE(2) Impedance (Eq. 10)**: 12D action
```python
# src/rl_policy/impedance_learning_env.py: line ~165-175
if controller_type == 'se2_impedance':
    action_dim_per_arm = 6  # [d_x, d_y, d_theta, k_x, k_y, k_theta]
    action_dim = 12  # Bimanual
```

**Screw-Decomposed (Eq. 11)**: 8D action
```python
# src/rl_policy/impedance_learning_env.py: line ~112
if controller_type == 'screw_decomposed':
    action_dim_per_arm = 4  # [D_parallel, K_parallel, D_perp, K_perp]
    action_dim = 8  # Bimanual
```

### Action Decoding
```python
# src/rl_policy/impedance_learning_env.py: line ~342-398
def _decode_action_se2(self, action):
    # Scale from [-1, 1] to [min, max]
    damping = [
        scale(action[0], min_D_linear, max_D_linear),
        scale(action[1], min_D_linear, max_D_linear),
        scale(action[2], min_D_angular, max_D_angular)
    ]
    stiffness = [
        scale(action[3], min_K_linear, max_K_linear),
        scale(action[4], min_K_linear, max_K_linear),
        scale(action[5], min_K_angular, max_K_angular)
    ]
    # ... (similarly for right arm)
```

```python
# src/rl_policy/impedance_learning_env.py: line ~400-447
def _decode_action_screw(self, action):
    D_parallel = scale(action[0], min_D_parallel, max_D_parallel)
    K_parallel = scale(action[1], min_K_parallel, max_K_parallel)
    D_perpendicular = scale(action[2], min_D_perp, max_D_perp)
    K_perpendicular = scale(action[3], min_K_perp, max_K_perp)
    # ... (similarly for right arm)
```

### Reward Function (Eq. 12)
```python
# src/rl_policy/impedance_learning_env.py: line ~475-483
def _compute_reward(self, obs, desired_poses, desired_twists, impedance_params):
    # Tracking error
    tracking_error = sum(
        norm(obs['ee_poses'][i] - desired_poses[i]) +
        0.1 * norm(obs['ee_twists'][i] - desired_twists[i])
    )
    reward -= self.config.tracking_weight * tracking_error

    # Wrench penalty
    wrench_magnitude = sum(norm(obs['external_wrenches'][i]))
    reward -= self.config.wrench_weight * wrench_magnitude

    # Smoothness penalty
    param_change = norm(impedance_params - prev_impedance_params)
    reward -= self.config.smoothness_weight * param_change

    return reward
```

**Configuration**: `scripts/configs/rl_config.yaml`
```yaml
rl_training:
  reward:
    tracking_weight: 1.0
    wrench_weight: 0.1
    smoothness_weight: 0.01
```

### PPO Training
```python
# src/rl_policy/ppo_impedance_policy.py
# Wraps stable-baselines3 PPO with:
# - Policy network: 256 hidden units
# - Learning rate: 3e-4
# - Training timesteps: 1M
```

**Training script**: `scripts/training/train_ll_policy.py`

## Section 3.6: Hierarchical Control Architecture

### Level 1: High-Level Planner
```python
# src/hl_planners/
# - flow_matching.py: Flow Matching Policy
# - diffusion_policy.py: Diffusion Policy
# - act.py: ACT Policy
# - teleoperation.py: Keyboard teleoperation
```

All planners implement the same interface:
```python
def get_action(self, obs) -> np.ndarray:
    """Returns desired poses (2, 3) for both arms."""
```

### Level 2: Trajectory Generator
```python
# src/trajectory_generator.py: line ~80-150
class MinimumJerkTrajectory:
    def __init__(self, start_pose, end_pose, duration=1.0):
        # Generates smooth minimum-jerk trajectory

    def evaluate(self, t):
        # Returns TrajectoryPoint at time t with:
        # - pose (spatial frame)
        # - velocity_body (body frame)
        # - velocity_spatial (spatial frame)
        # - acceleration (spatial frame, converted to body)
```

Used in RL environment:
```python
# src/rl_policy/impedance_learning_env.py: line ~360-380
def _update_trajectories(self, obs):
    desired_poses = self.hl_policy.get_action(obs)
    for i in range(2):
        self.trajectories[i] = MinimumJerkTrajectory(
            start_pose=obs['ee_poses'][i],
            end_pose=desired_poses[i],
            duration=1.0
        )
```

### Level 3: Low-Level Controller
The complete hierarchical control loop:

```python
# src/rl_policy/impedance_learning_env.py: line ~260-290
def step(self, action):
    # 1. Decode impedance parameters from RL action
    impedance_params = self._decode_action(action)
    self._update_controller_gains(impedance_params)

    # 2. Get desired trajectories from HL planner
    desired_poses, desired_twists, desired_accels = self._get_trajectory_targets()

    # 3. Compute impedance control wrenches
    for i in range(2):
        if controller_type == 'se2_impedance':
            wrench = self.controllers[i].compute_wrench(
                current_pose, desired_pose, measured_wrench,
                current_velocity, desired_velocity, desired_acceleration
            )
        elif controller_type == 'screw_decomposed':
            wrench, _ = self.controllers[i].compute_control(
                current_pose, desired_pose,
                body_twist_current, body_twist_desired
            )

    # 4. Execute wrench commands in physics
    obs, reward, done, truncated, info = self.base_env.step(wrenches)
```

### Control Frequencies
```python
# Configuration: scripts/configs/rl_config.yaml
environment:
  control_dt: 0.01   # 100 Hz (low-level impedance control)
  policy_dt: 0.1     # 10 Hz (RL policy updates)

hl_policy:
  flow_matching:
    output_frequency: 10.0  # 10 Hz (high-level planner)
```

Implementation in environment:
```python
# src/rl_policy/impedance_learning_env.py: line ~107-108
self.steps_per_policy_update = int(policy_dt / control_dt)  # = 10
# RL policy runs once, impedance controller runs 10 times
```

## Section 3.7: Implementation Details

### BiArt Environment
```python
# src/envs/biart.py
class BiArtEnv(gym.Env):
    """
    SE(2) bimanual manipulation environment
    - Two 2-DOF planar arms
    - 1-DOF articulated object (revolute or prismatic)
    - PyBullet physics at 240 Hz
    - 6-axis F/T sensors
    """
```

### Configuration System
All experiments configured via YAML:

**Main config**: `scripts/configs/rl_config.yaml`
- High-level policy settings
- Low-level controller settings
- RL training hyperparameters
- Reward weights
- Evaluation settings

### Training Scripts

**HL Policy Training**: `scripts/training/train_hl_policy.py`
```bash
python scripts/training/train_hl_policy.py --policy flow_matching
```

**LL Policy Training**: `scripts/training/train_ll_policy.py`
```bash
python scripts/training/train_ll_policy.py \
    --hl_policy flow_matching \
    --controller se2_impedance
```

**Evaluation**: `scripts/evaluation/evaluate_hierarchical.py`
```bash
python scripts/evaluation/evaluate_hierarchical.py \
    --ll_checkpoint checkpoints/impedance_policy.zip \
    --num_episodes 50
```

## Key Files Summary

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| SE(2) Math | `src/se2_math.py` | Frame transformations, pose error |
| SE(2) Dynamics | `src/se2_dynamics.py` | `SE2Dynamics`, task space inertia |
| SE(2) Impedance | `src/ll_controllers/se2_impedance_controller.py` | `SE2ImpedanceController` |
| Screw Impedance | `src/ll_controllers/se2_screw_decomposed_impedance.py` | `SE2ScrewDecomposedImpedanceController` |
| Trajectory Gen | `src/trajectory_generator.py` | `MinimumJerkTrajectory` |
| RL Environment | `src/rl_policy/impedance_learning_env.py` | `ImpedanceLearningEnv` |
| PPO Policy | `src/rl_policy/ppo_impedance_policy.py` | `PPOImpedancePolicy` |
| BiArt Env | `src/envs/biart.py` | `BiArtEnv` |
| Object Manager | `src/envs/object_manager.py` | Screw axis extraction |
| Config | `scripts/configs/rl_config.yaml` | All hyperparameters |

## Experiment Reproducibility

All experiments can be reproduced using the configuration files and training scripts. The hierarchical architecture ensures that:

1. **HL policies** can be trained independently via imitation learning
2. **LL policies** can be trained with any combination of (HL policy type × controller type)
3. **Evaluation** tests the full hierarchical pipeline

This enables systematic ablation studies to validate the design choices presented in the paper.
