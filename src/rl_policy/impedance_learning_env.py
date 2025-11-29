"""
SWIVL Impedance Learning Environment

Gym Environment for learning optimal impedance parameters using the
SE2ScrewDecomposedImpedanceController (SWIVL Layer 3).

This environment learns the impedance modulation policy:
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

Following Modern Robotics (Lynch & Park) convention:
- Twist: V = [ω, vx, vy]^T (angular velocity first!)
- Wrench: F = [τ, fx, fy]^T (torque first!)

State space (per arm):
- External wrench (3): [tau, fx, fy] (MR convention!)
- Current pose (3): [x, y, theta]
- Current twist (3): [omega, vx, vy] (MR convention!)
- Desired pose (3): [x_d, y_d, theta_d]
- Desired twist (3): [omega_d, vx_d, vy_d] (MR convention!)
Total: 15 * 2 = 30 dimensions (bimanual)

Action space (SWIVL):
- d_l_∥: Left arm parallel damping (internal motion)
- d_r_∥: Right arm parallel damping
- d_l_⊥: Left arm perpendicular damping (bulk motion)
- d_r_⊥: Right arm perpendicular damping
- k_p_l: Left arm pose error correction gain
- k_p_r: Right arm pose error correction gain
- α: Shared characteristic length (metric tensor)
Total: 7 dimensions

Reference: SWIVL Paper, Section 3.3 - Impedance Variable Modulation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from src.envs.biart import BiArtEnv
from src.trajectory_generator import MinimumJerkTrajectory, CubicSplineTrajectory
from src.ll_controllers.se2_impedance_controller import (
    SE2ImpedanceController,
    MultiGripperSE2ImpedanceController
)
from src.ll_controllers.se2_screw_decomposed_impedance import (
    SE2ScrewDecomposedImpedanceController,
    MultiGripperSE2ScrewDecomposedImpedanceController,
    ScrewDecomposedImpedanceParams
)
from src.se2_dynamics import SE2Dynamics, SE2RobotParams


@dataclass
class ImpedanceLearningConfig:
    """Configuration for SWIVL impedance learning environment."""
    
    # Controller type: 'se2_impedance' or 'screw_decomposed'
    controller_type: str = 'screw_decomposed'

    # Robot parameters
    robot_mass: float = 1.2       # kg
    robot_inertia: float = 97.6   # kg⋅m² (pixels²)

    # =========================================================================
    # SE(2) Impedance Controller bounds (classical impedance)
    # =========================================================================
    min_damping_linear: float = 1.0
    max_damping_linear: float = 50.0
    min_damping_angular: float = 0.5
    max_damping_angular: float = 20.0
    min_stiffness_linear: float = 10.0
    max_stiffness_linear: float = 200.0
    min_stiffness_angular: float = 5.0
    max_stiffness_angular: float = 100.0

    # =========================================================================
    # SWIVL Screw-Decomposed Impedance Controller bounds
    # Action: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7
    # =========================================================================
    
    # Parallel damping d_∥ (internal motion, compliant)
    min_d_parallel: float = 1.0
    max_d_parallel: float = 50.0

    # Perpendicular damping d_⊥ (bulk motion, stiff)
    min_d_perp: float = 10.0
    max_d_perp: float = 200.0

    # Pose error correction gain k_p
    min_k_p: float = 0.5
    max_k_p: float = 10.0

    # Characteristic length α (metric tensor)
    min_alpha: float = 1.0
    max_alpha: float = 50.0

    # Controller limits
    max_force: float = 100.0
    max_torque: float = 500.0

    # =========================================================================
    # Reward weights
    # =========================================================================
    tracking_weight: float = 1.0        # Tracking error penalty
    fighting_force_weight: float = 0.5  # Fighting force penalty (F_⊥)
    wrench_weight: float = 0.1          # Total wrench penalty
    smoothness_weight: float = 0.01     # Impedance smoothness penalty

    # =========================================================================
    # Timing
    # =========================================================================
    control_dt: float = 0.01    # 100 Hz (low-level control)
    policy_dt: float = 0.01     # 100 Hz (RL policy, same as control for now)
    hl_chunk_duration: float = 1.0  # High-level policy chunk duration

    # Episode settings
    max_episode_steps: int = 1000


class ImpedanceLearningEnv(gym.Env):
    """
    SWIVL Impedance Learning Environment.

    Learns optimal impedance parameters for bimanual manipulation
    using the screw-decomposed twist-driven impedance controller.

    The RL agent outputs impedance modulation variables that are used
    by the Layer 4 controller to achieve compliant manipulation.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        config: Optional[ImpedanceLearningConfig] = None,
        hl_policy=None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize environment.

        Args:
            config: Environment configuration
            hl_policy: Pre-trained high-level policy (generates desired poses)
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config if config is not None else ImpedanceLearningConfig()
        self.hl_policy = hl_policy
        self.render_mode = render_mode

        # HL policy timing
        self.hl_period = self.config.hl_chunk_duration
        self.elapsed_time_in_chunk = 0.0

        # Create base environment
        base_control_hz = int(1.0 / self.config.control_dt)
        self.base_env = BiArtEnv(
            render_mode=render_mode,
            control_hz=base_control_hz,
            physics_hz=100
        )

        # Robot parameters
        self.robot_params = SE2RobotParams(
            mass=self.config.robot_mass,
            inertia=self.config.robot_inertia
        )

        # Controller setup based on type
        if self.config.controller_type == 'screw_decomposed':
            self._setup_screw_decomposed_controller()
        elif self.config.controller_type == 'se2_impedance':
            self._setup_se2_impedance_controller()
        else:
            raise ValueError(f"Unknown controller type: {self.config.controller_type}")

        # Trajectory trackers
        self.trajectories = [None, None]

        # Episode tracking
        self.episode_steps = 0

        # Previous action for smoothness penalty
        self.prev_action = None

        # Normalization constants
        self.NORM_POS_SCALE = 512.0
        self.NORM_ANGLE_SCALE = np.pi
        self.NORM_WRENCH_SCALE = 100.0
        self.NORM_TWIST_LINEAR = 500.0
        self.NORM_TWIST_ANGULAR = 10.0

        # Define observation space (30D bimanual)
        obs_dim = 30
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space based on controller type
        self._setup_action_space()

    def _setup_screw_decomposed_controller(self):
        """Setup SWIVL screw-decomposed controller."""
        # Initial screw axes (will be updated from environment in reset)
        initial_screw_axes = np.array([
            [1.0, 0.0, 0.0],  # Revolute default
            [1.0, 0.0, 0.0],
        ])

        self.controller = MultiGripperSE2ScrewDecomposedImpedanceController(
            num_grippers=2,
            screw_axes=initial_screw_axes,
            robot_params=self.robot_params,
            max_force=self.config.max_force,
            max_torque=self.config.max_torque
        )

        # Set default impedance variables
        self.controller.set_impedance_variables(
            d_l_parallel=10.0,
            d_r_parallel=10.0,
            d_l_perp=100.0,
            d_r_perp=100.0,
            k_p_l=3.0,
            k_p_r=3.0,
            alpha=10.0
        )

        # Action dimension: 7 (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α)
        self.action_dim = 7

    def _setup_se2_impedance_controller(self):
        """Setup classical SE(2) impedance controller."""
        self.controller = MultiGripperSE2ImpedanceController(
            num_grippers=2,
            robot_params=self.robot_params,
            M_d=np.diag([self.config.robot_inertia, self.config.robot_mass, self.config.robot_mass]),
            D_d=np.diag([50.0, 10.0, 10.0]),
            K_d=np.diag([200.0, 50.0, 50.0]),
            model_matching=True,
            max_force=self.config.max_force,
            max_torque=self.config.max_torque
        )

        # Action dimension: 12 (damping 6 + stiffness 6)
        self.action_dim = 12

    def _setup_action_space(self):
        """Setup action space based on controller type."""
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation vector to roughly [-1, 1]."""
        normalized = obs.copy()

        # Structure: [wrenches(6), poses(6), twists(6), des_poses(6), des_twists(6)]
        
        # Wrenches [tau, fx, fy] * 2
        normalized[[0, 3]] /= self.NORM_WRENCH_SCALE * 0.5  # Torque
        normalized[[1, 2, 4, 5]] /= self.NORM_WRENCH_SCALE  # Forces

        # Poses [x, y, theta] * 2 (indices 6-11 and 18-23)
        for start_idx in [6, 18]:
            normalized[[start_idx, start_idx+1, start_idx+3, start_idx+4]] /= self.NORM_POS_SCALE
            normalized[[start_idx+2, start_idx+5]] /= self.NORM_ANGLE_SCALE

        # Twists [omega, vx, vy] * 2 (indices 12-17 and 24-29)
        for start_idx in [12, 24]:
            normalized[[start_idx, start_idx+3]] /= self.NORM_TWIST_ANGULAR
            normalized[[start_idx+1, start_idx+2, start_idx+4, start_idx+5]] /= self.NORM_TWIST_LINEAR

        return normalized

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset base environment
        obs, info = self.base_env.reset(seed=seed, options=options)

        # Reset high-level policy
        if self.hl_policy is not None:
            self.hl_policy.reset()

        # Update screw axes from environment
        if self.config.controller_type == 'screw_decomposed':
            screw_axes = self.base_env.get_joint_axis_screws()
            if screw_axes is not None:
                B_left, B_right = screw_axes
                self.controller.set_screw_axes(np.array([B_left, B_right]))

        # Reset controller
        self.controller.reset()

        # Reset timing
        self.episode_steps = 0
        self.elapsed_time_in_chunk = self.hl_period  # Force update on first step
        self.prev_action = None

        # Initialize trajectories
        self._update_trajectories(obs)

        # Get initial observation
        rl_obs = self._get_rl_observation(obs)
        norm_obs = self._normalize_obs(rl_obs)

        return norm_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Impedance parameters (normalized to [-1, 1])

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode and apply impedance parameters
        self._apply_impedance_action(action)

        # Update trajectories if needed
        if self.elapsed_time_in_chunk >= self.hl_period:
            obs = self.base_env.get_obs()
            self._update_trajectories(obs)
            self.elapsed_time_in_chunk = 0.0

        # Get current observation
        obs = self.base_env.get_obs()

        # Get trajectory targets
        desired_poses, desired_twists = self._get_trajectory_targets(self.elapsed_time_in_chunk)

        # Compute control wrenches
        current_poses = obs['ee_poses']
        current_twists = self._get_body_twists(obs)
        external_wrenches = obs['external_wrenches']

        if self.config.controller_type == 'screw_decomposed':
            wrenches, control_info = self.controller.compute_wrenches(
                current_poses,
                desired_poses,
                current_twists,
                desired_twists,
                external_wrenches
            )
        else:
            wrenches = self.controller.compute_wrenches(
                current_poses,
                desired_poses,
                current_twists,
                desired_twists,
                external_wrenches=external_wrenches
            )
            control_info = {}

        # Step base environment
        obs, _, terminated, truncated, info = self.base_env.step(wrenches)

        # Compute reward
        reward = self._compute_reward(
            obs, desired_poses, desired_twists, action, control_info
        )

        # Update timing
        self.elapsed_time_in_chunk += self.config.control_dt
        self.episode_steps += 1

        # Check episode timeout
        if self.episode_steps >= self.config.max_episode_steps:
            truncated = True

        # Get observation
        rl_obs = self._get_rl_observation(obs)
        norm_obs = self._normalize_obs(rl_obs)

        # Store action for smoothness penalty
        self.prev_action = action.copy()

        # Add control info to info dict
        info['control_info'] = control_info

        return norm_obs, reward, terminated, truncated, info

    def _apply_impedance_action(self, action: np.ndarray):
        """Apply action to update controller impedance parameters."""
        if self.config.controller_type == 'screw_decomposed':
            self._apply_screw_decomposed_action(action)
        else:
            self._apply_se2_impedance_action(action)

    def _apply_screw_decomposed_action(self, action: np.ndarray):
        """
        Apply SWIVL impedance action.

        Action: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7
        """
        # Scale actions from [-1, 1] to parameter ranges
        d_l_parallel = self._scale_action(action[0], self.config.min_d_parallel, self.config.max_d_parallel)
        d_r_parallel = self._scale_action(action[1], self.config.min_d_parallel, self.config.max_d_parallel)
        d_l_perp = self._scale_action(action[2], self.config.min_d_perp, self.config.max_d_perp)
        d_r_perp = self._scale_action(action[3], self.config.min_d_perp, self.config.max_d_perp)
        k_p_l = self._scale_action(action[4], self.config.min_k_p, self.config.max_k_p)
        k_p_r = self._scale_action(action[5], self.config.min_k_p, self.config.max_k_p)
        alpha = self._scale_action(action[6], self.config.min_alpha, self.config.max_alpha)

        self.controller.set_impedance_variables(
            d_l_parallel=d_l_parallel,
            d_r_parallel=d_r_parallel,
            d_l_perp=d_l_perp,
            d_r_perp=d_r_perp,
            k_p_l=k_p_l,
            k_p_r=k_p_r,
            alpha=alpha
        )

    def _apply_se2_impedance_action(self, action: np.ndarray):
        """Apply classical SE(2) impedance action."""
        # Split for two arms: 6 per arm (3 damping + 3 stiffness)
        for i in range(2):
            arm_action = action[i*6:(i+1)*6]

            # Damping [D_angular, D_x, D_y]
            damping = np.array([
                self._scale_action(arm_action[0], self.config.min_damping_angular, self.config.max_damping_angular),
                self._scale_action(arm_action[1], self.config.min_damping_linear, self.config.max_damping_linear),
                self._scale_action(arm_action[2], self.config.min_damping_linear, self.config.max_damping_linear)
            ])

            # Stiffness [K_angular, K_x, K_y]
            stiffness = np.array([
                self._scale_action(arm_action[3], self.config.min_stiffness_angular, self.config.max_stiffness_angular),
                self._scale_action(arm_action[4], self.config.min_stiffness_linear, self.config.max_stiffness_linear),
                self._scale_action(arm_action[5], self.config.min_stiffness_linear, self.config.max_stiffness_linear)
            ])

            self.controller.controllers[i].set_impedance_parameters(
                D_d=np.diag(damping),
                K_d=np.diag(stiffness)
            )

    def _scale_action(self, normalized_value: float, min_val: float, max_val: float) -> float:
        """Scale normalized action from [-1, 1] to [min_val, max_val]."""
        return min_val + (normalized_value + 1.0) * 0.5 * (max_val - min_val)

    def _get_body_twists(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get body twists from observation."""
        if 'ee_body_twists' in obs:
            return obs['ee_body_twists']
        else:
            # Convert point velocities to body twists
            body_twists = []
            for i in range(2):
                point_vel = obs['ee_velocities'][i]  # [vx, vy, omega]
                pose = obs['ee_poses'][i]
                theta = pose[2]
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)

                # Rotate to body frame (simple rotation, NOT adjoint!)
                vx_body = cos_theta * point_vel[0] + sin_theta * point_vel[1]
                vy_body = -sin_theta * point_vel[0] + cos_theta * point_vel[1]
                body_twists.append(np.array([point_vel[2], vx_body, vy_body]))

            return np.array(body_twists)

    def _update_trajectories(self, obs: Dict[str, np.ndarray]):
        """Update trajectories using high-level policy."""
        if self.hl_policy is None:
            # Hold position
            for i in range(2):
                current_pose = obs['ee_poses'][i]
                self.trajectories[i] = MinimumJerkTrajectory(
                    start_pose=current_pose,
                    end_pose=current_pose,
                    duration=self.hl_period
                )
        else:
            # Render image for vision-based HL policy
            image = self.base_env._render_frame(visualize=False)
            
            # Get point velocities from observation
            ee_velocities = obs.get('ee_velocities', np.zeros((2, 3)))
            
            # Construct observation for HL policy
            hl_obs = {
                'image': image,  # (H, W, 3)
                'ee_poses': obs['ee_poses'],
                'ee_velocities': ee_velocities,  # Point velocities [vx, vy, omega]
                'link_poses': obs['link_poses'],
                'external_wrenches': obs['external_wrenches'],
            }

            action_chunk = self.hl_policy.get_action_chunk(hl_obs)
            num_steps = action_chunk.shape[0]
            chunk_dt = 0.1
            times = np.linspace(0.0, num_steps * chunk_dt, num_steps + 1)

            for i in range(2):
                current_pose = obs['ee_poses'][i]
                waypoints = np.vstack([current_pose[np.newaxis, :], action_chunk[:, i, :]])

                self.trajectories[i] = CubicSplineTrajectory(
                    waypoints=waypoints,
                    times=times,
                    boundary_conditions='natural'
                )
                self.trajectories[i].set_duration(self.hl_period)

    def _get_trajectory_targets(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get desired poses and twists from trajectories."""
        desired_poses = []
        desired_twists = []

        for i in range(2):
            if self.trajectories[i] is None:
                desired_poses.append(np.zeros(3))
                desired_twists.append(np.zeros(3))
            else:
                eval_t = np.clip(t, 0.0, self.hl_period)
                traj_point = self.trajectories[i].evaluate(eval_t)

                desired_poses.append(traj_point.pose)
                desired_twists.append(traj_point.velocity_body)

        return np.array(desired_poses), np.array(desired_twists)

    def _get_rl_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct RL observation from environment observation."""
        desired_poses, desired_twists = self._get_trajectory_targets(self.elapsed_time_in_chunk)
        current_twists = self._get_body_twists(obs)

        rl_obs = np.concatenate([
            obs['external_wrenches'].flatten(),  # 6
            obs['ee_poses'].flatten(),            # 6
            current_twists.flatten(),             # 6
            desired_poses.flatten(),              # 6
            desired_twists.flatten()              # 6
        ])

        return rl_obs.astype(np.float32)

    def _compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        desired_poses: np.ndarray,
        desired_twists: np.ndarray,
        action: np.ndarray,
        control_info: Dict
    ) -> float:
        """
        Compute reward for SWIVL impedance learning.

        Reward components:
        1. Tracking error (negative) - pose + velocity error
        2. Fighting force (negative) - F_⊥ from screw decomposition
        3. Total wrench magnitude (negative) - energy efficiency
        4. Smoothness (negative) - impedance parameter changes
        """
        reward = 0.0

        # 1. Tracking error
        tracking_error = 0.0
        current_twists = self._get_body_twists(obs)
        for i in range(2):
            pose_error = np.linalg.norm(obs['ee_poses'][i] - desired_poses[i])
            twist_error = np.linalg.norm(current_twists[i] - desired_twists[i])
            tracking_error += pose_error + 0.1 * twist_error
        reward -= self.config.tracking_weight * tracking_error

        # 2. Fighting force (screw decomposition specific)
        if self.config.controller_type == 'screw_decomposed' and control_info:
            fighting_force = control_info.get('total_fighting_force', 0.0)
            reward -= self.config.fighting_force_weight * fighting_force

        # 3. Wrench magnitude
        wrench_magnitude = 0.0
        for i in range(2):
            wrench_magnitude += np.linalg.norm(obs['external_wrenches'][i])
        reward -= self.config.wrench_weight * wrench_magnitude

        # 4. Smoothness penalty
        if self.prev_action is not None:
            action_change = np.linalg.norm(action - self.prev_action)
            reward -= self.config.smoothness_weight * action_change

        return reward

    def render(self):
        """Render environment."""
        return self.base_env.render()

    def close(self):
        """Close environment."""
        self.base_env.close()


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SWIVL Impedance Learning Environment Test")
    print("=" * 70)

    # Test screw decomposed controller
    config = ImpedanceLearningConfig(
        controller_type='screw_decomposed',
        max_episode_steps=100
    )

    env = ImpedanceLearningEnv(config=config)

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Test step
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, obs_range=[{obs.min():.2f}, {obs.max():.2f}]")

    env.close()
    print("\n✓ Environment test passed!")
