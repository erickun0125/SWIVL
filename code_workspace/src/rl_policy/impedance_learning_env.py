"""
SWIVL Impedance Learning Environment

Gym Environment for learning optimal impedance parameters using the
SE2ScrewDecomposedImpedanceController (SWIVL Layer 3).

This environment learns the impedance modulation policy:
    a_t = (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7

Following Modern Robotics (Lynch & Park) convention:
- Twist: V = [ω, vx, vy]^T (angular velocity first!)
- Wrench: F = [τ, fx, fy]^T (torque first!)

Observation Space (per SWIVL Paper Appendix B):
1. Reference Twists (6): V_l^ref, V_r^ref ∈ R^3 × 2
2. Object Constraints/Screw Axes (6): B_l, B_r ∈ R^3 × 2
3. Wrench Feedback (6): F_l, F_r ∈ R^3 × 2
4. Proprioception (12):
   - End-effector poses: (x, y, θ) × 2 = 6
   - End-effector body twists: (ω, vx, vy) × 2 = 6
Total: 30 dimensions

Action space (SWIVL Layer 3):
- d_l_∥: Left arm parallel damping (internal motion)
- d_r_∥: Right arm parallel damping
- d_l_⊥: Left arm perpendicular damping (bulk motion)
- d_r_⊥: Right arm perpendicular damping
- k_p_l: Left arm pose error correction gain
- k_p_r: Right arm pose error correction gain
- α: Shared characteristic length (metric tensor G = diag(α², 1, 1))
Total: 7 dimensions

Reward Function (per SWIVL Paper Section 3.3.2):
- r_track: G-metric velocity tracking error
- r_safety: Fighting force (F_⊥) penalty
- r_reg: Twist acceleration regularization

Reference: SWIVL Paper, Section 3.3 - Wrench-Adaptive Impedance Learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

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
    """
    Configuration for SWIVL impedance learning environment.
    
    All settings should be loaded from rl_config.yaml via from_dict().
    Default values are provided for backward compatibility only.
    """
    
    # Controller type: 'se2_impedance' or 'screw_decomposed'
    controller_type: str = 'screw_decomposed'

    # Robot parameters
    robot_mass: float = 1.2       # kg
    robot_inertia: float = 97.6   # kg⋅m² (pixels²)

    # =========================================================================
    # SWIVL Screw-Decomposed Impedance Controller bounds
    # Action: (d_l_∥, d_r_∥, d_l_⊥, d_r_⊥, k_p_l, k_p_r, α) ∈ R^7
    # =========================================================================
    min_d_parallel: float = 1.0
    max_d_parallel: float = 50.0
    min_d_perp: float = 10.0
    max_d_perp: float = 200.0
    min_k_p: float = 0.5
    max_k_p: float = 10.0
    min_alpha: float = 1.0
    max_alpha: float = 50.0
    default_alpha: float = 10.0
    max_force: float = 100.0
    max_torque: float = 500.0

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
    # Reward weights (per SWIVL Paper Section 3.3.2)
    # =========================================================================
    tracking_weight: float = 1.0
    # Exponential safety reward: r_safety = w_safety * exp(-k * ||F_perp||^2)
    # Provides alive bonus when fighting forces are low
    safety_reward_weight: float = 1.0
    safety_exp_scale: float = 0.01  # k in exp(-k * ||F_perp||^2)
    twist_accel_weight: float = 0.01
    
    # Termination penalty for failure cases (grasp drift, wrench limit)
    termination_penalty: float = 10.0

    # =========================================================================
    # Timing
    # =========================================================================
    control_dt: float = 0.01
    policy_dt: float = 0.01
    hl_chunk_duration: float = 1.0

    # Episode settings
    max_episode_steps: int = 1000
    max_grasp_drift: float = 50.0
    
    # Wrench limit for termination (failure case)
    max_external_wrench: float = 200.0  # N (combined force magnitude)

    # =========================================================================
    # Normalization scales
    # =========================================================================
    norm_pos_scale: float = 512.0
    norm_angle_scale: float = 3.14159
    norm_wrench_scale: float = 100.0
    norm_twist_linear: float = 500.0
    norm_twist_angular: float = 10.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ImpedanceLearningConfig':
        """
        Create config from YAML config dictionary.
        
        This is the preferred way to create a config instance.
        Load from rl_config.yaml for consistent configuration.
        
        Args:
            config: Full configuration dictionary from rl_config.yaml
            
        Returns:
            ImpedanceLearningConfig instance
        """
        env_cfg = config.get('environment', {})
        ll_cfg = config.get('ll_controller', {})
        rl_cfg = config.get('rl_training', {})
        
        # Get controller type
        controller_type = ll_cfg.get('type', 'screw_decomposed')
        
        # Robot parameters
        robot_cfg = ll_cfg.get('robot', {})
        
        # Controller-specific parameters
        screw_cfg = ll_cfg.get('screw_decomposed', {})
        se2_cfg = ll_cfg.get('se2_impedance', {})
        
        # Reward weights
        reward_cfg = rl_cfg.get('reward', {})
        
        # Normalization scales
        norm_cfg = env_cfg.get('normalization', {})
        
        return cls(
            # Controller type
            controller_type=controller_type,
            
            # Robot parameters
            robot_mass=robot_cfg.get('mass', 1.2),
            robot_inertia=robot_cfg.get('inertia', 97.6),
            
            # Screw-decomposed controller bounds
            min_d_parallel=screw_cfg.get('min_d_parallel', 1.0),
            max_d_parallel=screw_cfg.get('max_d_parallel', 50.0),
            min_d_perp=screw_cfg.get('min_d_perp', 10.0),
            max_d_perp=screw_cfg.get('max_d_perp', 200.0),
            min_k_p=screw_cfg.get('min_k_p', 0.5),
            max_k_p=screw_cfg.get('max_k_p', 10.0),
            min_alpha=screw_cfg.get('min_alpha', 1.0),
            max_alpha=screw_cfg.get('max_alpha', 50.0),
            default_alpha=screw_cfg.get('default_alpha', 10.0),
            max_force=screw_cfg.get('max_force', 100.0),
            max_torque=screw_cfg.get('max_torque', 500.0),
            
            # SE(2) impedance controller bounds
            min_damping_linear=se2_cfg.get('min_damping_linear', 1.0),
            max_damping_linear=se2_cfg.get('max_damping_linear', 50.0),
            min_damping_angular=se2_cfg.get('min_damping_angular', 0.5),
            max_damping_angular=se2_cfg.get('max_damping_angular', 20.0),
            min_stiffness_linear=se2_cfg.get('min_stiffness_linear', 10.0),
            max_stiffness_linear=se2_cfg.get('max_stiffness_linear', 200.0),
            min_stiffness_angular=se2_cfg.get('min_stiffness_angular', 5.0),
            max_stiffness_angular=se2_cfg.get('max_stiffness_angular', 100.0),
            
            # Reward weights
            tracking_weight=reward_cfg.get('tracking_weight', 1.0),
            safety_reward_weight=reward_cfg.get('safety_reward_weight', 1.0),
            safety_exp_scale=reward_cfg.get('safety_exp_scale', 0.01),
            twist_accel_weight=reward_cfg.get('twist_accel_weight', 0.01),
            termination_penalty=reward_cfg.get('termination_penalty', 10.0),
            
            # Timing
            control_dt=env_cfg.get('control_dt', 0.01),
            policy_dt=env_cfg.get('policy_dt', 0.01),
            hl_chunk_duration=env_cfg.get('hl_chunk_duration', 1.0),
            
            # Episode settings
            max_episode_steps=env_cfg.get('max_episode_steps', 1000),
            max_grasp_drift=env_cfg.get('max_grasp_drift', 50.0),
            max_external_wrench=env_cfg.get('max_external_wrench', 200.0),
            
            # Normalization scales
            norm_pos_scale=norm_cfg.get('pos_scale', 512.0),
            norm_angle_scale=norm_cfg.get('angle_scale', 3.14159),
            norm_wrench_scale=norm_cfg.get('wrench_scale', 100.0),
            norm_twist_linear=norm_cfg.get('twist_linear_scale', 500.0),
            norm_twist_angular=norm_cfg.get('twist_angular_scale', 10.0),
        )


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
            config: Environment configuration (use ImpedanceLearningConfig.from_dict())
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

        # Previous twist for acceleration computation (per paper r_reg)
        self.prev_twists = None
        
        # Current impedance parameters (for reward computation)
        self.current_alpha = self.config.default_alpha
        
        # Screw axes (updated from environment)
        self.screw_axes = np.array([
            [1.0, 0.0, 0.0],  # Default revolute
            [1.0, 0.0, 0.0],
        ])
        
        # Current reference twists (computed from trajectory)
        self.current_ref_twists = np.zeros((2, 3))
        
        # Initial grasp poses for termination check
        self.initial_grasp_poses = None

        # Define observation space (30D per SWIVL paper Appendix B)
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
        initial_screw_axes = np.array([
            [1.0, 0.0, 0.0],
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
            alpha=self.config.default_alpha
        )

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
        """
        Normalize observation vector to roughly [-1, 1].
        
        Per SWIVL Paper Appendix B, observation structure:
        [ref_twists(6), screw_axes(6), wrenches(6), poses(6), twists(6)] = 30D
        """
        normalized = obs.copy()
        cfg = self.config

        # Reference twists [omega, vx, vy] * 2 (indices 0-5)
        normalized[[0, 3]] /= cfg.norm_twist_angular
        normalized[[1, 2, 4, 5]] /= cfg.norm_twist_linear
        
        # Screw axes [s_omega, s_x, s_y] * 2 (indices 6-11) - already normalized
        
        # Wrenches [tau, fx, fy] * 2 (indices 12-17)
        normalized[[12, 15]] /= cfg.norm_wrench_scale * 0.5
        normalized[[13, 14, 16, 17]] /= cfg.norm_wrench_scale

        # Poses [x, y, theta] * 2 (indices 18-23)
        normalized[[18, 19, 21, 22]] /= cfg.norm_pos_scale
        normalized[[20, 23]] /= cfg.norm_angle_scale

        # Twists [omega, vx, vy] * 2 (indices 24-29)
        normalized[[24, 27]] /= cfg.norm_twist_angular
        normalized[[25, 26, 28, 29]] /= cfg.norm_twist_linear

        return normalized

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        obs, info = self.base_env.reset(seed=seed, options=options)

        if self.hl_policy is not None:
            self.hl_policy.reset()

        # Update screw axes from environment
        if self.config.controller_type == 'screw_decomposed':
            screw_axes = self.base_env.get_joint_axis_screws()
            if screw_axes is not None:
                B_left, B_right = screw_axes
                self.screw_axes = np.array([B_left, B_right])
                self.controller.set_screw_axes(self.screw_axes)

        self.controller.reset()

        # Reset timing
        self.episode_steps = 0
        self.elapsed_time_in_chunk = self.hl_period
        self.prev_twists = None
        self.current_alpha = self.config.default_alpha
        self.initial_grasp_poses = obs['ee_poses'].copy()

        # Initialize trajectories
        self._update_trajectories(obs)

        rl_obs = self._get_rl_observation(obs)
        norm_obs = self._normalize_obs(rl_obs)

        return norm_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self._apply_impedance_action(action)
        
        if self.config.controller_type == 'screw_decomposed':
            self.current_alpha = self._scale_action(
                action[6], self.config.min_alpha, self.config.max_alpha
            )

        if self.elapsed_time_in_chunk >= self.hl_period:
            obs = self.base_env.get_obs()
            self._update_trajectories(obs)
            self.elapsed_time_in_chunk = 0.0

        obs = self.base_env.get_obs()
        desired_poses, desired_twists = self._get_trajectory_targets(self.elapsed_time_in_chunk)

        current_poses = obs['ee_poses']
        current_twists = self._get_body_twists(obs)
        external_wrenches = obs['external_wrenches']
        
        self.current_ref_twists = self._compute_reference_twists(
            current_poses, desired_poses, desired_twists, action
        )

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

        obs, _, terminated, truncated, info = self.base_env.step(wrenches)

        # Check termination conditions
        failure_termination = False
        termination_reason = None
        
        if self._check_grasp_drift(obs):
            terminated = True
            failure_termination = True
            termination_reason = 'grasp_drift'
            
        if self._check_wrench_limit(obs):
            terminated = True
            failure_termination = True
            termination_reason = 'wrench_limit'

        # Compute reward with termination penalty for failure cases
        reward = self._compute_reward(
            obs, current_twists, control_info, 
            failure_termination=failure_termination
        )

        self.elapsed_time_in_chunk += self.config.control_dt
        self.episode_steps += 1

        if self.episode_steps >= self.config.max_episode_steps:
            truncated = True
        
        if termination_reason is not None:
            info['termination_reason'] = termination_reason

        rl_obs = self._get_rl_observation(obs)
        norm_obs = self._normalize_obs(rl_obs)

        self.prev_twists = current_twists.copy()

        info['control_info'] = control_info
        info['current_alpha'] = self.current_alpha
        info['failure_termination'] = failure_termination

        return norm_obs, reward, terminated, truncated, info

    def _apply_impedance_action(self, action: np.ndarray):
        """Apply action to update controller impedance parameters."""
        if self.config.controller_type == 'screw_decomposed':
            self._apply_screw_decomposed_action(action)
        else:
            self._apply_se2_impedance_action(action)

    def _apply_screw_decomposed_action(self, action: np.ndarray):
        """Apply SWIVL impedance action."""
        cfg = self.config
        d_l_parallel = self._scale_action(action[0], cfg.min_d_parallel, cfg.max_d_parallel)
        d_r_parallel = self._scale_action(action[1], cfg.min_d_parallel, cfg.max_d_parallel)
        d_l_perp = self._scale_action(action[2], cfg.min_d_perp, cfg.max_d_perp)
        d_r_perp = self._scale_action(action[3], cfg.min_d_perp, cfg.max_d_perp)
        k_p_l = self._scale_action(action[4], cfg.min_k_p, cfg.max_k_p)
        k_p_r = self._scale_action(action[5], cfg.min_k_p, cfg.max_k_p)
        alpha = self._scale_action(action[6], cfg.min_alpha, cfg.max_alpha)

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
        cfg = self.config
        for i in range(2):
            arm_action = action[i*6:(i+1)*6]

            damping = np.array([
                self._scale_action(arm_action[0], cfg.min_damping_angular, cfg.max_damping_angular),
                self._scale_action(arm_action[1], cfg.min_damping_linear, cfg.max_damping_linear),
                self._scale_action(arm_action[2], cfg.min_damping_linear, cfg.max_damping_linear)
            ])

            stiffness = np.array([
                self._scale_action(arm_action[3], cfg.min_stiffness_angular, cfg.max_stiffness_angular),
                self._scale_action(arm_action[4], cfg.min_stiffness_linear, cfg.max_stiffness_linear),
                self._scale_action(arm_action[5], cfg.min_stiffness_linear, cfg.max_stiffness_linear)
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
            body_twists = []
            for i in range(2):
                point_vel = obs['ee_velocities'][i]
                pose = obs['ee_poses'][i]
                theta = pose[2]
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)

                vx_body = cos_theta * point_vel[0] + sin_theta * point_vel[1]
                vy_body = -sin_theta * point_vel[0] + cos_theta * point_vel[1]
                body_twists.append(np.array([point_vel[2], vx_body, vy_body]))

            return np.array(body_twists)
    
    def _compute_reference_twists(
        self,
        current_poses: np.ndarray,
        desired_poses: np.ndarray,
        desired_twists: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """Compute reference twists per SWIVL paper Eq. (6)."""
        ref_twists = np.zeros((2, 3))
        
        k_p_l = self._scale_action(action[4], self.config.min_k_p, self.config.max_k_p)
        k_p_r = self._scale_action(action[5], self.config.min_k_p, self.config.max_k_p)
        k_p_values = [k_p_l, k_p_r]
        
        for i in range(2):
            theta_b = current_poses[i, 2]
            dx = desired_poses[i, 0] - current_poses[i, 0]
            dy = desired_poses[i, 1] - current_poses[i, 1]
            
            cos_b, sin_b = np.cos(theta_b), np.sin(theta_b)
            e_x = cos_b * dx + sin_b * dy
            e_y = -sin_b * dx + cos_b * dy
            e_theta = self._normalize_angle(desired_poses[i, 2] - theta_b)
            
            E = np.array([e_theta, e_x, e_y])
            ref_twists[i] = desired_twists[i] + k_p_values[i] * E
            
        return ref_twists
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _check_grasp_drift(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if grasp has drifted beyond threshold."""
        if self.initial_grasp_poses is None:
            return False
            
        for i in range(2):
            dx = obs['ee_poses'][i, 0] - self.initial_grasp_poses[i, 0]
            dy = obs['ee_poses'][i, 1] - self.initial_grasp_poses[i, 1]
            dtheta = self._normalize_angle(
                obs['ee_poses'][i, 2] - self.initial_grasp_poses[i, 2]
            )
            
            alpha = self.current_alpha
            drift = np.sqrt(alpha**2 * dtheta**2 + dx**2 + dy**2)
            
            if drift > self.config.max_grasp_drift:
                return True
                
        return False
    
    def _check_wrench_limit(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Check if external wrench exceeds safety limit.
        
        Per SWIVL Paper Section 3.3.2:
        Episodes terminate when external wrenches exceed safe operating limits,
        as excessive wrenches risk hardware damage and grasp instability.
        
        Uses the dual metric G^(-1) = diag(1/α², 1, 1) to compute wrench magnitude,
        ensuring consistent weighting between moment and force components:
        
        ||F||_{G^(-1)}² = τ²/α² + fx² + fy²
        
        This is dual to the twist metric G = diag(α², 1, 1), preserving the
        reciprocal product (virtual power) relationship between twists and wrenches.
        """
        alpha = self.current_alpha
        
        for i in range(2):
            wrench = obs['external_wrenches'][i]
            # wrench = [tau, fx, fy] in SE(2)
            tau, fx, fy = wrench[0], wrench[1], wrench[2]
            
            # Compute wrench magnitude using dual metric G^(-1)
            # ||F||_{G^(-1)}² = τ²/α² + fx² + fy²
            wrench_magnitude_sq = (tau**2 / (alpha**2)) + fx**2 + fy**2
            wrench_magnitude = np.sqrt(wrench_magnitude_sq)
            
            if wrench_magnitude > self.config.max_external_wrench:
                return True
                
        return False

    def _update_trajectories(self, obs: Dict[str, np.ndarray]):
        """Update trajectories using high-level policy."""
        if self.hl_policy is None:
            for i in range(2):
                current_pose = obs['ee_poses'][i]
                self.trajectories[i] = MinimumJerkTrajectory(
                    start_pose=current_pose,
                    end_pose=current_pose,
                    duration=self.hl_period
                )
        else:
            image = self.base_env._render_frame(visualize=False)
            ee_velocities = obs.get('ee_velocities', np.zeros((2, 3)))
            
            hl_obs = {
                'image': image,
                'ee_poses': obs['ee_poses'],
                'ee_velocities': ee_velocities,
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
        current_twists = self._get_body_twists(obs)

        rl_obs = np.concatenate([
            self.current_ref_twists.flatten(),
            self.screw_axes.flatten(),
            obs['external_wrenches'].flatten(),
            obs['ee_poses'].flatten(),
            current_twists.flatten(),
        ])

        return rl_obs.astype(np.float32)

    def _compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        current_twists: np.ndarray,
        control_info: Dict,
        failure_termination: bool = False
    ) -> float:
        """
        Compute reward for SWIVL impedance learning.
        
        Per SWIVL Paper Section 3.3.2:
        r_t = r_track + r_safety + r_reg + r_term
        
        Key design choices:
        - r_safety uses exponential form to provide positive "alive bonus" 
          when fighting forces are low, avoiding the pathological behavior
          where agents learn to terminate early (all-negative rewards)
        - r_term applies termination penalty for failure cases (grasp drift,
          wrench limit) to discourage unsafe behaviors
        """
        reward = 0.0
        alpha = self.current_alpha
        G = np.diag([alpha**2, 1.0, 1.0])
        cfg = self.config

        # =====================================================================
        # r_track: G-metric velocity tracking error (negative penalty)
        # =====================================================================
        tracking_error = 0.0
        for i in range(2):
            twist_error = current_twists[i] - self.current_ref_twists[i]
            g_norm_sq = twist_error @ G @ twist_error
            tracking_error += g_norm_sq
        reward -= cfg.tracking_weight * tracking_error

        # =====================================================================
        # r_safety: Exponential safety reward (positive alive bonus)
        # 
        # r_safety = w_safety * exp(-k * Σ||F_⊥||_{G^{-1}}²)
        # 
        # Uses the dual metric G^{-1} = diag(1/α², 1, 1) for wrench norms,
        # which is mathematically consistent with the twist metric G and
        # ensures dimensional consistency (moment and force have same units).
        # 
        # ||F||_{G^{-1}}² = τ²/α² + fx² + fy²
        # 
        # Properties:
        # - When ||F_⊥|| ≈ 0: r_safety ≈ w_safety (max positive reward)
        # - As ||F_⊥|| increases: r_safety → 0 exponentially
        # - Acts as "alive bonus" encouraging safe, low-force behavior
        # =====================================================================
        if self.config.controller_type == 'screw_decomposed':
            fighting_force_sq = 0.0
            for i in range(2):
                P_perp = self.controller.controllers[i].P_perp
                F_ext = obs['external_wrenches'][i]
                F_perp = P_perp.T @ F_ext
                
                # Use dual metric G^{-1} for wrench norm (dimensionally consistent)
                # ||F_perp||_{G^{-1}}² = τ²/α² + fx² + fy²
                tau_perp, fx_perp, fy_perp = F_perp[0], F_perp[1], F_perp[2]
                fighting_force_sq += (tau_perp**2 / (alpha**2)) + fx_perp**2 + fy_perp**2
            
            # Exponential safety reward
            safety_reward = cfg.safety_reward_weight * np.exp(
                -cfg.safety_exp_scale * fighting_force_sq
            )
            reward += safety_reward

        # =====================================================================
        # r_reg: Twist acceleration regularization (negative penalty)
        # =====================================================================
        if self.prev_twists is not None:
            twist_accel_sq = 0.0
            dt = cfg.control_dt
            for i in range(2):
                twist_accel = (current_twists[i] - self.prev_twists[i]) / dt
                twist_accel_sq += np.linalg.norm(twist_accel)**2
            reward -= cfg.twist_accel_weight * twist_accel_sq

        # =====================================================================
        # r_term: Termination penalty for failure cases
        # 
        # Applied when episode terminates due to:
        # - Grasp drift exceeding threshold
        # - External wrench exceeding safety limit
        # 
        # This discourages the agent from learning to intentionally fail
        # =====================================================================
        if failure_termination:
            reward -= cfg.termination_penalty

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

    config = ImpedanceLearningConfig(
        controller_type='screw_decomposed',
        max_episode_steps=100
    )

    env = ImpedanceLearningEnv(config=config)

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}")
        if terminated:
            break

    env.close()
    print("\n✓ Environment test passed!")
