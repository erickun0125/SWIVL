"""
Gym Environment for Impedance Parameter Learning

This environment wraps the bimanual manipulation task and provides
an interface for learning optimal impedance parameters using RL.

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

Action space (per arm):
- For SE(2) impedance: Damping (3) [D_angular, D_linear, D_linear] +
                       Stiffness (3) [K_angular, K_linear, K_linear]
- For screw decomposed: [D_parallel, K_parallel, D_perpendicular, K_perpendicular]
Total: 12 dimensions (bimanual SE(2)) or 8 dimensions (bimanual screw)

Note: Actions are scaled and clipped to safe ranges.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from src.envs.biart import BiArtEnv
from src.trajectory_generator import MinimumJerkTrajectory, TrajectoryPoint
from src.ll_controllers.task_space_impedance import TaskSpaceImpedanceController, ImpedanceGains
from src.ll_controllers.se2_screw_decomposed_impedance import (
    SE2ScrewDecomposedImpedanceController,
    ScrewImpedanceParams
)
from src.se2_dynamics import SE2Dynamics, SE2RobotParams


@dataclass
class ImpedanceLearningConfig:
    """Configuration for impedance learning environment."""
    # Controller type: 'se2_impedance' or 'screw_decomposed'
    controller_type: str = 'se2_impedance'

    # Robot parameters (for screw decomposed controller)
    robot_mass: float = 1.0
    robot_inertia: float = 0.1

    # Damping bounds (D)
    min_damping_linear: float = 1.0
    max_damping_linear: float = 50.0
    min_damping_angular: float = 0.5
    max_damping_angular: float = 20.0

    # Stiffness bounds (K)
    min_stiffness_linear: float = 10.0
    max_stiffness_linear: float = 200.0
    min_stiffness_angular: float = 5.0
    max_stiffness_angular: float = 100.0

    # Screw decomposed bounds (parallel and perpendicular)
    min_damping_parallel: float = 1.0
    max_damping_parallel: float = 50.0
    min_stiffness_parallel: float = 5.0
    max_stiffness_parallel: float = 100.0

    min_damping_perpendicular: float = 5.0
    max_damping_perpendicular: float = 100.0
    min_stiffness_perpendicular: float = 20.0
    max_stiffness_perpendicular: float = 500.0

    # Reward weights
    tracking_weight: float = 1.0
    wrench_weight: float = 0.1
    smoothness_weight: float = 0.01

    # Control frequency
    control_dt: float = 0.01  # 100 Hz
    policy_dt: float = 0.1  # 10 Hz (RL policy updates at 10 Hz)

    # Episode settings
    max_episode_steps: int = 1000


class ImpedanceLearningEnv(gym.Env):
    """
    Gym environment for learning impedance parameters.

    The environment assumes a pre-trained high-level policy that generates
    desired poses. The RL agent learns to adjust impedance parameters
    to achieve good tracking while minimizing external forces.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        config: Optional[ImpedanceLearningConfig] = None,
        hl_policy = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize environment.

        Args:
            config: Environment configuration
            hl_policy: Pre-trained high-level policy (FlowMatchingPolicy, DiffusionPolicy, or ACTPolicy)
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config if config is not None else ImpedanceLearningConfig()
        self.hl_policy = hl_policy
        self.render_mode = render_mode

        # Create base environment
        self.base_env = BiArtEnv(render_mode=render_mode)

        # Create impedance controllers for both arms based on controller type
        if self.config.controller_type == 'se2_impedance':
            self.controllers = [
                TaskSpaceImpedanceController(),
                TaskSpaceImpedanceController()
            ]
            self.action_dim_per_arm = 6  # damping (3) + stiffness (3)
        elif self.config.controller_type == 'screw_decomposed':
            # For screw decomposed, we need joint axis screws
            # These will be initialized in reset()
            robot_params = SE2RobotParams(
                mass=self.config.robot_mass,
                inertia=self.config.robot_inertia
            )
            self.controllers = [None, None]  # Will be initialized in reset()
            self.robot_params = robot_params
            self.action_dim_per_arm = 4  # D_parallel, K_parallel, D_perpendicular, K_perpendicular
        else:
            raise ValueError(f"Unknown controller type: {self.config.controller_type}")

        # Trajectory trackers
        self.trajectories = [None, None]
        self.trajectory_time = 0.0

        # Control timestep management
        self.steps_per_policy_update = int(self.config.policy_dt / self.config.control_dt)
        self.control_step_counter = 0

        # Episode counter
        self.episode_steps = 0

        # Previous impedance parameters (for smoothness penalty)
        self.prev_impedance_params = None

        # Define observation space
        # Per arm: external_wrench (3) + current_pose (3) + current_twist (3) +
        #          desired_pose (3) + desired_twist (3) = 15
        # Bimanual: 15 * 2 = 30
        obs_dim = 30
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space
        # SE2 impedance: Per arm: damping (3) + stiffness (3) = 6
        # Screw decomposed: Per arm: D_parallel, K_parallel, D_perp, K_perp = 4
        # Bimanual: action_dim_per_arm * 2
        # Actions are normalized to [-1, 1]
        action_dim = self.action_dim_per_arm * 2
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset base environment
        obs, info = self.base_env.reset(seed=seed, options=options)

        # Reset high-level policy if provided
        if self.hl_policy is not None:
            self.hl_policy.reset()

        # Initialize screw-decomposed controllers if needed
        if self.config.controller_type == 'screw_decomposed':
            # Get joint axis screws from object
            B_left, B_right = self.base_env.object_manager.get_joint_axis_screws()

            # Create controllers with screw axes
            initial_params = ScrewImpedanceParams(
                M_parallel=self.config.robot_mass,
                D_parallel=10.0,
                K_parallel=20.0,
                M_perpendicular=self.config.robot_mass,
                D_perpendicular=20.0,
                K_perpendicular=100.0
            )

            self.controllers[0] = SE2ScrewDecomposedImpedanceController(
                screw_axis=B_left,
                params=initial_params,
                robot_dynamics=SE2Dynamics(self.robot_params),
                model_matching=True,
                use_feedforward=True
            )

            self.controllers[1] = SE2ScrewDecomposedImpedanceController(
                screw_axis=B_right,
                params=initial_params,
                robot_dynamics=SE2Dynamics(self.robot_params),
                model_matching=True,
                use_feedforward=True
            )

        # Reset controllers
        for controller in self.controllers:
            if controller is not None:
                controller.reset()

        # Reset counters
        self.episode_steps = 0
        self.control_step_counter = 0
        self.trajectory_time = 0.0
        self.prev_impedance_params = None

        # Initialize trajectories
        self._update_trajectories(obs)

        # Get initial RL observation
        rl_obs = self._get_rl_observation(obs)

        return rl_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Impedance parameters (normalized to [-1, 1])

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode impedance parameters from action
        impedance_params = self._decode_action(action)

        # Update controller gains
        self._update_controller_gains(impedance_params)

        # Execute multiple control steps per RL policy step
        total_reward = 0.0
        for _ in range(self.steps_per_policy_update):
            # Get current observation
            obs = self.base_env.get_obs()

            # Get desired poses, twists, and accelerations from trajectories
            desired_poses, desired_twists, desired_accels = self._get_trajectory_targets()

            # Compute control wrenches using impedance controllers
            wrenches = []
            for i in range(2):  # For each arm
                if self.config.controller_type == 'se2_impedance':
                    wrench = self.controllers[i].compute_wrench(
                        current_pose=obs['ee_poses'][i],
                        desired_pose=desired_poses[i],
                        measured_wrench=obs['external_wrenches'][i],
                        current_velocity=obs['ee_twists'][i],  # Spatial frame velocity
                        desired_velocity=desired_twists[i],  # Body frame twist
                        desired_acceleration=desired_accels[i]  # Body frame acceleration (feedforward)
                    )
                elif self.config.controller_type == 'screw_decomposed':
                    # Use proper current body twist from observation
                    current_body_twist = obs.get('ee_body_twists', obs['ee_twists'])[i]  # Fallback for compatibility

                    wrench, _ = self.controllers[i].compute_control(
                        current_pose=obs['ee_poses'][i],
                        desired_pose=desired_poses[i],
                        body_twist_current=current_body_twist,  # ✅ Current body twist (MR convention!)
                        body_twist_desired=desired_twists[i],   # Desired body twist
                        body_accel_desired=desired_accels[i],   # Desired body acceleration (feedforward)
                        F_ext=obs['external_wrenches'][i]       # External wrench for impedance modulation
                    )
                wrenches.append(wrench)

            wrenches = np.array(wrenches)

            # Step base environment with wrenches
            obs, _, terminated, truncated, info = self.base_env.step(wrenches)

            # Compute reward
            reward = self._compute_reward(
                obs,
                desired_poses,
                desired_twists,
                impedance_params
            )
            total_reward += reward

            # Update trajectory time
            self.trajectory_time += self.config.control_dt

            # Check termination
            if terminated or truncated:
                break

        # Update episode counter
        self.episode_steps += 1
        self.control_step_counter += self.steps_per_policy_update

        # Check episode timeout
        if self.episode_steps >= self.config.max_episode_steps:
            truncated = True

        # Generate new trajectories periodically
        if self.control_step_counter % 100 == 0:  # Every 1 second
            self._update_trajectories(obs)

        # Get RL observation
        rl_obs = self._get_rl_observation(obs)

        # Store impedance params for next step
        self.prev_impedance_params = impedance_params

        # Average reward over control steps
        avg_reward = total_reward / self.steps_per_policy_update

        return rl_obs, avg_reward, terminated, truncated, info

    def _decode_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decode normalized action to impedance parameters.

        Args:
            action: Normalized action in [-1, 1]

        Returns:
            Dictionary with damping and stiffness for both arms
        """
        if self.config.controller_type == 'se2_impedance':
            return self._decode_action_se2(action)
        elif self.config.controller_type == 'screw_decomposed':
            return self._decode_action_screw(action)
        else:
            raise ValueError(f"Unknown controller type: {self.config.controller_type}")

    def _decode_action_se2(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decode action for SE2 impedance controller.

        Following Modern Robotics convention:
        - Damping: [D_angular, D_linear, D_linear]
        - Stiffness: [K_angular, K_linear, K_linear]

        Action indices:
        - 0: D_angular
        - 1,2: D_linear (x, y)
        - 3: K_angular
        - 4,5: K_linear (x, y)
        """
        # Split action for two arms
        action_arm0 = action[:6]
        action_arm1 = action[6:]

        params = {
            'damping': [],
            'stiffness': []
        }

        for arm_action in [action_arm0, action_arm1]:
            # Damping: [D_angular, D_linear, D_linear] (MR convention!)
            damping = np.array([
                self._scale_action(
                    arm_action[0],
                    self.config.min_damping_angular,
                    self.config.max_damping_angular
                ),
                self._scale_action(
                    arm_action[1],
                    self.config.min_damping_linear,
                    self.config.max_damping_linear
                ),
                self._scale_action(
                    arm_action[2],
                    self.config.min_damping_linear,
                    self.config.max_damping_linear
                )
            ])

            # Stiffness: [K_angular, K_linear, K_linear] (MR convention!)
            stiffness = np.array([
                self._scale_action(
                    arm_action[3],
                    self.config.min_stiffness_angular,
                    self.config.max_stiffness_angular
                ),
                self._scale_action(
                    arm_action[4],
                    self.config.min_stiffness_linear,
                    self.config.max_stiffness_linear
                ),
                self._scale_action(
                    arm_action[5],
                    self.config.min_stiffness_linear,
                    self.config.max_stiffness_linear
                )
            ])

            params['damping'].append(damping)
            params['stiffness'].append(stiffness)

        params['damping'] = np.array(params['damping'])
        params['stiffness'] = np.array(params['stiffness'])

        return params

    def _decode_action_screw(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Decode action for screw-decomposed controller."""
        # Split action for two arms
        action_arm0 = action[:4]
        action_arm1 = action[4:]

        params = {
            'D_parallel': [],
            'K_parallel': [],
            'D_perpendicular': [],
            'K_perpendicular': []
        }

        for arm_action in [action_arm0, action_arm1]:
            # Parallel direction (compliant)
            D_parallel = self._scale_action(
                arm_action[0],
                self.config.min_damping_parallel,
                self.config.max_damping_parallel
            )
            K_parallel = self._scale_action(
                arm_action[1],
                self.config.min_stiffness_parallel,
                self.config.max_stiffness_parallel
            )

            # Perpendicular direction (stiff)
            D_perpendicular = self._scale_action(
                arm_action[2],
                self.config.min_damping_perpendicular,
                self.config.max_damping_perpendicular
            )
            K_perpendicular = self._scale_action(
                arm_action[3],
                self.config.min_stiffness_perpendicular,
                self.config.max_stiffness_perpendicular
            )

            params['D_parallel'].append(D_parallel)
            params['K_parallel'].append(K_parallel)
            params['D_perpendicular'].append(D_perpendicular)
            params['K_perpendicular'].append(K_perpendicular)

        # Convert to arrays
        for key in params:
            params[key] = np.array(params[key])

        return params

    def _scale_action(self, normalized_value: float, min_val: float, max_val: float) -> float:
        """Scale normalized action from [-1, 1] to [min_val, max_val]."""
        return min_val + (normalized_value + 1.0) * 0.5 * (max_val - min_val)

    def _update_controller_gains(self, impedance_params: Dict[str, np.ndarray]):
        """Update impedance controller gains."""
        if self.config.controller_type == 'se2_impedance':
            self._update_se2_gains(impedance_params)
        elif self.config.controller_type == 'screw_decomposed':
            self._update_screw_gains(impedance_params)

    def _update_se2_gains(self, impedance_params: Dict[str, np.ndarray]):
        """
        Update SE2 impedance controller gains.

        Following Modern Robotics convention:
        - damping: [D_angular, D_linear_x, D_linear_y]
        - stiffness: [K_angular, K_linear_x, K_linear_y]

        Note: TaskSpaceImpedanceController uses isotropic linear gains (same for x and y).
        We average the x and y components to maintain backward compatibility.
        """
        for i in range(2):
            damping = impedance_params['damping'][i]
            stiffness = impedance_params['stiffness'][i]

            # Update gains (MR convention: angular at index 0, linear at indices 1,2)
            self.controllers[i].gains.kd_angular = damping[0]  # MR: angular first!
            # Average x and y damping for isotropic controller
            self.controllers[i].gains.kd_linear = (damping[1] + damping[2]) / 2.0
            self.controllers[i].gains.kp_angular = stiffness[0]  # MR: angular first!
            # Average x and y stiffness for isotropic controller
            self.controllers[i].gains.kp_linear = (stiffness[1] + stiffness[2]) / 2.0

    def _update_screw_gains(self, impedance_params: Dict[str, np.ndarray]):
        """Update screw-decomposed controller gains."""
        for i in range(2):
            # Create new screw impedance params
            new_params = ScrewImpedanceParams(
                M_parallel=self.config.robot_mass,
                D_parallel=impedance_params['D_parallel'][i],
                K_parallel=impedance_params['K_parallel'][i],
                M_perpendicular=self.config.robot_mass,
                D_perpendicular=impedance_params['D_perpendicular'][i],
                K_perpendicular=impedance_params['K_perpendicular'][i]
            )

            # Update controller params
            self.controllers[i].params = new_params

    def _update_trajectories(self, obs: Dict[str, np.ndarray]):
        """Update trajectories using high-level policy."""
        if self.hl_policy is None:
            # Use current poses as desired poses (hold position)
            for i in range(2):
                current_pose = obs['ee_poses'][i]
                # Small perturbation for exploration
                target_pose = current_pose + np.random.randn(3) * 0.01

                self.trajectories[i] = MinimumJerkTrajectory(
                    start_pose=current_pose,
                    end_pose=target_pose,
                    duration=1.0
                )
        else:
            # Get desired poses from high-level policy
            hl_obs = {
                'ee_poses': obs['ee_poses'],
                'link_poses': obs['link_poses'],
                'external_wrenches': obs['external_wrenches']
            }
            desired_poses = self.hl_policy.get_action(hl_obs)

            # Create trajectories for both arms
            for i in range(2):
                current_pose = obs['ee_poses'][i]
                target_pose = desired_poses[i]

                self.trajectories[i] = MinimumJerkTrajectory(
                    start_pose=current_pose,
                    end_pose=target_pose,
                    duration=1.0
                )

        self.trajectory_time = 0.0

    def _get_trajectory_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get desired poses, twists, and accelerations from trajectories.

        Returns:
            Tuple of (desired_poses, desired_twists, desired_accelerations) where:
                - desired_poses: (2, 3) array in spatial frame (T_si^des)
                - desired_twists: (2, 3) array in body frame (i^V_i^des)
                - desired_accelerations: (2, 3) array in body frame (i^dV_i^des)
        """
        from src.se2_math import world_to_body_acceleration

        desired_poses = []
        desired_twists = []
        desired_accelerations = []

        for i in range(2):
            if self.trajectories[i] is None:
                # Fallback to zero
                desired_poses.append(np.zeros(3))
                desired_twists.append(np.zeros(3))
                desired_accelerations.append(np.zeros(3))
            else:
                traj_point = self.trajectories[i].evaluate(self.trajectory_time)
                # Pose in spatial frame
                desired_poses.append(traj_point.pose)
                # Twist in BODY frame (this is what we want!)
                desired_twists.append(traj_point.velocity_body)
                # Acceleration in BODY frame (transform from spatial)
                accel_body = world_to_body_acceleration(
                    pose=traj_point.pose,
                    vel_world=traj_point.velocity_spatial,
                    accel_world=traj_point.acceleration
                )
                desired_accelerations.append(accel_body)

        return np.array(desired_poses), np.array(desired_twists), np.array(desired_accelerations)

    def _get_rl_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Construct RL observation from environment observation.

        Observation includes:
        - External wrenches (2, 3) - MR convention: [tau, fx, fy]
        - Current poses (2, 3)
        - Current body twists (2, 3) - MR convention: [omega, vx_b, vy_b]
        - Desired poses (2, 3)
        - Desired twists (2, 3) - MR convention: [omega, vx_b, vy_b]
        """
        desired_poses, desired_twists, _ = self._get_trajectory_targets()

        # Get current body twists (use new field if available, fallback to spatial for compatibility)
        current_twists = obs.get('ee_body_twists', obs.get('ee_twists', np.zeros((2, 3))))

        # Concatenate all observations
        rl_obs = np.concatenate([
            obs['external_wrenches'].flatten(),  # 6 - [tau, fx, fy] for both arms
            obs['ee_poses'].flatten(),  # 6
            current_twists.flatten(),  # 6 - [omega, vx_b, vy_b] for both arms
            desired_poses.flatten(),  # 6
            desired_twists.flatten()  # 6 - [omega, vx_b, vy_b] for both arms
        ])

        return rl_obs.astype(np.float32)

    def _compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        desired_poses: np.ndarray,
        desired_twists: np.ndarray,
        impedance_params: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute reward.

        Reward components:
        1. Tracking error (negative)
        2. External wrench magnitude (negative)
        3. Smoothness penalty (negative change in impedance params)
        """
        reward = 0.0

        # 1. Tracking error
        tracking_error = 0.0
        for i in range(2):
            pose_error = np.linalg.norm(obs['ee_poses'][i] - desired_poses[i])
            twist_error = np.linalg.norm(obs.get('ee_twists', np.zeros((2, 3)))[i] - desired_twists[i])
            tracking_error += pose_error + 0.1 * twist_error

        reward -= self.config.tracking_weight * tracking_error

        # 2. External wrench minimization
        wrench_magnitude = 0.0
        for i in range(2):
            wrench_magnitude += np.linalg.norm(obs['external_wrenches'][i])

        reward -= self.config.wrench_weight * wrench_magnitude

        # 3. Smoothness penalty
        if self.prev_impedance_params is not None:
            damping_change = np.linalg.norm(
                impedance_params['damping'] - self.prev_impedance_params['damping']
            )
            stiffness_change = np.linalg.norm(
                impedance_params['stiffness'] - self.prev_impedance_params['stiffness']
            )
            smoothness_penalty = damping_change + stiffness_change
            reward -= self.config.smoothness_weight * smoothness_penalty

        return reward

    def render(self):
        """Render environment."""
        return self.base_env.render()

    def close(self):
        """Close environment."""
        self.base_env.close()
