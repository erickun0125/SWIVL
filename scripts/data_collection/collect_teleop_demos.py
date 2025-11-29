"""
Teleoperation Data Collection Script for BiArtEnv

This script allows a user to control the BiArtEnv using keyboard inputs and saves
the resulting trajectories to an HDF5 file for training high-level policies.

Usage:
    python scripts/data_collection/collect_teleop_demos.py --output data/demos.h5 --num_demos 10

Controls:
    Arrow Keys: Move controlled EE (Linear)
    Q / W: Rotate controlled EE (Angular)
    1 / 2: Switch controlled EE (Left / Right)
    Space: Stop
    ESC: Quit / Finish Episode
"""

import os
import sys
import argparse
import time
import numpy as np
import h5py
import pygame
import cv2

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.envs.biart import BiArtEnv
from src.hl_planners.keyboard_teleoperation import KeyboardTeleoperationPlanner, TeleopConfig
from src.ll_controllers.se2_impedance_controller import SE2ImpedanceController
from src.se2_dynamics import SE2RobotParams, SE2Dynamics
from src.se2_math import world_to_body_velocity

def collect_demos(output_path: str, num_demos: int, max_steps: int = 500):
    """
    Collect teleoperation demonstrations.

    Args:
        output_path: Path to save HDF5 file
        num_demos: Number of demonstrations to collect
        max_steps: Maximum steps per episode
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize Environment
    env = BiArtEnv(render_mode="human", control_hz=10)
    
    # Initialize Planner (High-Level)
    teleop_config = TeleopConfig(
        linear_speed=50.0,
        angular_speed=1.5,
        joint_speed=0.5,
        control_dt=1.0/env.control_hz,
        controlled_gripper="left"
    )
    planner = KeyboardTeleoperationPlanner(
        config=teleop_config,
        joint_type="revolute", # Default
        link_length=env.object_config.link_length
    )

    # Initialize Controller (Low-Level)
    # We use impedance controller to track the desired poses from teleop
    robot_params = SE2RobotParams(mass=1.0, inertia=0.1)
    controller = SE2ImpedanceController.create_diagonal_impedance(
        I_d=0.1,
        m_d=1.0,
        d_theta=2.0,
        d_x=20.0,
        d_y=20.0,
        k_theta=50.0,
        k_x=500.0,
        k_y=500.0,
        robot_params=robot_params,
        model_matching=True,
        max_force=100.0,
        max_torque=50.0
    )

    # Data buffer
    all_demos = []

    print(f"Starting data collection. Target: {num_demos} demos.")
    print("Controls: Arrows (Move), Q/W (Rotate), 1/2 (Switch Hand), ESC (Finish Demo)")

    for demo_idx in range(num_demos):
        print(f"\n=== Recording Demo {demo_idx + 1}/{num_demos} ===")
        
        obs, _ = env.reset()
        planner.reset(obs['ee_poses'], obs['link_poses'])
        
        # Episode storage
    episode_obs = {
        'ee_poses': [],
        'link_poses': [],
        'external_wrenches': [],
        'ee_body_twists': [],
        'images': []  # Optional: if pixels obs_type used
    }
    episode_actions = {
        'desired_poses': [],
        'desired_body_twists': []
    }
        
    step_count = 0
    done = False
    
    while not done and step_count < max_steps:
        # 1. Handle Pygame Events (Keyboard)
        events = pygame.event.get()
        keys = pygame.key.get_pressed()
        
        # Quit check
        for event in events:
            if event.type == pygame.QUIT:
                print("Collection aborted.")
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True # Finish this episode manually
                elif event.key == pygame.K_1:
                    planner.config.controlled_gripper = "left"
                    print("Controlling Left Gripper")
                elif event.key == pygame.K_2:
                    planner.config.controlled_gripper = "right"
                    print("Controlling Right Gripper")

        # 2. Get High-Level Action (Desired Poses) from Teleop Planner
        # Note: planner expects dict {keycode: bool} for keys usually, but process_keyboard_input uses internal map
        # Let's construct the key map expected by the planner
        key_map = {
            pygame.K_UP: keys[pygame.K_UP],
            pygame.K_DOWN: keys[pygame.K_DOWN],
            pygame.K_LEFT: keys[pygame.K_LEFT],
            pygame.K_RIGHT: keys[pygame.K_RIGHT],
            pygame.K_q: keys[pygame.K_q],
            pygame.K_w: keys[pygame.K_w],
            pygame.K_a: keys[pygame.K_a],
            pygame.K_d: keys[pygame.K_d],
        }
        
        hl_action = planner.get_action(
            keyboard_events=key_map,
            current_ee_poses=obs['ee_poses'],
            current_link_poses=obs['link_poses'],
            current_ee_velocities=obs['ee_velocities'],  # point velocities [vx, vy, omega]
            current_link_velocities=None  # Optional
        )
        
        desired_poses = hl_action['desired_poses']  # (2, 3)
        
        # 3. Get Low-Level Action (Wrenches) from Controller
        # Desired velocity is also available from planner for feedforward D term
        desired_body_twists = hl_action['desired_body_twists']
        
        # Convert point velocities to body twists
        # NOTE: ee_velocities is [vx, vy, omega] in world frame (NOT a twist!)
        # We convert to body twist [omega, vx_b, vy_b] for the controller
        current_body_twists = np.zeros((2, 3))
        for i in range(2):
            point_vel = obs['ee_velocities'][i]  # [vx, vy, omega]
            # Reorder to MR convention [omega, vx, vy] then transform
            spatial_mr = np.array([point_vel[2], point_vel[0], point_vel[1]])
            current_body_twists[i] = world_to_body_velocity(obs['ee_poses'][i], spatial_mr)

        # Compute wrenches for each end-effector separately
        wrenches = []
        for i in range(2):
            wrench, _ = controller.compute_control(
                current_pose=obs['ee_poses'][i],
                desired_pose=desired_poses[i],
                body_twist_current=current_body_twists[i],
                body_twist_desired=desired_body_twists[i]
            )
            wrenches.append(wrench)
        wrenches = np.array(wrenches)
        
        # 4. Step Environment
        next_obs, _, terminated, truncated, _ = env.step(wrenches)
        
        # 5. Record Data (s_t, a_t) -> We record OBSERVATION and HIGH-LEVEL ACTION (Desired Pose)
        # Ideally we record the observation used to generate the action
        episode_obs['ee_poses'].append(obs['ee_poses'])
        episode_obs['link_poses'].append(obs['link_poses'])
        episode_obs['external_wrenches'].append(obs['external_wrenches'])
        if 'ee_body_twists' in obs:
            episode_obs['ee_body_twists'].append(obs['ee_body_twists'])
        else:
            episode_obs['ee_body_twists'].append(np.zeros((2, 3)))
        episode_actions['desired_poses'].append(desired_poses)
        episode_actions['desired_body_twists'].append(desired_body_twists)
        
        obs = next_obs
        step_count += 1
        
        if terminated or truncated:
            done = True
            
        # Render
        env.render()
        
    # End of episode
    print(f"Episode finished. Steps: {step_count}")
    
    if step_count > 10: # Only save meaningful episodes
        all_demos.append({
            'obs': episode_obs,
            'action': episode_actions
        })
    else:
        print("Episode too short, discarding.")
            
    # Save to HDF5
    print(f"\nSaving {len(all_demos)} demos to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['env_name'] = 'BiArtEnv'
        f.attrs['num_demos'] = len(all_demos)
        
        for i, demo in enumerate(all_demos):
            g = f.create_group(f'demo_{i}')
            
            # Observations group
            obs_g = g.create_group('obs')
            obs_g.create_dataset('ee_poses', data=np.array(demo['obs']['ee_poses']))
            obs_g.create_dataset('link_poses', data=np.array(demo['obs']['link_poses']))
            obs_g.create_dataset('external_wrenches', data=np.array(demo['obs']['external_wrenches']))
            obs_g.create_dataset('ee_body_twists', data=np.array(demo['obs']['ee_body_twists']))

            # Actions dataset (Desired Poses + Twists)
            action_g = g.create_group('action')
            action_g.create_dataset('desired_poses', data=np.array(demo['action']['desired_poses']))
            action_g.create_dataset('desired_body_twists', data=np.array(demo['action']['desired_body_twists']))
            
            # Length
            g.attrs['num_samples'] = len(demo['action']['desired_poses'])
            
    env.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/teleop_demos.h5')
    parser.add_argument('--num_demos', type=int, default=5)
    args = parser.parse_args()
    
    collect_demos(args.output, args.num_demos)

