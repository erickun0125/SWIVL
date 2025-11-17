# Usage Examples for Bimanual Manipulation System

This document provides examples of how to use the newly implemented components for bimanual manipulation.

## Components

### 1. Linkage Manager (`gym_biart/envs/linkage_manager.py`)

Manages articulated linkage objects with different joint types (Revolute, Prismatic, Fixed).

```python
from gym_biart.envs.linkage_manager import LinkageObject, JointType, create_two_link_object

# Create a two-link object with revolute joint
linkage = create_two_link_object(JointType.REVOLUTE)

# Set linkage bodies from pymunk simulation
linkage.set_link_body(0, link1_body)
linkage.set_link_body(1, link2_body)

# Update joint states
linkage.update_joint_states()

# Get current configuration
config = linkage.get_configuration()  # Returns joint values

# Get all link poses
poses = linkage.get_all_link_poses()  # Returns (num_links, 3) array

# Forward kinematics
base_pose = np.array([256.0, 256.0, 0.0])
joint_values = np.array([np.pi/4])
link_poses = linkage.forward_kinematics(base_pose, joint_values)

# Get suggested grasp poses
grasp_poses = linkage.compute_grasp_poses(num_grippers=2)
```

### 2. PD Controller (`gym_biart/envs/pd_controller.py`)

Naive PD controller that takes desired pose and outputs wrench commands.

```python
from gym_biart.envs.pd_controller import MultiGripperController, PDGains

# Create controller with custom gains
gains = PDGains(
    kp_linear=50.0,
    kd_linear=10.0,
    kp_angular=20.0,
    kd_angular=5.0
)

controller = MultiGripperController(
    num_grippers=2,
    gains=gains,
    max_force=100.0,
    max_torque=50.0
)

controller.set_timestep(0.01)

# Compute control wrenches
current_poses = np.array([
    [100.0, 100.0, 0.0],      # Gripper 0: [x, y, theta]
    [200.0, 200.0, np.pi/4]   # Gripper 1: [x, y, theta]
])

desired_poses = np.array([
    [120.0, 110.0, 0.1],
    [210.0, 205.0, np.pi/4 + 0.1]
])

current_velocities = np.array([
    [10.0, 5.0, 0.0],   # Gripper 0: [vx, vy, omega]
    [8.0, 3.0, 0.1]     # Gripper 1: [vx, vy, omega]
])

# Returns (num_grippers, 3) array with [fx, fy, tau] for each gripper
wrenches = controller.compute_wrenches(
    current_poses,
    desired_poses,
    current_velocities  # Optional
)

# Use wrenches as action for environment
action = np.concatenate([wrenches[0], wrenches[1]])
```

### 3. Keyboard Planner (`gym_biart/envs/keyboard_planner.py`)

High-level planner that maps keyboard input to desired poses.

```python
import pygame
from gym_biart.envs.keyboard_planner import MultiEEPlanner
from gym_biart.envs.linkage_manager import create_two_link_object, JointType

# Initialize pygame
pygame.init()

# Create linkage object
linkage = create_two_link_object(JointType.REVOLUTE)
linkage.set_link_body(0, link1_body)
linkage.set_link_body(1, link2_body)

# Create planner
planner = MultiEEPlanner(
    num_end_effectors=2,
    linkage_object=linkage,
    control_dt=0.1
)

# Initialize with current poses
current_ee_poses = np.array([
    [150.0, 350.0, np.pi/2],
    [350.0, 350.0, np.pi/2]
])
planner.initialize_from_current_state(current_ee_poses)

# In your main loop:
while running:
    events = pygame.event.get()

    # Update planner with keyboard input
    desired_poses, actions = planner.update(events, current_ee_poses)

    # Check for quit
    if actions['quit']:
        break

    # Use desired poses for control
    # ...
```

## Keyboard Controls

- **Arrow Keys**: Linear velocity
  - Up: Forward (+x in body frame)
  - Down: Backward (-x in body frame)
  - Left: Left (+y in body frame)
  - Right: Right (-y in body frame)

- **Q**: Rotate counterclockwise (+angular velocity)
- **W**: Rotate clockwise (-angular velocity)

- **1/2**: Switch controlled end-effector (0 or 1)
- **Space**: Reset velocity to zero
- **ESC**: Exit

## Complete Integration Example

```python
import numpy as np
import pygame
import gymnasium as gym

from gym_biart.envs.linkage_manager import create_two_link_object, JointType
from gym_biart.envs.pd_controller import MultiGripperController, PDGains
from gym_biart.envs.keyboard_planner import MultiEEPlanner

# Create environment
env = gym.make('BiArt-v0', joint_type='revolute', render_mode='human')

# Create linkage object
linkage = create_two_link_object(JointType.REVOLUTE)

# Create PD controller
gains = PDGains(kp_linear=30.0, kd_linear=8.0, kp_angular=15.0, kd_angular=3.0)
controller = MultiGripperController(num_grippers=2, gains=gains)
controller.set_timestep(env.dt)

# Create keyboard planner
planner = MultiEEPlanner(
    num_end_effectors=2,
    linkage_object=linkage,
    control_dt=1.0 / env.control_hz
)

# Reset environment
obs, info = env.reset()

# Initialize
initial_ee_poses = np.array([obs[0:3], obs[3:6]])
planner.initialize_from_current_state(initial_ee_poses)

# Set linkage bodies
linkage.set_link_body(0, env.link1)
linkage.set_link_body(1, env.link2)

# Main loop
clock = pygame.time.Clock()
running = True

while running:
    # Get events
    events = pygame.event.get()

    # Current state
    current_ee_poses = np.array([
        [env.left_gripper.position.x, env.left_gripper.position.y, env.left_gripper.angle],
        [env.right_gripper.position.x, env.right_gripper.position.y, env.right_gripper.angle],
    ])

    current_velocities = np.array([
        [env.left_gripper.velocity.x, env.left_gripper.velocity.y, env.left_gripper.angular_velocity],
        [env.right_gripper.velocity.x, env.right_gripper.velocity.y, env.right_gripper.angular_velocity],
    ])

    # Update planner
    desired_poses, actions = planner.update(events, current_ee_poses)

    if actions['quit']:
        break

    # Compute control wrenches
    wrenches = controller.compute_wrenches(
        current_ee_poses,
        desired_poses,
        current_velocities
    )

    # Apply action
    action = np.concatenate([wrenches[0], wrenches[1]])
    obs, reward, terminated, truncated, info = env.step(action)

    # Render
    env.render()
    clock.tick(env.metadata['render_fps'])

    if terminated or truncated:
        obs, info = env.reset()
        initial_ee_poses = np.array([obs[0:3], obs[3:6]])
        planner.initialize_from_current_state(initial_ee_poses)

env.close()
```

## Testing

Run the integrated test suite:

```bash
python test_integrated_system.py
```

This will test all three components:
1. Linkage manager with R, P, and Fixed joints
2. PD controller for wrench computation
3. Keyboard planner for teleoperation

## Notes

- The keyboard planner automatically maintains grasp on non-controlled end-effectors
- PD controller operates in body frame for better control
- Linkage manager supports forward kinematics for planning
- All components are designed to work together seamlessly
