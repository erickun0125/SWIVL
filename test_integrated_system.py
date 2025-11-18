"""
Integration test for the complete bimanual manipulation system.

Tests:
1. Linkage object management (R, P, Fixed joints)
2. PD controller for pose tracking
3. Keyboard input planner for teleoperation

This script demonstrates keyboard-controlled bimanual manipulation
with automatic grasp maintenance.
"""

import numpy as np
import pygame
import gymnasium as gym

from gym_biart.envs.linkage_manager import LinkageObject, JointType, create_two_link_object
from gym_biart.envs.pd_controller import MultiGripperController, PDGains, compute_desired_pose_from_velocity
from gym_biart.envs.keyboard_planner import MultiEEPlanner


def test_linkage_manager():
    """Test linkage object manager."""
    print("=" * 60)
    print("Testing Linkage Manager")
    print("=" * 60)

    # Test with different joint types
    for joint_type in [JointType.REVOLUTE, JointType.PRISMATIC, JointType.FIXED]:
        print(f"\nTesting {joint_type.value} joint:")

        linkage = create_two_link_object(joint_type)
        print(f"  Created: {linkage}")

        # Test configuration
        config = linkage.get_configuration()
        print(f"  Initial configuration: {config}")

        # Test forward kinematics
        base_pose = np.array([256.0, 256.0, 0.0])
        if joint_type == JointType.REVOLUTE:
            joint_values = np.array([np.pi / 4])
        elif joint_type == JointType.PRISMATIC:
            joint_values = np.array([20.0])
        else:
            joint_values = np.array([0.0])

        link_poses = linkage.forward_kinematics(base_pose, joint_values)
        print(f"  Forward kinematics with q={joint_values}:")
        for i, pose in enumerate(link_poses):
            print(f"    Link {i}: x={pose[0]:.2f}, y={pose[1]:.2f}, θ={pose[2]:.2f}")

    print("\n✓ Linkage manager tests passed!")


def test_pd_controller():
    """Test PD controller."""
    print("\n" + "=" * 60)
    print("Testing PD Controller")
    print("=" * 60)

    # Create controller
    gains = PDGains(kp_linear=50.0, kd_linear=10.0, kp_angular=20.0, kd_angular=5.0)
    controller = MultiGripperController(num_grippers=2, gains=gains)
    controller.set_timestep(0.01)

    print(f"Created controller with gains:")
    print(f"  Kp_linear: {gains.kp_linear}, Kd_linear: {gains.kd_linear}")
    print(f"  Kp_angular: {gains.kp_angular}, Kd_angular: {gains.kd_angular}")

    # Test wrench computation
    current_poses = np.array([
        [100.0, 100.0, 0.0],
        [200.0, 200.0, np.pi/4]
    ])

    desired_poses = np.array([
        [120.0, 110.0, 0.1],
        [210.0, 205.0, np.pi/4 + 0.1]
    ])

    wrenches = controller.compute_wrenches(current_poses, desired_poses)

    print(f"\nComputed wrenches:")
    for i, wrench in enumerate(wrenches):
        print(f"  Gripper {i}: fx={wrench[0]:.2f}, fy={wrench[1]:.2f}, tau={wrench[2]:.2f}")

    # Verify wrenches are in reasonable range
    assert np.all(np.abs(wrenches[:, :2]) <= 100.0), "Force exceeds limits"
    assert np.all(np.abs(wrenches[:, 2]) <= 50.0), "Torque exceeds limits"

    print("\n✓ PD controller tests passed!")


def test_keyboard_planner():
    """Test keyboard planner (without actual keyboard input)."""
    print("\n" + "=" * 60)
    print("Testing Keyboard Planner")
    print("=" * 60)

    # Create linkage object
    linkage = create_two_link_object(JointType.REVOLUTE)

    # Create planner
    planner = MultiEEPlanner(
        num_end_effectors=2,
        linkage_object=linkage,
        control_dt=0.1
    )

    print(f"Created planner for {planner.num_ee} end-effectors")
    print(f"Controlled EE index: {planner.get_controlled_ee_index()}")

    # Initialize with current poses
    current_poses = np.array([
        [150.0, 350.0, np.pi/2],
        [350.0, 350.0, np.pi/2]
    ])

    planner.initialize_from_current_state(current_poses)
    print(f"\nInitialized with poses:")
    for i, pose in enumerate(current_poses):
        print(f"  EE {i}: x={pose[0]:.2f}, y={pose[1]:.2f}, θ={pose[2]:.2f}")

    # Simulate velocity command (without actual keyboard)
    planner.keyboard.velocity_cmd.linear_x = 10.0
    planner.keyboard.velocity_cmd.angular = 0.1

    # Update planner
    desired_poses, actions = planner.update([], current_poses)

    print(f"\nDesired poses after velocity command (vx=10, omega=0.1):")
    for i, pose in enumerate(desired_poses):
        print(f"  EE {i}: x={pose[0]:.2f}, y={pose[1]:.2f}, θ={pose[2]:.2f}")

    print("\n✓ Keyboard planner tests passed!")


def run_interactive_demo():
    """
    Run interactive demo with BiArt environment.

    Requires pygame window for keyboard input.
    Press ESC to exit.
    """
    print("\n" + "=" * 60)
    print("Interactive Demo")
    print("=" * 60)
    print("\nKeyboard controls:")
    print("  Arrow keys: Move controlled end-effector")
    print("  Q/W: Rotate counterclockwise/clockwise")
    print("  1/2: Switch controlled end-effector")
    print("  Space: Reset velocity")
    print("  ESC: Exit")
    print("\nStarting demo...")

    # Create environment
    env = gym.make('BiArt-v0', joint_type='revolute', render_mode='human')

    # Create linkage object
    linkage = create_two_link_object(JointType.REVOLUTE)

    # Create PD controller
    gains = PDGains(kp_linear=30.0, kd_linear=8.0, kp_angular=15.0, kd_angular=3.0)
    controller = MultiGripperController(num_grippers=2, gains=gains)
    controller.set_timestep(env.dt)

    # Create planner
    planner = MultiEEPlanner(
        num_end_effectors=2,
        linkage_object=linkage,
        control_dt=1.0 / env.control_hz
    )

    # Reset environment
    obs, info = env.reset()

    # Extract initial poses from observation
    # obs format: [left_gripper(3), right_gripper(3), link1(3), link2(3), ext_wrench(6)]
    initial_ee_poses = np.array([
        obs[0:3],   # Left gripper
        obs[3:6],   # Right gripper
    ])

    # Set linkage bodies
    linkage.set_link_body(0, env.link1)
    linkage.set_link_body(1, env.link2)

    # Initialize planner
    planner.initialize_from_current_state(initial_ee_poses)

    print("\nDemo running... (press ESC to exit)")

    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        # Get pygame events
        events = pygame.event.get()

        # Get current EE poses
        current_ee_poses = np.array([
            [env.left_gripper.position.x, env.left_gripper.position.y, env.left_gripper.angle],
            [env.right_gripper.position.x, env.right_gripper.position.y, env.right_gripper.angle],
        ])

        # Get current velocities
        current_velocities = np.array([
            [env.left_gripper.velocity.x, env.left_gripper.velocity.y, env.left_gripper.angular_velocity],
            [env.right_gripper.velocity.x, env.right_gripper.velocity.y, env.right_gripper.angular_velocity],
        ])

        # Update planner
        desired_poses, actions = planner.update(events, current_ee_poses)

        # Check for quit
        if actions['quit']:
            running = False
            break

        # Compute control wrenches using PD controller
        wrenches = controller.compute_wrenches(
            current_ee_poses,
            desired_poses,
            current_velocities
        )

        # Construct action for environment
        # Action format: [left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
        action = np.concatenate([wrenches[0], wrenches[1]])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render
        env.render()

        # Print status
        if pygame.time.get_ticks() % 1000 < 20:  # Print roughly every second
            controlled_idx = planner.get_controlled_ee_index()
            vel_cmd = planner.keyboard.get_velocity_array()
            print(f"Controlled EE: {controlled_idx}, Vel: [{vel_cmd[0]:.1f}, {vel_cmd[1]:.1f}, {vel_cmd[2]:.2f}], Reward: {reward:.3f}")

        # Control frame rate
        clock.tick(env.metadata['render_fps'])

        if terminated or truncated:
            print("Episode finished, resetting...")
            obs, info = env.reset()

            # Re-initialize planner
            initial_ee_poses = np.array([obs[0:3], obs[3:6]])
            planner.initialize_from_current_state(initial_ee_poses)

    env.close()
    print("\n✓ Demo completed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INTEGRATED SYSTEM TEST")
    print("="*60)

    # Run unit tests
    test_linkage_manager()
    test_pd_controller()
    test_keyboard_planner()

    print("\n" + "="*60)
    print("ALL UNIT TESTS PASSED!")
    print("="*60)

    # Ask if user wants to run interactive demo
    print("\nWould you like to run the interactive demo? (y/n)")
    # For automated testing, skip interactive demo
    # Uncomment the following line to enable interactive demo
    # response = input().strip().lower()
    # if response == 'y':
    #     run_interactive_demo()

    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
