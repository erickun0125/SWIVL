"""
Test script for BiArt environment with proper external wrench sensing and grip initialization.

This script tests:
1. External wrench sensing (force measurement in body frame)
2. Gripper initialization (grippers start holding the object)
3. Grip constraints
"""

import sys
sys.path.insert(0, '.')

import gymnasium as gym
import numpy as np
import gym_biart


def test_external_wrench_sensing():
    """Test that external wrench sensing is working."""
    print("=" * 60)
    print("Testing External Wrench Sensing")
    print("=" * 60)

    env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type="revolute")
    obs, info = env.reset(seed=42)

    print("\nInitial observation (with external wrenches):")
    print(f"  Left gripper pose: {obs[0:3]}")
    print(f"  Right gripper pose: {obs[3:6]}")
    print(f"  Link1 pose: {obs[6:9]}")
    print(f"  Link2 pose: {obs[9:12]}")
    print(f"  External wrench left: {obs[12:15]}")
    print(f"  External wrench right: {obs[15:18]}")

    # Apply forces and measure external wrenches
    print("\nApplying forces and measuring external wrenches...")

    for step in range(10):
        # Apply constant force
        action = np.array([20.0, 0.0, 0.0, -20.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        ext_wrench_left = obs[12:15]
        ext_wrench_right = obs[15:18]

        if step % 3 == 0:
            print(f"\nStep {step + 1}:")
            print(f"  External wrench left: [{ext_wrench_left[0]:8.3f}, {ext_wrench_left[1]:8.3f}, {ext_wrench_left[2]:8.3f}]")
            print(f"  External wrench right: [{ext_wrench_right[0]:8.3f}, {ext_wrench_right[1]:8.3f}, {ext_wrench_right[2]:8.3f}]")
            print(f"  Magnitude left: {np.linalg.norm(ext_wrench_left):8.3f}")
            print(f"  Magnitude right: {np.linalg.norm(ext_wrench_right):8.3f}")

    env.close()
    print("\n✓ External wrench sensing test completed!")


def test_grip_initialization():
    """Test that grippers start holding the object."""
    print("\n" + "=" * 60)
    print("Testing Grip Initialization")
    print("=" * 60)

    env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type="revolute")

    # Test multiple resets
    for trial in range(3):
        obs, info = env.reset(seed=trial)

        left_pos = obs[0:2]
        right_pos = obs[3:5]
        link1_pos = obs[6:8]
        link2_pos = obs[9:11]

        # Calculate distances
        dist_left_to_link1 = np.linalg.norm(left_pos - link1_pos)
        dist_right_to_link2 = np.linalg.norm(right_pos - link2_pos)

        print(f"\nTrial {trial + 1}:")
        print(f"  Left gripper pos: ({left_pos[0]:.1f}, {left_pos[1]:.1f})")
        print(f"  Link1 pos: ({link1_pos[0]:.1f}, {link1_pos[1]:.1f})")
        print(f"  Distance left-to-link1: {dist_left_to_link1:.2f} pixels")
        print(f"  ")
        print(f"  Right gripper pos: ({right_pos[0]:.1f}, {right_pos[1]:.1f})")
        print(f"  Link2 pos: ({link2_pos[0]:.1f}, {link2_pos[1]:.1f})")
        print(f"  Distance right-to-link2: {dist_right_to_link2:.2f} pixels")

        # Check if grippers are close to links (should be very close due to constraints)
        if dist_left_to_link1 < 20 and dist_right_to_link2 < 20:
            print(f"  ✓ Grippers are properly initialized near the objects!")
        else:
            print(f"  ✗ Warning: Grippers may not be properly grasping!")

        # Run a few steps with zero action and check if grip is maintained
        print(f"\n  Running 10 steps with zero action to test grip stability...")
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6))

        left_pos_after = obs[0:2]
        right_pos_after = obs[3:5]
        link1_pos_after = obs[6:8]
        link2_pos_after = obs[9:11]

        dist_left_after = np.linalg.norm(left_pos_after - link1_pos_after)
        dist_right_after = np.linalg.norm(right_pos_after - link2_pos_after)

        print(f"  After 10 steps:")
        print(f"    Distance left-to-link1: {dist_left_after:.2f} pixels")
        print(f"    Distance right-to-link2: {dist_right_after:.2f} pixels")

        if dist_left_after < 20 and dist_right_after < 20:
            print(f"    ✓ Grip is stable! Constraints are working.")
        else:
            print(f"    ✗ Warning: Grip may be unstable!")

    env.close()
    print("\n✓ Grip initialization test completed!")


def test_different_joint_types():
    """Test grip initialization for different joint types."""
    print("\n" + "=" * 60)
    print("Testing Grip Initialization for Different Joint Types")
    print("=" * 60)

    for joint_type in ["revolute", "prismatic", "fixed"]:
        print(f"\n--- Testing {joint_type.upper()} joint ---")

        env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type=joint_type)
        obs, info = env.reset(seed=42)

        left_pos = obs[0:2]
        right_pos = obs[3:5]
        link1_pos = obs[6:8]
        link2_pos = obs[9:11]

        dist_left = np.linalg.norm(left_pos - link1_pos)
        dist_right = np.linalg.norm(right_pos - link2_pos)

        print(f"  Distance left-to-link1: {dist_left:.2f} pixels")
        print(f"  Distance right-to-link2: {dist_right:.2f} pixels")

        if dist_left < 20 and dist_right < 20:
            print(f"  ✓ {joint_type} joint: Grip initialized correctly")
        else:
            print(f"  ✗ {joint_type} joint: Grip may have issues")

        env.close()

    print("\n✓ All joint types tested!")


def test_wrench_under_different_actions():
    """Test external wrench under different action patterns."""
    print("\n" + "=" * 60)
    print("Testing External Wrench Under Different Actions")
    print("=" * 60)

    env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type="revolute")
    obs, info = env.reset(seed=42)

    test_actions = [
        ("Push right", np.array([20.0, 0.0, 0.0, -20.0, 0.0, 0.0])),
        ("Push up", np.array([0.0, 20.0, 0.0, 0.0, 20.0, 0.0])),
        ("Rotate CW", np.array([0.0, 0.0, 10.0, 0.0, 0.0, -10.0])),
        ("Zero force", np.zeros(6)),
    ]

    for action_name, action in test_actions:
        print(f"\n{action_name}:")

        # Apply action for a few steps
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action)

        ext_wrench_left = obs[12:15]
        ext_wrench_right = obs[15:18]

        print(f"  Left wrench: [{ext_wrench_left[0]:7.2f}, {ext_wrench_left[1]:7.2f}, {ext_wrench_left[2]:7.2f}]")
        print(f"  Right wrench: [{ext_wrench_right[0]:7.2f}, {ext_wrench_right[1]:7.2f}, {ext_wrench_right[2]:7.2f}]")
        print(f"  Magnitude left: {np.linalg.norm(ext_wrench_left):7.2f}")
        print(f"  Magnitude right: {np.linalg.norm(ext_wrench_right):7.2f}")

    env.close()
    print("\n✓ Action-wrench test completed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BiArt Environment - External Wrench & Grip Test Suite")
    print("=" * 60)

    # Run all tests
    test_grip_initialization()
    test_external_wrench_sensing()
    test_different_joint_types()
    test_wrench_under_different_actions()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
