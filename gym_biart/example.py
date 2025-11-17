"""
Example script for testing the BiArt environment.

This script demonstrates how to use the BiArt environment for bimanual
manipulation of articulated objects in SE(2).
"""

import gymnasium as gym
import numpy as np
import gym_biart


def test_random_actions():
    """Test the environment with random actions."""
    print("Testing BiArt environment with random actions...")

    # Create environment
    env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type="revolute")

    # Reset environment
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run episode
    episode_reward = 0
    for step in range(500):
        # Random wrench commands
        action = env.action_space.sample()

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Render
        env.render()

        # Print info every 50 steps
        if step % 50 == 0:
            print(f"Step {step}:")
            print(f"  Reward: {reward:.4f}")
            print(f"  Tracking reward: {info['tracking_reward']:.4f}")
            print(f"  Safety penalty: {info['safety_penalty']:.4f}")
            print(f"  Position error: {info['pos_error']:.2f}")
            print(f"  Contacts: {info['n_contacts']}")

        # Reset if terminated
        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            print(f"Total reward: {episode_reward:.4f}")
            observation, info = env.reset()
            episode_reward = 0

    env.close()
    print("Test completed!")


def test_zero_action():
    """Test the environment with zero actions (gravity/damping test)."""
    print("\nTesting BiArt environment with zero actions...")

    # Create environment
    env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type="revolute")

    # Reset environment
    observation, info = env.reset()

    # Run episode with zero actions
    for step in range(200):
        # Zero wrench (no control)
        action = np.zeros(6)

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Render
        env.render()

        if step % 50 == 0:
            print(f"Step {step}: Observation norm = {np.linalg.norm(observation):.4f}")

    env.close()
    print("Zero action test completed!")


def test_constant_force():
    """Test the environment with constant force commands."""
    print("\nTesting BiArt environment with constant forces...")

    # Create environment
    env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type="revolute")

    # Reset environment
    observation, info = env.reset()

    # Run episode with constant forces
    for step in range(300):
        # Constant force to the right for left gripper, left for right gripper
        action = np.array([10.0, 0.0, 0.0, -10.0, 0.0, 0.0])

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Render
        env.render()

        if step % 50 == 0:
            print(f"Step {step}:")
            print(f"  Left gripper pos: {observation[0:2]}")
            print(f"  Right gripper pos: {observation[3:5]}")

    env.close()
    print("Constant force test completed!")


def test_different_joint_types():
    """Test different joint types."""
    print("\nTesting different joint types...")

    for joint_type in ["revolute", "prismatic", "fixed"]:
        print(f"\n--- Testing {joint_type} joint ---")

        env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type=joint_type)
        observation, info = env.reset()

        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

        print(f"  {joint_type} joint test successful!")
        env.close()

    print("\nAll joint type tests completed!")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("BiArt Environment Test Suite")
    print("=" * 60)

    # Test 1: Random actions
    test_random_actions()

    # Test 2: Zero action
    # test_zero_action()

    # Test 3: Constant force
    # test_constant_force()

    # Test 4: Different joint types
    # test_different_joint_types()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
