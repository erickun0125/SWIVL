"""Simple test script for BiArt environment without visualization."""

import sys
sys.path.insert(0, '.')

import gymnasium as gym
import numpy as np
import gym_biart

def test_environment():
    """Test basic environment functionality."""
    print("Creating BiArt environment...")

    # Create environment without human rendering
    env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type="revolute")

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Reset environment
    print("\nResetting environment...")
    observation, info = env.reset(seed=42)

    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation:\n{observation}")

    # Run a few steps
    print("\nRunning 10 steps with random actions...")
    for step in range(10):
        # Random action
        action = env.action_space.sample()

        # Step
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Tracking reward: {info['tracking_reward']:.6f}")
        print(f"  Safety penalty: {info['safety_penalty']:.6f}")
        print(f"  Is success: {info['is_success']}")
        print(f"  Position error: {info['pos_error']:.2f}")

        if terminated or truncated:
            print("  Episode terminated!")
            break

    # Test rendering
    print("\nTesting rendering...")
    img = env.render()
    print(f"Rendered image shape: {img.shape}")

    env.close()
    print("\nTest completed successfully!")


def test_different_joints():
    """Test different joint types."""
    print("\n" + "=" * 60)
    print("Testing different joint types...")
    print("=" * 60)

    for joint_type in ["revolute", "prismatic", "fixed"]:
        print(f"\nTesting {joint_type} joint...")

        env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type=joint_type)
        observation, info = env.reset(seed=42)

        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

        print(f"  {joint_type} joint test successful!")
        env.close()


if __name__ == "__main__":
    test_environment()
    test_different_joints()
