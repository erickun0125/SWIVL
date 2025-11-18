"""
Visualization Script for BiArt Environment

This script provides various visualization options for testing the BiArt environment.
"""

import sys
sys.path.insert(0, '.')

import gymnasium as gym
import numpy as np
import gym_biart
import argparse
import time
import os
from datetime import datetime


def run_with_visualization(joint_type="revolute", steps=1000, save_video=False, slow_motion=False):
    """Run environment with human rendering."""
    print(f"\n{'='*60}")
    print(f"Running BiArt Environment - Joint Type: {joint_type}")
    print(f"{'='*60}\n")

    env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type=joint_type)

    obs, info = env.reset(seed=42)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"\nPress Ctrl+C to stop\n")

    frames = []
    episode = 0
    step_count = 0
    total_reward = 0

    try:
        for step in range(steps):
            # Sample random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Render
            env.render()

            # Slow motion mode
            if slow_motion:
                time.sleep(0.05)

            # Print info every 50 steps
            if step % 50 == 0:
                print(f"[Episode {episode}] Step {step_count}:")
                print(f"  Reward: {reward:.4f} | Total: {total_reward:.4f}")
                print(f"  Tracking: {info['tracking_reward']:.4f}")
                print(f"  Safety: {info['safety_penalty']:.4f}")
                print(f"  Position Error: {info['pos_error']:.2f}")
                print(f"  Success: {info['is_success']}")

            # Reset if terminated
            if terminated or truncated:
                print(f"\n{'='*40}")
                print(f"Episode {episode} completed!")
                print(f"  Steps: {step_count}")
                print(f"  Total Reward: {total_reward:.4f}")
                print(f"  Success: {info['is_success']}")
                print(f"{'='*40}\n")

                obs, info = env.reset()
                episode += 1
                step_count = 0
                total_reward = 0

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        env.close()
        print(f"\nVisualization completed. Total steps: {step}")


def run_without_visualization(joint_type="revolute", steps=500, save_images=False):
    """Run environment without rendering and optionally save images."""
    print(f"\n{'='*60}")
    print(f"Running BiArt (No Display) - Joint Type: {joint_type}")
    print(f"{'='*60}\n")

    env = gym.make("gym_biart/BiArt-v0", render_mode="rgb_array", joint_type=joint_type)

    obs, info = env.reset(seed=42)

    # Create output directory for images
    if save_images:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/biart_{joint_type}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving images to: {output_dir}/")

    episode_rewards = []
    total_reward = 0

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Save image periodically
        if save_images and step % 20 == 0:
            img = env.render()
            import cv2
            cv2.imwrite(f"{output_dir}/step_{step:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.4f}, Total={total_reward:.4f}")

        if terminated or truncated:
            episode_rewards.append(total_reward)
            total_reward = 0
            obs, info = env.reset()

    env.close()

    if episode_rewards:
        print(f"\n{'='*40}")
        print(f"Statistics:")
        print(f"  Episodes: {len(episode_rewards)}")
        print(f"  Avg Reward: {np.mean(episode_rewards):.4f}")
        print(f"  Max Reward: {np.max(episode_rewards):.4f}")
        print(f"  Min Reward: {np.min(episode_rewards):.4f}")
        print(f"{'='*40}\n")

    if save_images:
        print(f"Images saved to: {output_dir}/")


def run_controlled_test(joint_type="revolute"):
    """Run environment with specific control patterns."""
    print(f"\n{'='*60}")
    print(f"Running Controlled Test - Joint Type: {joint_type}")
    print(f"{'='*60}\n")

    env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type=joint_type)
    obs, info = env.reset(seed=42)

    # Test different control patterns
    patterns = [
        ("Push Right", np.array([20.0, 0.0, 0.0, -20.0, 0.0, 0.0])),
        ("Push Up", np.array([0.0, 20.0, 0.0, 0.0, 20.0, 0.0])),
        ("Rotate CW", np.array([0.0, 0.0, 10.0, 0.0, 0.0, -10.0])),
        ("Pull In", np.array([-10.0, 0.0, 0.0, 10.0, 0.0, 0.0])),
        ("Zero Force", np.zeros(6)),
    ]

    steps_per_pattern = 50

    for pattern_name, action in patterns:
        print(f"\nTesting: {pattern_name}")
        print(f"  Action: {action}")

        for step in range(steps_per_pattern):
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)

            if terminated or truncated:
                obs, info = env.reset()
                break

    env.close()
    print("\nControlled test completed!")


def compare_joint_types():
    """Compare different joint types side by side."""
    print(f"\n{'='*60}")
    print(f"Comparing Joint Types")
    print(f"{'='*60}\n")

    joint_types = ["revolute", "prismatic", "fixed"]

    for joint_type in joint_types:
        print(f"\n--- Testing {joint_type.upper()} joint ---")
        env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type=joint_type)
        obs, info = env.reset(seed=42)

        print(f"Initial joint angle/position: {obs[8]:.2f}")

        # Apply same force pattern
        action = np.array([10.0, 10.0, 0.0, -10.0, 10.0, 0.0])

        for step in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.01)

            if terminated or truncated:
                break

        print(f"Final joint angle/position: {obs[8]:.2f}")
        print(f"Total displacement: {abs(obs[8] - obs[8]):.2f}")

        env.close()
        time.sleep(1)  # Pause between tests

    print("\nComparison completed!")


def interactive_mode():
    """Interactive mode with keyboard control (simplified)."""
    print(f"\n{'='*60}")
    print(f"Interactive Mode")
    print(f"{'='*60}\n")
    print("This is a simplified version. Full keyboard control requires pygame events.")
    print("Running with mouse control instructions...\n")

    env = gym.make("gym_biart/BiArt-v0", render_mode="human", joint_type="revolute")
    obs, info = env.reset()

    print("Use random actions for now.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            action = env.action_space.sample() * 0.5  # Smaller actions
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.05)

            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass

    env.close()


def main():
    parser = argparse.ArgumentParser(description="BiArt Environment Visualization")
    parser.add_argument("--mode", type=str, default="visual",
                        choices=["visual", "headless", "controlled", "compare", "interactive"],
                        help="Visualization mode")
    parser.add_argument("--joint", type=str, default="revolute",
                        choices=["revolute", "prismatic", "fixed"],
                        help="Joint type")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps to run")
    parser.add_argument("--save-images", action="store_true",
                        help="Save images (headless mode only)")
    parser.add_argument("--slow", action="store_true",
                        help="Slow motion mode")

    args = parser.parse_args()

    if args.mode == "visual":
        run_with_visualization(args.joint, args.steps, slow_motion=args.slow)
    elif args.mode == "headless":
        run_without_visualization(args.joint, args.steps, save_images=args.save_images)
    elif args.mode == "controlled":
        run_controlled_test(args.joint)
    elif args.mode == "compare":
        compare_joint_types()
    elif args.mode == "interactive":
        interactive_mode()


if __name__ == "__main__":
    main()
