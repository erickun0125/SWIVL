"""
Test Refactored BiArt Environment

Tests the refactored BiArt environment that uses the manager pattern:
- EndEffectorManager for grippers
- ObjectManager for articulated objects
- RewardManager for rewards
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.envs.biart import BiArtEnv


def test_environment_creation():
    """Test that environment can be created."""
    print("\n=== Testing Environment Creation ===")

    try:
        env = BiArtEnv(
            obs_type="state",
            render_mode="rgb_array",
            joint_type="revolute"
        )
        print("✅ Environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_reset():
    """Test that environment can be reset."""
    print("\n=== Testing Environment Reset ===")

    try:
        env = BiArtEnv(
            obs_type="state",
            render_mode="rgb_array",
            joint_type="revolute"
        )

        obs, info = env.reset(seed=42)

        print(f"Observation shape: {obs.shape}")
        print(f"Expected shape: (18,)")

        if obs.shape == (18,):
            print("✅ Environment reset successfully")
            print(f"   Left gripper: {obs[0:3]}")
            print(f"   Right gripper: {obs[3:6]}")
            print(f"   Link1: {obs[6:9]}")
            print(f"   Link2: {obs[9:12]}")
            print(f"   Ext wrench left: {obs[12:15]}")
            print(f"   Ext wrench right: {obs[15:18]}")
            return True
        else:
            print(f"❌ Observation shape mismatch: {obs.shape}")
            return False

    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_step():
    """Test that environment can step."""
    print("\n=== Testing Environment Step ===")

    try:
        env = BiArtEnv(
            obs_type="state",
            render_mode="rgb_array",
            joint_type="revolute"
        )

        obs, info = env.reset(seed=42)

        # Zero action
        action = np.zeros(6)

        obs_next, reward, terminated, truncated, info = env.step(action)

        print(f"Next observation shape: {obs_next.shape}")
        print(f"Reward: {reward:.4f}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info keys: {list(info.keys())}")

        if obs_next.shape == (18,):
            print("✅ Environment step successful")
            return True
        else:
            print(f"❌ Observation shape mismatch: {obs_next.shape}")
            return False

    except Exception as e:
        print(f"❌ Failed to step environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_steps():
    """Test multiple steps with random actions."""
    print("\n=== Testing Multiple Steps ===")

    try:
        env = BiArtEnv(
            obs_type="state",
            render_mode="rgb_array",
            joint_type="revolute"
        )

        obs, info = env.reset(seed=42)

        total_reward = 0.0
        num_steps = 10

        for step in range(num_steps):
            # Small random actions
            action = np.random.randn(6) * 5.0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break

        print(f"Completed {num_steps} steps")
        print(f"Total reward: {total_reward:.4f}")
        print("✅ Multiple steps successful")
        return True

    except Exception as e:
        print(f"❌ Failed during multiple steps: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_joint_types():
    """Test different joint types."""
    print("\n=== Testing Different Joint Types ===")

    all_passed = True

    for joint_type in ["revolute", "prismatic", "fixed"]:
        try:
            print(f"\n  Testing {joint_type} joint...")

            env = BiArtEnv(
                obs_type="state",
                render_mode="rgb_array",
                joint_type=joint_type
            )

            obs, info = env.reset(seed=42)
            action = np.zeros(6)
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"  ✅ {joint_type} joint works")

        except Exception as e:
            print(f"  ❌ {joint_type} joint failed: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All joint types work")

    return all_passed


def test_managers():
    """Test that managers are properly initialized."""
    print("\n=== Testing Managers ===")

    try:
        env = BiArtEnv(
            obs_type="state",
            render_mode="rgb_array",
            joint_type="revolute"
        )

        obs, info = env.reset(seed=42)

        # Check that managers exist
        assert env.ee_manager is not None, "EndEffectorManager not initialized"
        assert env.object_manager is not None, "ObjectManager not initialized"
        assert env.reward_manager is not None, "RewardManager not initialized"

        # Test manager methods
        ee_poses = env.ee_manager.get_poses()
        print(f"EE poses shape: {ee_poses.shape}")
        assert ee_poses.shape == (2, 3), f"Wrong EE poses shape: {ee_poses.shape}"

        link_poses = env.object_manager.get_link_poses()
        print(f"Link poses shape: {link_poses.shape}")
        assert link_poses.shape == (2, 3), f"Wrong link poses shape: {link_poses.shape}"

        external_wrenches = env.ee_manager.get_external_wrenches()
        print(f"External wrenches shape: {external_wrenches.shape}")
        assert external_wrenches.shape == (2, 3), f"Wrong external wrenches shape: {external_wrenches.shape}"

        grasping_poses = env.object_manager.get_grasping_poses()
        print(f"Grasping poses keys: {list(grasping_poses.keys())}")
        assert "left" in grasping_poses, "Left grasping pose missing"
        assert "right" in grasping_poses, "Right grasping pose missing"

        print("✅ All managers working correctly")
        return True

    except Exception as e:
        print(f"❌ Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("REFACTORED BIART ENVIRONMENT TESTS")
    print("="*60)

    all_passed = True

    all_passed &= test_environment_creation()
    all_passed &= test_environment_reset()
    all_passed &= test_environment_step()
    all_passed &= test_multiple_steps()
    all_passed &= test_different_joint_types()
    all_passed &= test_managers()

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
