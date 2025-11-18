"""
Quick smoke test for demo_teleoperation.py
Tests basic initialization without full interaction.
"""

import os
import sys

# Prevent actual window creation
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from gym_biart.envs.biart import BiArtEnv
from gym_biart.envs.linkage_manager import create_two_link_object, JointType
from gym_biart.envs.pd_controller import MultiGripperController, PDGains
from gym_biart.envs.keyboard_planner import MultiEEPlanner
import numpy as np

def test_demo_initialization():
    """Test that demo components can be initialized together."""
    print("Testing demo initialization...")

    try:
        # Create environment
        env = BiArtEnv(
            obs_type='state',
            render_mode='rgb_array',  # Use rgb_array to avoid window
            joint_type='revolute'
        )
        print("  ✓ Environment created")

        # Create linkage
        linkage = create_two_link_object(JointType.REVOLUTE)
        print("  ✓ Linkage created")

        # Create controller
        gains = PDGains(kp_linear=30.0, kd_linear=8.0, kp_angular=15.0, kd_angular=3.0)
        controller = MultiGripperController(num_grippers=2, gains=gains)
        controller.set_timestep(env.dt)
        print("  ✓ Controller created")

        # Create planner
        planner = MultiEEPlanner(
            num_end_effectors=2,
            linkage_object=linkage,
            control_dt=1.0 / env.control_hz
        )
        print("  ✓ Planner created")

        # Reset environment
        obs, info = env.reset()
        print("  ✓ Environment reset")

        # Set linkage bodies
        linkage.set_link_body(0, env.link1)
        linkage.set_link_body(1, env.link2)
        print("  ✓ Linkage bodies set")

        # Initialize planner
        initial_ee_poses = np.array([obs[0:3], obs[3:6]])
        planner.initialize_from_current_state(initial_ee_poses)
        print("  ✓ Planner initialized")

        # Test one step
        current_ee_poses = np.array([
            [env.left_gripper.position.x, env.left_gripper.position.y, env.left_gripper.angle],
            [env.right_gripper.position.x, env.right_gripper.position.y, env.right_gripper.angle],
        ])

        current_velocities = np.array([
            [env.left_gripper.velocity.x, env.left_gripper.velocity.y, env.left_gripper.angular_velocity],
            [env.right_gripper.velocity.x, env.right_gripper.velocity.y, env.right_gripper.angular_velocity],
        ])

        # Update planner
        desired_poses, actions = planner.update([], current_ee_poses)
        print("  ✓ Planner update")

        # Compute wrenches
        wrenches = controller.compute_wrenches(
            current_ee_poses,
            desired_poses,
            current_velocities
        )
        print("  ✓ Wrenches computed")

        # Step environment
        action = np.concatenate([wrenches[0], wrenches[1]])
        obs, reward, terminated, truncated, info = env.step(action)
        print("  ✓ Environment step")

        # Clean up
        env.close()
        print("  ✓ Environment closed")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("DEMO SMOKE TEST")
    print("="*60)

    if test_demo_initialization():
        print("\n✅ SMOKE TEST PASSED")
        print("\nDemo should work correctly!")
        print("\nTo run the interactive demo:")
        print("  python demo_teleoperation.py revolute")
        return 0
    else:
        print("\n❌ SMOKE TEST FAILED")
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
