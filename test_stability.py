"""
Test stability and bouncing issues.
Runs static hold demo for several steps and checks for excessive motion.
"""

import os
import sys
import numpy as np

# Prevent window creation
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from gym_biart.envs.biart import BiArtEnv
from gym_biart.envs.linkage_manager import create_two_link_object, JointType
from gym_biart.envs.pd_controller import MultiGripperController, PDGains


def test_static_hold(joint_type='revolute', num_steps=100):
    """Test that grippers hold position without excessive movement."""
    print(f"\n{'='*60}")
    print(f"Testing static hold for {joint_type} joint")
    print('='*60)

    # Create environment
    env = BiArtEnv(
        obs_type='state',
        render_mode='rgb_array',
        joint_type=joint_type
    )

    # Map joint type
    joint_type_map = {
        'revolute': JointType.REVOLUTE,
        'prismatic': JointType.PRISMATIC,
        'fixed': JointType.FIXED
    }
    joint_enum = joint_type_map[joint_type]

    # Create linkage
    linkage = create_two_link_object(joint_enum)

    # Create controller with soft gains
    gains = PDGains(
        kp_linear=15.0,
        kd_linear=5.0,
        kp_angular=8.0,
        kd_angular=2.0
    )
    controller = MultiGripperController(num_grippers=2, gains=gains)
    controller.set_timestep(env.dt)

    # Reset environment
    obs, info = env.reset()

    # Set linkage bodies
    linkage.set_link_body(0, env.link1)
    linkage.set_link_body(1, env.link2)

    # Register pymunk joint
    if len(linkage.joints) > 0 and env.joint is not None:
        linkage.joints[0].pymunk_joint = env.joint

    # Initial poses (hold these)
    initial_ee_poses = np.array([
        obs[0:3],   # Left gripper
        obs[3:6],   # Right gripper
    ])
    desired_poses = initial_ee_poses.copy()

    print(f"\nInitial poses:")
    print(f"  Left EE:  x={initial_ee_poses[0,0]:.1f}, y={initial_ee_poses[0,1]:.1f}, θ={np.degrees(initial_ee_poses[0,2]):.1f}°")
    print(f"  Right EE: x={initial_ee_poses[1,0]:.1f}, y={initial_ee_poses[1,1]:.1f}, θ={np.degrees(initial_ee_poses[1,2]):.1f}°")

    # Track maximum deviations
    max_pos_error = 0.0
    max_vel = 0.0
    errors = []
    velocities = []

    # Run simulation
    for step in range(num_steps):
        # Get current state
        current_ee_poses = np.array([
            [env.left_gripper.position.x,
             env.left_gripper.position.y,
             env.left_gripper.angle],
            [env.right_gripper.position.x,
             env.right_gripper.position.y,
             env.right_gripper.angle],
        ])

        current_velocities = np.array([
            [env.left_gripper.velocity.x,
             env.left_gripper.velocity.y,
             env.left_gripper.angular_velocity],
            [env.right_gripper.velocity.x,
             env.right_gripper.velocity.y,
             env.right_gripper.angular_velocity],
        ])

        # Compute control wrenches
        wrenches = controller.compute_wrenches(
            current_ee_poses,
            desired_poses,
            current_velocities
        )

        # Apply action
        action = np.concatenate([wrenches[0], wrenches[1]])
        obs, reward, terminated, truncated, info = env.step(action)

        # Track errors
        pos_error = np.linalg.norm(current_ee_poses[:, :2] - desired_poses[:, :2], axis=1)
        vel_mag = np.linalg.norm(current_velocities[:, :2], axis=1)

        max_pos_error = max(max_pos_error, np.max(pos_error))
        max_vel = max(max_vel, np.max(vel_mag))

        errors.append(np.mean(pos_error))
        velocities.append(np.mean(vel_mag))

        # Print periodic status
        if step % 20 == 0:
            print(f"[Step {step:3d}] Avg pos error: {np.mean(pos_error):5.2f} | "
                  f"Avg velocity: {np.mean(vel_mag):5.2f} | "
                  f"Max pos error: {max_pos_error:5.2f}")

    # Final analysis
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"Max position error: {max_pos_error:.2f} pixels")
    print(f"Max velocity: {max_vel:.2f} pixels/s")
    print(f"Final avg position error: {np.mean(errors[-10:]):.2f} pixels")
    print(f"Final avg velocity: {np.mean(velocities[-10:]):.2f} pixels/s")

    # Check for bouncing/flying
    POSITION_THRESHOLD = 10.0  # pixels
    VELOCITY_THRESHOLD = 20.0  # pixels/s

    stable = True
    issues = []

    if max_pos_error > POSITION_THRESHOLD:
        stable = False
        issues.append(f"Position error too large: {max_pos_error:.1f} > {POSITION_THRESHOLD}")

    if max_vel > VELOCITY_THRESHOLD:
        stable = False
        issues.append(f"Velocity too high: {max_vel:.1f} > {VELOCITY_THRESHOLD}")

    if np.mean(velocities[-10:]) > 5.0:
        stable = False
        issues.append(f"Failed to settle: final velocity {np.mean(velocities[-10:]):.1f} > 5.0")

    # Check V-shape for revolute
    if joint_type == 'revolute':
        linkage.update_joint_states()
        joint_config = linkage.get_configuration()
        if len(joint_config) > 0:
            joint_angle_deg = np.degrees(joint_config[0])
            print(f"Joint angle: {joint_angle_deg:.1f}°")

            if not (-120 <= joint_angle_deg <= -60):
                stable = False
                issues.append(f"Joint angle not in V-shape range: {joint_angle_deg:.1f}° not in [-120°, -60°]")

    env.close()

    if stable:
        print("\n✅ STABILITY TEST PASSED")
        return True
    else:
        print("\n❌ STABILITY TEST FAILED")
        print("\nIssues detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False


def main():
    print("\n" + "="*60)
    print("STABILITY TEST SUITE")
    print("="*60)

    all_passed = True

    for joint_type in ['revolute', 'prismatic', 'fixed']:
        passed = test_static_hold(joint_type)
        all_passed = all_passed and passed
        print()

    print("="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nSystem is stable!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the issues above.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
