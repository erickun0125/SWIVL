"""
Test gripper positioning to verify they don't interfere with joint.
"""

import os
import sys
import numpy as np

# Prevent window creation
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from gym_biart.envs.biart import BiArtEnv


def test_gripper_positions():
    """Test that grippers are positioned away from joint."""
    print("\n" + "="*60)
    print("GRIPPER POSITIONING TEST")
    print("="*60)

    for joint_type in ['revolute', 'prismatic', 'fixed']:
        print(f"\n--- Testing {joint_type.upper()} joint ---")

        env = BiArtEnv(
            obs_type='state',
            render_mode='rgb_array',
            joint_type=joint_type
        )

        obs, info = env.reset()

        # Get gripper and link positions
        left_gripper_pos = env.left_gripper.position
        right_gripper_pos = env.right_gripper.position
        link1_pos = env.link1.position
        link2_pos = env.link2.position

        # Calculate joint position (link1's right end = link2's left end)
        joint_pos_from_link1 = env.link1.local_to_world((env.link_length / 2, 0))
        joint_pos_from_link2 = env.link2.local_to_world((-env.link_length / 2, 0))

        print(f"\nLink1 center: ({link1_pos.x:.1f}, {link1_pos.y:.1f})")
        print(f"Link2 center: ({link2_pos.x:.1f}, {link2_pos.y:.1f})")
        print(f"Joint position: ({joint_pos_from_link1.x:.1f}, {joint_pos_from_link1.y:.1f})")
        print(f"Joint check (from link2): ({joint_pos_from_link2.x:.1f}, {joint_pos_from_link2.y:.1f})")

        print(f"\nLeft gripper (blue): ({left_gripper_pos.x:.1f}, {left_gripper_pos.y:.1f})")
        print(f"Right gripper (red): ({right_gripper_pos.x:.1f}, {right_gripper_pos.y:.1f})")

        # Calculate distances
        dist_left_to_joint = np.linalg.norm([
            left_gripper_pos.x - joint_pos_from_link1.x,
            left_gripper_pos.y - joint_pos_from_link1.y
        ])
        dist_right_to_joint = np.linalg.norm([
            right_gripper_pos.x - joint_pos_from_link1.x,
            right_gripper_pos.y - joint_pos_from_link1.y
        ])
        dist_left_to_link1 = np.linalg.norm([
            left_gripper_pos.x - link1_pos.x,
            left_gripper_pos.y - link1_pos.y
        ])
        dist_right_to_link2 = np.linalg.norm([
            right_gripper_pos.x - link2_pos.x,
            right_gripper_pos.y - link2_pos.y
        ])

        print(f"\nDistances:")
        print(f"  Left gripper -> Joint: {dist_left_to_joint:.1f} pixels")
        print(f"  Right gripper -> Joint: {dist_right_to_joint:.1f} pixels")
        print(f"  Left gripper -> Link1 center: {dist_left_to_link1:.1f} pixels")
        print(f"  Right gripper -> Link2 center: {dist_right_to_link2:.1f} pixels")

        # Check if grippers are offset from link centers
        OFFSET_THRESHOLD = 5.0  # Should be offset by at least 5 pixels
        success = True

        if dist_left_to_link1 < OFFSET_THRESHOLD:
            print(f"  ⚠️  Left gripper too close to link1 center (should be offset)")
            success = False
        else:
            print(f"  ✓ Left gripper properly offset from link1 center")

        if dist_right_to_link2 < OFFSET_THRESHOLD:
            print(f"  ⚠️  Right gripper too close to link2 center (should be offset)")
            success = False
        else:
            print(f"  ✓ Right gripper properly offset from link2 center")

        # Check if grippers maintain reasonable distance from joint
        MIN_JOINT_DISTANCE = 15.0  # At least 15 pixels from joint

        if dist_left_to_joint < MIN_JOINT_DISTANCE:
            print(f"  ⚠️  Left gripper too close to joint (< {MIN_JOINT_DISTANCE})")
            success = False
        else:
            print(f"  ✓ Left gripper maintains distance from joint")

        if dist_right_to_joint < MIN_JOINT_DISTANCE:
            print(f"  ⚠️  Right gripper too close to joint (< {MIN_JOINT_DISTANCE})")
            success = False
        else:
            print(f"  ✓ Right gripper maintains distance from joint")

        env.close()

        if success:
            print(f"\n✅ {joint_type.upper()}: PASS")
        else:
            print(f"\n❌ {joint_type.upper()}: FAIL")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    test_gripper_positions()
