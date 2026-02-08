"""
Test Updated Low-Level Controllers with Desired Velocity

Tests all low-level controllers with the new interface that accepts
desired_pose, desired_vel, and desired_accel.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from src.ll_controllers.pd_controller import PDController, PDGains


def test_pd_controller_with_desired_velocity():
    """Test PD controller with desired velocity."""
    print("\n=== Testing PD Controller with Desired Velocity ===")

    controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    desired_velocity = np.array([5.0, 2.5, 0.1])  # Moving toward desired pose
    desired_acceleration = np.zeros(3)  # Zero acceleration

    wrench = controller.compute_wrench(
        current_pose,
        desired_pose,
        desired_velocity,
        desired_acceleration
    )

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Desired velocity: {desired_velocity}")
    print(f"Computed wrench: {wrench}")

    # Check that wrench has reasonable magnitude
    force_mag = np.linalg.norm(wrench[:2])
    if force_mag > 0 and force_mag < 1000:
        print("✅ PD Controller with desired velocity test passed")
        return True
    else:
        print(f"❌ PD Controller produced unreasonable wrench magnitude: {force_mag}")
        return False


def test_pd_controller_with_current_velocity():
    """Test PD controller with both desired and current velocity."""
    print("\n=== Testing PD Controller with Current Velocity ===")

    controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    current_velocity = np.array([1.0, 0.5, 0.05])  # Currently moving
    desired_pose = np.array([10.0, 5.0, 0.5])
    desired_velocity = np.array([5.0, 2.5, 0.1])
    desired_acceleration = np.zeros(3)

    wrench = controller.compute_wrench(
        current_pose,
        desired_pose,
        desired_velocity,
        desired_acceleration,
        current_velocity
    )

    print(f"Current pose: {current_pose}")
    print(f"Current velocity: {current_velocity}")
    print(f"Desired pose: {desired_pose}")
    print(f"Desired velocity: {desired_velocity}")
    print(f"Computed wrench: {wrench}")

    force_mag = np.linalg.norm(wrench[:2])
    if force_mag > 0 and force_mag < 1000:
        print("✅ PD Controller with current velocity test passed")
        return True
    else:
        print(f"❌ PD Controller produced unreasonable wrench magnitude: {force_mag}")
        return False


def test_trajectory_following():
    """Test controller tracking a simple trajectory."""
    print("\n=== Testing Trajectory Following ===")

    controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
    controller.set_timestep(0.01)

    # Simulate a straight line trajectory
    dt = 0.01
    num_steps = 10

    current_pose = np.array([0.0, 0.0, 0.0])
    target_pose = np.array([10.0, 5.0, 0.5])

    max_wrench_mag = 0.0

    for step in range(num_steps):
        # Desired trajectory (linear interpolation)
        alpha = step / num_steps
        desired_pose = (1 - alpha) * current_pose + alpha * target_pose

        # Desired velocity (constant)
        desired_velocity = (target_pose - current_pose) / (num_steps * dt)

        # Compute wrench
        wrench = controller.compute_wrench(
            current_pose,
            desired_pose,
            desired_velocity,
            np.zeros(3)  # Zero acceleration
        )

        force_mag = np.linalg.norm(wrench[:2])
        max_wrench_mag = max(max_wrench_mag, force_mag)

        # Simple integration (for testing)
        # In real system, physics engine would handle this
        current_pose += desired_velocity * dt

    print(f"Completed {num_steps} steps")
    print(f"Final pose: {current_pose}")
    print(f"Max wrench magnitude: {max_wrench_mag:.2f}")

    if max_wrench_mag < 1000:
        print("✅ Trajectory following test passed")
        return True
    else:
        print("❌ Trajectory following produced excessive wrenches")
        return False


def main():
    print("="*60)
    print("UPDATED CONTROLLER TESTS")
    print("="*60)

    all_passed = True

    all_passed &= test_pd_controller_with_desired_velocity()
    all_passed &= test_pd_controller_with_current_velocity()
    all_passed &= test_trajectory_following()

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
