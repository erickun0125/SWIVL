"""
Test Low-Level Controllers

Tests all low-level controllers with the BiArt environment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from src.ll_controllers.pd_controller import PDController, PDGains
from src.ll_controllers.task_space_impedance import TaskSpaceImpedanceController, ImpedanceGains
from src.ll_controllers.screw_impedance import ScrewImpedanceController, ScrewImpedanceGains


def test_pd_controller():
    """Test PD controller."""
    print("\n=== Testing PD Controller ===")

    controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])

    wrench = controller.compute_wrench(current_pose, desired_pose)

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Computed wrench: {wrench}")
    print("✅ PD Controller test passed")

    return True


def test_impedance_controller():
    """Test task space impedance controller."""
    print("\n=== Testing Task Space Impedance Controller ===")

    controller = TaskSpaceImpedanceController(ImpedanceGains())
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    measured_wrench = np.array([15.0, 10.0, 2.0])

    wrench = controller.compute_wrench(
        current_pose, desired_pose, measured_wrench
    )

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Measured wrench: {measured_wrench}")
    print(f"Computed wrench: {wrench}")
    print("✅ Impedance Controller test passed")

    return True


def test_screw_impedance_controller():
    """Test screw-aware impedance controller."""
    print("\n=== Testing Screw-Aware Impedance Controller ===")

    controller = ScrewImpedanceController(ScrewImpedanceGains())
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    measured_wrench = np.array([15.0, 10.0, 2.0])

    wrench = controller.compute_wrench(
        current_pose, desired_pose, measured_wrench
    )

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Measured wrench: {measured_wrench}")
    print(f"Computed wrench: {wrench}")
    print("✅ Screw Impedance Controller test passed")

    return True


def main():
    print("="*60)
    print("LOW-LEVEL CONTROLLER TESTS")
    print("="*60)

    all_passed = True

    all_passed &= test_pd_controller()
    all_passed &= test_impedance_controller()
    all_passed &= test_screw_impedance_controller()

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
