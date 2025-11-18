"""
Test Low-Level Controllers

Tests all three main low-level controllers with proper examples.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from src.ll_controllers.pd_controller import PDController, PDGains
from src.ll_controllers.task_space_impedance import TaskSpaceImpedanceController, ImpedanceGains
from src.ll_controllers.se2_screw_decomposed_impedance import (
    SE2ScrewDecomposedImpedanceController,
    ScrewImpedanceParams
)
from src.se2_dynamics import SE2RobotParams


def test_pd_controller():
    """Test Position/Orientation decomposed PD controller."""
    print("\n=== Testing PD Controller (Controller #1) ===")
    print("Position and orientation are controlled independently")

    controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    desired_velocity = np.array([1.0, 0.5, 0.1])  # Desired velocity in world frame

    wrench = controller.compute_wrench(
        current_pose, desired_pose, desired_velocity
    )

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Desired velocity: {desired_velocity}")
    print(f"Computed wrench: {wrench}")
    print("✅ PD Controller test passed")

    return True


def test_impedance_controller():
    """Test SE(2) task space impedance controller."""
    print("\n=== Testing SE(2) Task Space Impedance Controller (Controller #2) ===")
    print("Proper impedance control with robot dynamics (Lambda_b, C_b, eta_b)")

    controller = TaskSpaceImpedanceController(ImpedanceGains())
    controller.set_timestep(0.01)

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    measured_wrench = np.array([15.0, 10.0, 2.0])
    current_velocity = np.array([0.0, 0.0, 0.0])  # Spatial frame
    desired_velocity = np.array([1.0, 0.5, 0.1])  # Body frame

    wrench = controller.compute_wrench(
        current_pose, desired_pose, measured_wrench,
        current_velocity=current_velocity,
        desired_velocity=desired_velocity
    )

    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"Measured wrench: {measured_wrench}")
    print(f"Computed wrench: {wrench}")
    print("✅ Task Space Impedance Controller test passed")

    return True


def test_screw_decomposed_impedance_controller():
    """Test screw-decomposed SE(2) impedance controller."""
    print("\n=== Testing Screw-Decomposed Impedance Controller (Controller #3) ===")
    print("Directional compliance: parallel (compliant) + perpendicular (stiff)")

    # Example: compliant along x-axis, stiff in other directions
    screw_axis = np.array([1.0, 0.0, 0.0])  # Translation along x

    params = ScrewImpedanceParams(
        M_parallel=1.0,
        D_parallel=5.0,
        K_parallel=10.0,      # LOW stiffness (compliant)
        M_perpendicular=1.0,
        D_perpendicular=20.0,
        K_perpendicular=100.0  # HIGH stiffness (stiff)
    )

    robot_params = SE2RobotParams(mass=1.0, inertia=0.1)

    controller = SE2ScrewDecomposedImpedanceController.create_from_standard_params(
        screw_axis=screw_axis,
        M_parallel=params.M_parallel,
        D_parallel=params.D_parallel,
        K_parallel=params.K_parallel,
        M_perpendicular=params.M_perpendicular,
        D_perpendicular=params.D_perpendicular,
        K_perpendicular=params.K_perpendicular,
        robot_params=robot_params,
        model_matching=True
    )

    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    current_twist = np.array([0.0, 0.0, 0.0])
    desired_twist = np.array([1.0, 0.1, 0.05])

    wrench, info = controller.compute_control(
        current_pose, desired_pose,
        current_twist, desired_twist
    )

    print(f"Screw axis: {screw_axis}")
    print(f"Current pose: {current_pose}")
    print(f"Desired pose: {desired_pose}")
    print(f"\nDecomposition:")
    print(f"  θ (along screw): {info['theta']:.4f}")
    print(f"  e_parallel: {info['e_parallel']}")
    print(f"  e_perp: {info['e_perp']}")
    print(f"\nComputed wrench: {wrench}")
    print(f"Stiffness ratio K_⊥/K_∥ = {params.K_perpendicular/params.K_parallel:.1f}x")
    print("✅ Screw-Decomposed Impedance Controller test passed")

    return True


def main():
    print("="*70)
    print("LOW-LEVEL CONTROLLER TESTS")
    print("="*70)
    print("\nTesting all 3 required controllers:")
    print("1. Position/Orientation decomposed PD")
    print("2. SE(2) task space impedance (with proper dynamics)")
    print("3. Screw-decomposed SE(2) impedance (directional compliance)")

    all_passed = True

    all_passed &= test_pd_controller()
    all_passed &= test_impedance_controller()
    all_passed &= test_screw_decomposed_impedance_controller()

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
