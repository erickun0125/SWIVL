"""
Quick verification script for demo_teleoperation.py

This script verifies that all components can be imported and initialized
without running the full interactive demo.
"""

import sys
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import gymnasium as gym
        print("  ✓ gymnasium")
    except ImportError as e:
        print(f"  ✗ gymnasium: {e}")
        return False

    try:
        import pygame
        print("  ✓ pygame")
    except ImportError as e:
        print(f"  ✗ pygame: {e}")
        return False

    try:
        from gym_biart.envs.linkage_manager import LinkageObject, JointType, create_two_link_object
        print("  ✓ linkage_manager")
    except ImportError as e:
        print(f"  ✗ linkage_manager: {e}")
        return False

    try:
        from gym_biart.envs.pd_controller import MultiGripperController, PDGains
        print("  ✓ pd_controller")
    except ImportError as e:
        print(f"  ✗ pd_controller: {e}")
        return False

    try:
        from gym_biart.envs.keyboard_planner import MultiEEPlanner
        print("  ✓ keyboard_planner")
    except ImportError as e:
        print(f"  ✗ keyboard_planner: {e}")
        return False

    return True


def test_component_initialization():
    """Test that all components can be initialized."""
    print("\nTesting component initialization...")

    from gym_biart.envs.linkage_manager import create_two_link_object, JointType
    from gym_biart.envs.pd_controller import MultiGripperController, PDGains
    from gym_biart.envs.keyboard_planner import MultiEEPlanner

    # Test linkage creation
    try:
        linkage = create_two_link_object(JointType.REVOLUTE)
        assert linkage.num_links == 2
        assert len(linkage.joints) == 1
        print("  ✓ Linkage object created")
    except Exception as e:
        print(f"  ✗ Linkage creation failed: {e}")
        return False

    # Test controller creation
    try:
        gains = PDGains(kp_linear=30.0, kd_linear=8.0, kp_angular=15.0, kd_angular=3.0)
        controller = MultiGripperController(num_grippers=2, gains=gains)
        print("  ✓ PD controller created")
    except Exception as e:
        print(f"  ✗ Controller creation failed: {e}")
        return False

    # Test planner creation
    try:
        planner = MultiEEPlanner(
            num_end_effectors=2,
            linkage_object=linkage,
            control_dt=0.1
        )
        print("  ✓ Keyboard planner created")
    except Exception as e:
        print(f"  ✗ Planner creation failed: {e}")
        return False

    return True


def test_demo_class():
    """Test that TeleoperationDemo class can be imported and inspected."""
    print("\nTesting demo class...")

    try:
        # Import without pygame init
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Prevent actual window creation

        from demo_teleoperation import TeleoperationDemo

        # Check class exists and has required methods
        required_methods = ['reset_environment', 'get_current_state', 'draw_info_overlay', 'run']
        for method in required_methods:
            if not hasattr(TeleoperationDemo, method):
                print(f"  ✗ Missing method: {method}")
                return False

        print("  ✓ TeleoperationDemo class structure verified")
        return True

    except Exception as e:
        print(f"  ✗ Demo class test failed: {e}")
        return False


def test_basic_workflow():
    """Test basic workflow without rendering."""
    print("\nTesting basic workflow...")

    from gym_biart.envs.linkage_manager import create_two_link_object, JointType
    from gym_biart.envs.pd_controller import MultiGripperController, PDGains

    try:
        # Create components
        linkage = create_two_link_object(JointType.REVOLUTE)
        gains = PDGains()
        controller = MultiGripperController(num_grippers=2, gains=gains)
        controller.set_timestep(0.01)

        # Simulate state
        current_poses = np.array([
            [150.0, 350.0, np.pi/2],
            [350.0, 350.0, np.pi/2]
        ])

        desired_poses = np.array([
            [160.0, 350.0, np.pi/2 + 0.1],
            [350.0, 350.0, np.pi/2]
        ])

        # Compute wrenches
        wrenches = controller.compute_wrenches(current_poses, desired_poses)

        # Verify output
        assert wrenches.shape == (2, 3), f"Expected shape (2, 3), got {wrenches.shape}"
        assert not np.any(np.isnan(wrenches)), "Wrenches contain NaN"
        assert not np.any(np.isinf(wrenches)), "Wrenches contain Inf"

        print("  ✓ Basic workflow successful")
        print(f"    Computed wrenches: shape={wrenches.shape}")
        print(f"    EE0 wrench: fx={wrenches[0,0]:.1f}, fy={wrenches[0,1]:.1f}, tau={wrenches[0,2]:.1f}")
        print(f"    EE1 wrench: fx={wrenches[1,0]:.1f}, fy={wrenches[1,1]:.1f}, tau={wrenches[1,2]:.1f}")

        return True

    except Exception as e:
        print(f"  ✗ Basic workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("DEMO VERIFICATION SCRIPT")
    print("="*60)

    all_passed = True

    # Run tests
    if not test_imports():
        print("\n❌ Import test failed")
        all_passed = False

    if not test_component_initialization():
        print("\n❌ Component initialization test failed")
        all_passed = False

    if not test_demo_class():
        print("\n❌ Demo class test failed")
        all_passed = False

    if not test_basic_workflow():
        print("\n❌ Basic workflow test failed")
        all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("="*60)
        print("\nYou can now run the interactive demo:")
        print("  python demo_teleoperation.py revolute")
        print("  python demo_teleoperation.py prismatic")
        print("  python demo_teleoperation.py fixed")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the errors above before running the demo.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
