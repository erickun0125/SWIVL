"""
Core functionality tests for SWIVL.

Tests the fundamental components:
1. SE(2) Math operations
2. SE(2) Dynamics
3. Controllers
4. Environment
5. RL Environment

Run with: python scripts/tests/test_core.py
"""

import numpy as np
import sys


def test_se2_math():
    """Test SE(2) Lie group operations."""
    print("\n=== Testing SE(2) Math ===")
    
    from src.se2_math import (
        se2_exp, se2_log, se2_compose, se2_inverse, se2_adjoint,
        SE2Pose, normalize_angle, body_to_spatial_twist, spatial_to_body_twist
    )
    
    # Test exp/log round-trip
    xi = np.array([0.3, 1.0, 0.5])  # [omega, vx, vy] MR convention
    T = se2_exp(xi)
    xi_back = se2_log(T)
    assert np.allclose(xi, xi_back, atol=1e-6), f"Exp/Log mismatch: {xi} vs {xi_back}"
    print("✓ Exp/Log round-trip")
    
    # Test composition
    T1 = se2_exp(np.array([0.1, 1.0, 0.0]))
    T2 = se2_exp(np.array([0.2, 0.0, 1.0]))
    T12 = se2_compose(T1, T2)
    assert T12.shape == (3, 3), "Composition shape error"
    print("✓ Composition")
    
    # Test inverse
    T_inv = se2_inverse(T1)
    identity = se2_compose(T1, T_inv)
    assert np.allclose(identity, np.eye(3), atol=1e-6), "Inverse error"
    print("✓ Inverse")
    
    # Test Adjoint
    Ad = se2_adjoint(T1)
    assert Ad.shape == (3, 3), "Adjoint shape error"
    print("✓ Adjoint")
    
    # Test twist transformations
    pose = np.array([100.0, 50.0, np.pi/4])
    body_twist = np.array([0.1, 1.0, 0.5])  # [omega, vx_b, vy_b]
    spatial_twist = body_to_spatial_twist(pose, body_twist)
    body_back = spatial_to_body_twist(pose, spatial_twist)
    assert np.allclose(body_twist, body_back, atol=1e-6), "Twist transform error"
    print("✓ Twist transformations")
    
    print("✓ All SE(2) Math tests passed")
    return True


def test_se2_dynamics():
    """Test SE(2) robot dynamics."""
    print("\n=== Testing SE(2) Dynamics ===")
    
    from src.se2_dynamics import SE2Dynamics, SE2RobotParams
    
    params = SE2RobotParams(mass=1.0, inertia=0.1)
    dynamics = SE2Dynamics(params)
    
    # Test mass matrix
    M = dynamics.get_mass_matrix()
    assert M.shape == (3, 3), "Mass matrix shape error"
    assert np.allclose(M, np.diag([0.1, 1.0, 1.0])), "Mass matrix values error"
    print("✓ Mass matrix")
    
    # Test Coriolis
    twist = np.array([0.2, 1.0, 0.5])  # [omega, vx, vy]
    C_b = dynamics.compute_coriolis_matrix(twist)
    assert C_b.shape == (3, 3), "Coriolis shape error"
    print("✓ Coriolis matrix")
    
    # Test Coriolis wrench
    mu = dynamics.compute_coriolis_wrench(twist)
    assert mu.shape == (3,), "Coriolis wrench shape error"
    # Verify mu = C_b @ twist
    assert np.allclose(mu, C_b @ twist), "Coriolis wrench computation error"
    print("✓ Coriolis wrench")
    
    print("✓ All SE(2) Dynamics tests passed")
    return True


def test_controllers():
    """Test low-level controllers."""
    print("\n=== Testing Controllers ===")
    
    from src.ll_controllers import (
        SE2ImpedanceController,
        SE2ScrewDecomposedImpedanceController,
        PDController, PDGains
    )
    from src.ll_controllers.se2_screw_decomposed_impedance import ScrewImpedanceParams
    from src.se2_dynamics import SE2Dynamics, SE2RobotParams
    
    robot_params = SE2RobotParams(mass=1.0, inertia=0.1)
    
    # Test SE(2) Impedance Controller
    controller = SE2ImpedanceController.create_diagonal_impedance(
        I_d=0.1, m_d=1.0,
        d_theta=5.0, d_x=10.0, d_y=10.0,
        k_theta=20.0, k_x=50.0, k_y=50.0,
        robot_params=robot_params
    )
    
    current_pose = np.array([0.0, 0.0, 0.0])
    desired_pose = np.array([10.0, 5.0, 0.5])
    body_twist_current = np.array([0.0, 0.0, 0.0])
    body_twist_desired = np.array([0.0, 0.0, 0.0])
    
    wrench, info = controller.compute_control(
        current_pose, desired_pose,
        body_twist_current, body_twist_desired
    )
    
    assert wrench.shape == (3,), "Wrench shape error"
    assert 'pose_error' in info, "Info missing pose_error"
    print("✓ SE(2) Impedance Controller")
    
    # Test Screw-Decomposed Controller
    screw_axis = np.array([1.0, 0.0, 0.0])  # Pure rotation
    screw_params = ScrewImpedanceParams(
        D_parallel=5.0, K_parallel=10.0,
        D_perpendicular=20.0, K_perpendicular=100.0
    )
    
    screw_controller = SE2ScrewDecomposedImpedanceController(
        screw_axis=screw_axis,
        params=screw_params,
        robot_dynamics=SE2Dynamics(robot_params)
    )
    
    wrench, info = screw_controller.compute_control(
        current_pose, desired_pose,
        body_twist_current, body_twist_desired
    )
    
    assert wrench.shape == (3,), "Screw wrench shape error"
    assert 'theta' in info, "Info missing theta"
    assert 'e_parallel' in info, "Info missing e_parallel"
    print("✓ Screw-Decomposed Impedance Controller")
    
    # Test controller reset
    controller.reset()
    screw_controller.reset()
    print("✓ Controller reset")
    
    # Test PD Controller
    pd_controller = PDController(PDGains(kp_linear=50.0, kd_linear=10.0))
    desired_velocity = np.array([0.0, 0.0, 0.0])
    wrench = pd_controller.compute_wrench(current_pose, desired_pose, desired_velocity)
    assert wrench.shape == (3,), "PD wrench shape error"
    print("✓ PD Controller")
    
    print("✓ All Controller tests passed")
    return True


def test_environment():
    """Test BiArt environment."""
    print("\n=== Testing BiArt Environment ===")
    
    from src.envs import BiArtEnv
    
    # Test with different joint types
    for joint_type in ['revolute', 'prismatic', 'fixed']:
        env = BiArtEnv(render_mode='rgb_array', joint_type=joint_type)
        
        # Test reset
        obs, info = env.reset()
        assert 'ee_poses' in obs, f"Missing ee_poses for {joint_type}"
        assert 'ee_body_twists' in obs, f"Missing ee_body_twists for {joint_type}"
        assert 'external_wrenches' in obs, f"Missing external_wrenches for {joint_type}"
        assert obs['ee_poses'].shape == (2, 3), f"ee_poses shape error for {joint_type}"
        
        # Test step
        action = np.zeros(6)  # [tau, fx, fy] x 2 (MR convention)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2['ee_poses'].shape == (2, 3), f"Step ee_poses shape error for {joint_type}"
        
        # Test joint axis screws
        B_left, B_right = env.get_joint_axis_screws()
        assert B_left.shape == (3,), f"B_left shape error for {joint_type}"
        assert B_right.shape == (3,), f"B_right shape error for {joint_type}"
        
        env.close()
        print(f"✓ {joint_type} joint")
    
    print("✓ All Environment tests passed")
    return True


def test_rl_environment():
    """Test Impedance Learning Environment."""
    print("\n=== Testing RL Environment ===")
    
    try:
        from src.rl_policy.impedance_learning_env import ImpedanceLearningEnv, ImpedanceLearningConfig
    except ImportError as e:
        print(f"⚠ Skipping RL tests (dependencies not installed): {e}")
        return True
    
    # Test SE(2) impedance controller type
    config = ImpedanceLearningConfig(
        controller_type='se2_impedance',
        max_episode_steps=10
    )
    env = ImpedanceLearningEnv(config=config, render_mode='rgb_array')
    
    obs, info = env.reset()
    assert obs.shape == (30,), f"Observation shape error: {obs.shape}"
    
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (30,), "Step observation shape error"
    
    env.close()
    print("✓ SE(2) Impedance RL Env")
    
    # Test screw-decomposed controller type
    config_screw = ImpedanceLearningConfig(
        controller_type='screw_decomposed',
        max_episode_steps=10
    )
    env_screw = ImpedanceLearningEnv(config=config_screw, render_mode='rgb_array')
    
    obs, info = env_screw.reset()
    action = env_screw.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env_screw.step(action)
    
    env_screw.close()
    print("✓ Screw-Decomposed RL Env")
    
    print("✓ All RL Environment tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SWIVL Core Functionality Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_se2_math()
    except Exception as e:
        print(f"❌ SE(2) Math tests failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_se2_dynamics()
    except Exception as e:
        print(f"❌ SE(2) Dynamics tests failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_controllers()
    except Exception as e:
        print(f"❌ Controller tests failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_environment()
    except Exception as e:
        print(f"❌ Environment tests failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_rl_environment()
    except Exception as e:
        print(f"❌ RL Environment tests failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())




