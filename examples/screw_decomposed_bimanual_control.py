"""
Example: Screw-Decomposed Bimanual Impedance Control

Demonstrates coordinated bimanual control using screw decomposition
based on object's joint axis. Shows how to:

1. Get joint axis screws from object
2. Create screw-decomposed impedance controllers
3. Achieve coordinated compliant manipulation

This example shows the natural integration of kinematic constraints
into impedance control for bimanual manipulation tasks.
"""

import numpy as np
import pymunk
from src.envs.object_manager import ObjectManager
from src.ll_controllers.se2_screw_decomposed_impedance import (
    SE2ScrewDecomposedImpedanceController,
    ScrewImpedanceParams
)
from src.se2_dynamics import SE2Dynamics, SE2RobotParams


def main():
    print("=" * 80)
    print("Screw-Decomposed Bimanual Impedance Control Example")
    print("=" * 80)

    # ========================================
    # Setup: Create articulated object
    # ========================================
    print("\n1. Setting up articulated object...")
    print("-" * 80)

    space = pymunk.Space()
    object_manager = ObjectManager(
        space,
        joint_type='revolute',  # V-shaped linkage with revolute joint
        object_params={'link_length': 40.0, 'link_width': 12.0, 'link_mass': 0.5}
    )
    object_manager.reset()

    print("✓ Articulated object created (revolute joint)")

    # ========================================
    # Step 1: Get joint axis screws
    # ========================================
    print("\n2. Getting joint axis screws in each EE frame...")
    print("-" * 80)

    B_left, B_right = object_manager.get_joint_axis_screws()

    print(f"Joint axis screw (left EE):  {B_left}")
    print(f"Joint axis screw (right EE): {B_right}")

    # Interpretation
    r_left = np.linalg.norm(B_left[:2])
    r_right = np.linalg.norm(B_right[:2])

    print(f"\nInterpretation (for unit angular velocity ω=1 rad/s):")
    print(f"  Left EE distance to joint:  {r_left:.2f} pixels")
    print(f"  Right EE distance to joint: {r_right:.2f} pixels")
    print(f"  Left EE linear velocity:    {B_left[:2]} pixels/s")
    print(f"  Right EE linear velocity:   {B_right[:2]} pixels/s")

    # ========================================
    # Step 2: Create robot dynamics
    # ========================================
    print("\n3. Creating robot dynamics models...")
    print("-" * 80)

    robot_params = SE2RobotParams(
        mass=1.0,      # 1 kg
        inertia=0.1    # 0.1 kg⋅m²
    )

    print(f"Robot parameters: mass={robot_params.mass} kg, inertia={robot_params.inertia} kg⋅m²")

    # ========================================
    # Step 3: Design impedance parameters
    # ========================================
    print("\n4. Designing screw-decomposed impedance parameters...")
    print("-" * 80)

    # Strategy:
    # - COMPLIANT along joint axis (allow object to rotate)
    # - STIFF perpendicular to joint (maintain grasp, prevent drift)

    screw_params = ScrewImpedanceParams(
        # Parallel (along joint axis): COMPLIANT
        M_parallel=1.0,       # Match robot inertia
        D_parallel=5.0,       # LOW damping for smooth motion
        K_parallel=10.0,      # LOW stiffness for compliance

        # Perpendicular (maintain grasp): STIFF
        M_perpendicular=1.0,  # Match robot inertia
        D_perpendicular=20.0, # HIGH damping to prevent oscillation
        K_perpendicular=100.0 # HIGH stiffness to maintain grasp
    )

    print("Impedance parameters:")
    print(f"  Parallel (along joint):")
    print(f"    M_∥ = {screw_params.M_parallel:.1f} kg")
    print(f"    D_∥ = {screw_params.D_parallel:.1f} N⋅s/m")
    print(f"    K_∥ = {screw_params.K_parallel:.1f} N/m")
    print(f"  Perpendicular (maintain grasp):")
    print(f"    M_⊥ = {screw_params.M_perpendicular:.1f} kg")
    print(f"    D_⊥ = {screw_params.D_perpendicular:.1f} N⋅s/m")
    print(f"    K_⊥ = {screw_params.K_perpendicular:.1f} N/m")
    print(f"\n  Stiffness ratio K_⊥/K_∥ = {screw_params.K_perpendicular/screw_params.K_parallel:.1f}x")
    print(f"  → Object can rotate freely, but grasp is maintained!")

    # ========================================
    # Step 4: Create screw-decomposed controllers
    # ========================================
    print("\n5. Creating screw-decomposed impedance controllers...")
    print("-" * 80)

    controller_left = SE2ScrewDecomposedImpedanceController(
        screw_axis=B_left,
        params=screw_params,
        robot_dynamics=SE2Dynamics(robot_params),
        model_matching=True,   # Use model matching for passivity
        use_feedforward=True,
        max_force=100.0,
        max_torque=50.0
    )

    controller_right = SE2ScrewDecomposedImpedanceController(
        screw_axis=B_right,
        params=screw_params,
        robot_dynamics=SE2Dynamics(robot_params),
        model_matching=True,
        use_feedforward=True,
        max_force=100.0,
        max_torque=50.0
    )

    print("✓ Controllers created with screw decomposition")
    print("  - Left controller uses B_left")
    print("  - Right controller uses B_right")

    # ========================================
    # Step 5: Simulate control scenario
    # ========================================
    print("\n6. Simulating control scenario...")
    print("-" * 80)

    # Get current grasping poses
    grasping_poses = object_manager.get_grasping_poses()
    current_pose_left = grasping_poses['left']
    current_pose_right = grasping_poses['right']

    print(f"Current poses (world frame):")
    print(f"  Left EE:  {current_pose_left}")
    print(f"  Right EE: {current_pose_right}")

    # Scenario: Rotate object by moving left EE
    # Desired: Left EE moves along circular arc (following joint motion)
    #          Right EE should follow automatically
    desired_pose_left = current_pose_left + np.array([5.0, -3.0, 0.1])
    desired_pose_right = current_pose_right + np.array([-5.0, 3.0, 0.1])

    # Zero velocity and acceleration for this example
    current_twist_left = np.zeros(3)
    current_twist_right = np.zeros(3)
    desired_twist_left = np.zeros(3)
    desired_twist_right = np.zeros(3)

    print(f"\nDesired poses (small displacement):")
    print(f"  Left EE:  {desired_pose_left}")
    print(f"  Right EE: {desired_pose_right}")

    # ========================================
    # Step 6: Compute control wrenches
    # ========================================
    print("\n7. Computing control wrenches...")
    print("-" * 80)

    F_cmd_left, info_left = controller_left.compute_control(
        current_pose=current_pose_left,
        desired_pose=desired_pose_left,
        body_twist_current=current_twist_left,
        body_twist_desired=desired_twist_left
    )

    F_cmd_right, info_right = controller_right.compute_control(
        current_pose=current_pose_right,
        desired_pose=desired_pose_right,
        body_twist_current=current_twist_right,
        body_twist_desired=desired_twist_right
    )

    print("Left EE control:")
    print(f"  Pose error: {info_left['pose_error']}")
    print(f"  Decomposition:")
    print(f"    θ (along joint) = {info_left['theta']:.4f}")
    print(f"    e_∥ = {info_left['e_parallel']}")
    print(f"    e_⊥ = {info_left['e_perp']}")
    print(f"  Control wrench: {F_cmd_left}")

    print(f"\nRight EE control:")
    print(f"  Pose error: {info_right['pose_error']}")
    print(f"  Decomposition:")
    print(f"    θ (along joint) = {info_right['theta']:.4f}")
    print(f"    e_∥ = {info_right['e_parallel']}")
    print(f"    e_⊥ = {info_right['e_perp']}")
    print(f"  Control wrench: {F_cmd_right}")

    # ========================================
    # Step 7: Analyze control behavior
    # ========================================
    print("\n8. Analyzing control behavior...")
    print("-" * 80)

    # Compute wrench decomposition
    tau_left, F_parallel_left, F_perp_left = controller_left.decompose_vector(F_cmd_left)
    tau_right, F_parallel_right, F_perp_right = controller_right.decompose_vector(F_cmd_right)

    print("Wrench decomposition (left):")
    print(f"  τ_∥ (generalized force along joint): {tau_left:.4f}")
    print(f"  F_∥ (parallel component):            {F_parallel_left}")
    print(f"  F_⊥ (perpendicular component):        {F_perp_left}")
    print(f"  |F_⊥| / |F_∥| = {np.linalg.norm(F_perp_left) / (np.linalg.norm(F_parallel_left) + 1e-6):.2f}")

    print(f"\nWrench decomposition (right):")
    print(f"  τ_∥ (generalized force along joint): {tau_right:.4f}")
    print(f"  F_∥ (parallel component):            {F_parallel_right}")
    print(f"  F_⊥ (perpendicular component):        {F_perp_right}")
    print(f"  |F_⊥| / |F_∥| = {np.linalg.norm(F_perp_right) / (np.linalg.norm(F_parallel_right) + 1e-6):.2f}")

    # ========================================
    # Step 8: Compare with standard controller
    # ========================================
    print("\n9. Comparison with standard impedance control...")
    print("-" * 80)

    from src.ll_controllers.se2_impedance_controller import SE2ImpedanceController

    # Standard controller with same average impedance
    K_d_std = np.diag([
        (screw_params.K_parallel + 2*screw_params.K_perpendicular) / 3,
        (screw_params.K_parallel + 2*screw_params.K_perpendicular) / 3,
        (screw_params.K_parallel + 2*screw_params.K_perpendicular) / 3
    ])
    D_d_std = np.diag([
        (screw_params.D_parallel + 2*screw_params.D_perpendicular) / 3,
        (screw_params.D_parallel + 2*screw_params.D_perpendicular) / 3,
        (screw_params.D_parallel + 2*screw_params.D_perpendicular) / 3
    ])
    M_d_std = np.diag([robot_params.mass, robot_params.mass, robot_params.inertia])

    controller_std = SE2ImpedanceController(
        M_d=M_d_std,
        D_d=D_d_std,
        K_d=K_d_std,
        robot_dynamics=SE2Dynamics(robot_params),
        model_matching=True
    )

    # Compute standard control wrench
    e_left_std = controller_std.compute_pose_error_from_arrays(
        current_pose_left, desired_pose_left
    )
    V_e_left_std = controller_std.compute_velocity_error(
        current_twist_left, desired_twist_left
    )
    F_cmd_std = controller_std.compute_control_wrench(
        e_left_std, V_e_left_std, current_twist_left
    )

    print("Standard impedance control (left EE):")
    print(f"  K_d (average): diag([{K_d_std[0,0]:.1f}, {K_d_std[1,1]:.1f}, {K_d_std[2,2]:.1f}])")
    print(f"  Control wrench: {F_cmd_std}")

    print(f"\nScrew-decomposed impedance control (left EE):")
    print(f"  Control wrench: {F_cmd_left}")

    print(f"\nDifference:")
    print(f"  ΔF = {F_cmd_left - F_cmd_std}")
    print(f"  → Screw decomposition applies different gains per direction!")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("Summary: Benefits of Screw-Decomposed Impedance Control")
    print("=" * 80)

    print("\n✓ Configuration-invariant constraint representation")
    print("  - Joint axis B_left, B_right independent of object pose")
    print("  - Natural coordinate system aligned with kinematic constraints")

    print("\n✓ Independent impedance tuning per direction")
    print("  - Compliant along joint axis (K_∥ = 10 N/m)")
    print("  - Stiff perpendicular to joint (K_⊥ = 100 N/m)")
    print("  - Ratio K_⊥/K_∥ = 10x ensures grasp maintenance")

    print("\n✓ Coordinated bimanual control")
    print("  - Both EEs respect object's kinematic constraint")
    print("  - Natural coupling through screw decomposition")
    print("  - No explicit coordination required!")

    print("\n✓ Applications:")
    print("  - Bimanual manipulation of articulated objects")
    print("  - Compliant assembly tasks")
    print("  - Contact-rich manipulation")
    print("  - Grasp constraint satisfaction")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
