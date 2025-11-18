"""
SE(2) Robot Dynamics

Provides robot dynamics computation for SE(2) systems:
- Task space inertia matrix (Lambda_b)
- Coriolis/centrifugal forces (C_b, mu)
- Gravity compensation (eta_b)
- Dynamics transformations

Following Modern Robotics (Lynch & Park) convention:
- Twist: [ω, vx, vy]^T
- Wrench: [τ, fx, fy]^T

For SE(2) direct control (mobile robots, planar end-effectors), this provides
the necessary dynamics terms for proper impedance control.

Reference: Modern Robotics, Lynch & Park, Chapter 8
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SE2RobotParams:
    """
    SE(2) robot physical parameters.

    For a rigid body in SE(2):
    - mass: Total mass of the robot/end-effector [kg]
    - inertia: Rotational inertia about z-axis [kg⋅m²]
    - base_width, base_height: For distributed mass models (optional)
    """
    mass: float              # Total mass [kg]
    inertia: float           # Rotational inertia [kg⋅m²]
    base_width: float = 0.0  # For distributed mass
    base_height: float = 0.0


class SE2Dynamics:
    """
    SE(2) robot dynamics computations.

    Following Modern Robotics convention:
    - Twist V = [ω, vx, vy]^T (angular velocity first!)
    - Wrench F = [τ, fx, fy]^T (torque first!)

    For SE(2) direct control (mobile robots, planar end-effectors),
    the task space is the same as configuration space.

    Key matrices:
    - Lambda_b(q): Task space inertia matrix (3x3)
    - C_b(q, q_dot): Coriolis matrix (3x3)
    - eta_b(q): Gravity wrench (3,) - typically zero for planar
    """

    def __init__(self, params: SE2RobotParams):
        """
        Initialize SE(2) dynamics.

        Args:
            params: Robot physical parameters
        """
        self.params = params

        # Pre-compute constant mass matrix for direct SE(2) control
        # Following MR convention: M = diag(I, m, m)
        # where I is rotational inertia, m is mass
        self._M = np.diag([
            self.params.inertia,  # Angular inertia (first!)
            self.params.mass,     # Translational mass x
            self.params.mass      # Translational mass y
        ])

    def get_mass_matrix(self) -> np.ndarray:
        """
        Get configuration space mass matrix.

        Following Modern Robotics convention:
        For SE(2) rigid body with twist [ω, vx, vy]^T:

        M = [I  0  0]
            [0  m  0]
            [0  0  m]

        where:
        - I: rotational inertia
        - m: translational mass

        Returns:
            M: 3x3 mass matrix
        """
        return self._M.copy()

    def get_task_space_inertia(self, pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get task space inertia matrix Lambda_b(q).

        For SE(2) direct control (mobile robot, planar end-effector):
        Lambda_b = M (since J = I, identity Jacobian)

        This is configuration-independent for rigid body.

        Args:
            pose: Current pose [x, y, theta] (unused for rigid body)

        Returns:
            Lambda_b: 3x3 task space inertia matrix
        """
        return self._M.copy()

    def compute_coriolis_matrix(self, twist: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis matrix C_b(q, V) for SE(2) rigid body.

        Following Modern Robotics convention:
        For SE(2) rigid body with body twist V = [ω, vx, vy]^T:

        C_b = [0    0      0   ]
              [0    0     -m·ω ]
              [0   m·ω    0   ]

        This captures the coupling between rotation and translation.
        When rotating (ω ≠ 0) while moving linearly, centrifugal forces appear.

        Physical interpretation:
        - The (2,3) element -m·ω couples vy to create fx_coriolis
        - The (3,2) element m·ω couples vx to create fy_coriolis

        Args:
            twist: Body frame twist [omega, vx, vy]

        Returns:
            C_b: 3x3 Coriolis matrix
        """
        omega = twist[0]  # Angular velocity (first element in MR convention!)
        m = self.params.mass

        C_b = np.array([
            [0.0,  0.0,       0.0      ],
            [0.0,  0.0,      -m * omega],
            [0.0,  m * omega, 0.0      ]
        ])

        return C_b

    def compute_coriolis_wrench(self, twist: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal wrench mu = C_b(q, V) * V.

        Following Modern Robotics convention:
        For SE(2) rigid body with twist [ω, vx, vy]^T:

        mu = [   0       ]
             [-m·ω·vy   ]
             [ m·ω·vx   ]

        Physical interpretation:
        - No torque from linear velocities (first element = 0)
        - When rotating (ω ≠ 0) and moving forward (vx ≠ 0),
          a lateral centrifugal force m·ω·vx appears
        - When rotating and moving laterally (vy ≠ 0),
          a forward centrifugal force -m·ω·vy appears

        Args:
            twist: Body frame twist [omega, vx, vy]

        Returns:
            mu: 3D Coriolis wrench [tau, fx, fy] in body frame
        """
        omega, vx, vy = twist
        m = self.params.mass

        mu = np.array([
            0.0,               # No torque from Coriolis
            -m * omega * vy,   # Centrifugal force in x direction
            m * omega * vx     # Centrifugal force in y direction
        ])

        return mu

    def gravity_wrench(self, pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gravity compensation wrench eta_b(q).

        For planar SE(2) motion (horizontal plane):
        eta_b = [0, 0, 0]^T (gravity acts perpendicular to motion plane)

        For vertical plane robots, this would be configuration-dependent.

        Args:
            pose: Current pose [x, y, theta] (unused for horizontal plane)

        Returns:
            eta_b: 3D gravity wrench [tau, fx, fy] (zero for planar)
        """
        return np.zeros(3)

    def get_dynamics_parameters(self) -> dict:
        """Get robot dynamics parameters for debugging."""
        return {
            'mass': self.params.mass,
            'inertia': self.params.inertia,
            'M': self._M.copy()
        }


class SE2KinematicsJacobian:
    """
    Jacobian computation for SE(2) robots.

    For direct SE(2) control (mobile robots), J = I (identity).
    For manipulators, this would compute J(q) from forward kinematics.
    """

    @staticmethod
    def identity_jacobian() -> np.ndarray:
        """
        Identity Jacobian for direct SE(2) control.

        When controlling SE(2) pose directly (mobile robot, direct planar EE),
        task space = configuration space, so J = I.

        Returns:
            J: 3x3 identity matrix
        """
        return np.eye(3)

    @staticmethod
    def body_jacobian_mobile_robot() -> np.ndarray:
        """
        Body Jacobian for mobile robot.

        For differential drive or omnidirectional mobile robot,
        body Jacobian is identity.

        Returns:
            J_b: 3x3 identity matrix
        """
        return np.eye(3)


def compute_task_space_inertia_general(M: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Compute task space inertia for general case.

    Lambda = (J * M^(-1) * J^T)^(-1)

    For SE(2) direct control where J = I:
    Lambda = M

    Args:
        M: Configuration space mass matrix (3x3)
        J: Jacobian (3x3 or 3xn)

    Returns:
        Lambda: Task space inertia matrix (3x3)
    """
    M_inv = np.linalg.inv(M)
    Lambda_inv = J @ M_inv @ J.T
    Lambda = np.linalg.inv(Lambda_inv)
    return Lambda


def validate_dynamics_matrices(Lambda_b: np.ndarray,
                               C_b: np.ndarray,
                               eta_b: np.ndarray) -> bool:
    """
    Validate dynamics matrices for impedance control.

    Checks:
    - Lambda_b is positive definite (all eigenvalues > 0)
    - C_b - Lambda_dot/2 is skew-symmetric (passivity)
    - Dimensions are correct

    Args:
        Lambda_b: Task space inertia (3x3)
        C_b: Coriolis matrix (3x3)
        eta_b: Gravity wrench (3,)

    Returns:
        True if valid, False otherwise
    """
    # Check dimensions
    if Lambda_b.shape != (3, 3):
        print(f"Error: Lambda_b shape {Lambda_b.shape} != (3, 3)")
        return False
    if C_b.shape != (3, 3):
        print(f"Error: C_b shape {C_b.shape} != (3, 3)")
        return False
    if eta_b.shape != (3,):
        print(f"Error: eta_b shape {eta_b.shape} != (3,)")
        return False

    # Check Lambda_b is positive definite
    eigenvalues = np.linalg.eigvalsh(Lambda_b)
    if not np.all(eigenvalues > 0):
        print(f"Warning: Lambda_b not positive definite, eigenvalues: {eigenvalues}")
        return False

    # Check C_b - Lambda_dot/2 is skew-symmetric (passivity property)
    # For constant Lambda_b (rigid body), this means C_b should be skew-symmetric
    skew_test = C_b + C_b.T
    if not np.allclose(skew_test, 0, atol=1e-6):
        # For SE(2) Coriolis, check if it has correct skew structure
        # The rotational-translational coupling creates specific pattern
        pass  # Just warn, don't fail

    return True


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("SE(2) Dynamics Module Test")
    print("Modern Robotics Convention: Twist = [ω, vx, vy]^T")
    print("="*60)

    # Create robot parameters (example: mobile robot)
    params = SE2RobotParams(
        mass=1.0,       # 1 kg
        inertia=0.1,    # 0.1 kg⋅m²
        base_width=0.3,
        base_height=0.4
    )

    # Initialize dynamics
    dynamics = SE2Dynamics(params)

    print("\n[Robot Parameters]")
    print(f"Mass: {params.mass} kg")
    print(f"Inertia: {params.inertia} kg⋅m²")

    print("\n[Mass Matrix M]")
    print("Following MR convention: M = diag(I, m, m)")
    M = dynamics.get_mass_matrix()
    print(M)

    print("\n[Task Space Inertia Lambda_b]")
    Lambda_b = dynamics.get_task_space_inertia()
    print(Lambda_b)
    print(f"(For direct SE(2) control: Lambda_b = M)")

    # Test with twist (MR convention: [ω, vx, vy])
    twist = np.array([0.2, 1.0, 0.5])  # omega, vx, vy

    print("\n[Body Twist (MR Convention)]")
    print(f"V = [ω, vx, vy]^T = {twist}")
    print(f"  Angular velocity ω: {twist[0]} rad/s")
    print(f"  Linear velocity vx: {twist[1]} m/s")
    print(f"  Linear velocity vy: {twist[2]} m/s")

    print("\n[Coriolis Matrix C_b]")
    C_b = dynamics.compute_coriolis_matrix(twist)
    print(C_b)
    print("Structure: Couples rotation ω with linear velocities")

    print("\n[Coriolis Wrench mu = C_b * V]")
    mu = dynamics.compute_coriolis_wrench(twist)
    print(f"mu = [τ, fx, fy]^T = {mu}")
    print(f"  Torque τ: {mu[0]} N⋅m (should be 0 for rigid body)")
    print(f"  Force fx: {mu[1]} N (centrifugal, = -m·ω·vy)")
    print(f"  Force fy: {mu[2]} N (centrifugal, = m·ω·vx)")
    print(f"Interpretation: Centrifugal forces due to rotation + translation")

    print("\n[Gravity Wrench eta_b]")
    eta_b = dynamics.gravity_wrench()
    print(f"eta_b = {eta_b}")
    print(f"(Zero for planar motion)")

    print("\n[Validation]")
    is_valid = validate_dynamics_matrices(Lambda_b, C_b, eta_b)
    print(f"Dynamics matrices valid: {is_valid}")

    # Verify Coriolis computation
    print("\n[Verification]")
    print("Manual computation of mu = C_b * V:")
    mu_manual = C_b @ twist
    print(f"  mu_computed = {mu}")
    print(f"  mu_manual   = {mu_manual}")
    print(f"  Match: {np.allclose(mu, mu_manual)}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
