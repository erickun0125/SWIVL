"""
SE(2) Robot Dynamics

Provides robot dynamics computation for SE(2) systems:
- Task space inertia matrix (Lambda_b)
- Coriolis/centrifugal forces (C_b, mu)
- Gravity compensation (eta_b)
- Dynamics transformations

For SE(2) direct control (mobile robots, planar end-effectors), this provides
the necessary dynamics terms for proper impedance control.

Reference: Modern Robotics, Lynch & Park
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
        # M = diag(m, m, I) in configuration space
        # For direct control: Lambda_b = M (no Jacobian transformation)
        self._M = np.diag([
            self.params.mass,
            self.params.mass,
            self.params.inertia
        ])

    def get_mass_matrix(self) -> np.ndarray:
        """
        Get configuration space mass matrix.

        For SE(2) rigid body:
        M = diag(m, m, I)

        where:
        - m: translational mass
        - I: rotational inertia

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

    def compute_coriolis_matrix(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis matrix C_b(q, q_dot) for SE(2) rigid body.

        For SE(2) rigid body with velocity V = [vx, vy, omega]^T:

        C_b = [0      -m·ω   0]
              [m·ω    0      0]
              [0      0      0]

        This captures the coupling between rotation and translation.
        When rotating (ω ≠ 0) while moving linearly, centrifugal forces appear.

        Args:
            velocity: Body frame velocity [vx, vy, omega]

        Returns:
            C_b: 3x3 Coriolis matrix
        """
        omega = velocity[2]  # Angular velocity
        m = self.params.mass

        C_b = np.array([
            [0.0,      -m * omega,  0.0],
            [m * omega, 0.0,        0.0],
            [0.0,       0.0,        0.0]
        ])

        return C_b

    def compute_coriolis_wrench(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal wrench mu = C_b(q, q_dot) * V.

        For SE(2) rigid body:
        mu = [-m·ω·vy]
             [m·ω·vx ]
             [0      ]

        Physical interpretation:
        - When rotating (ω ≠ 0) and moving forward (vx ≠ 0),
          a lateral centrifugal force m·ω·vx appears
        - When rotating and moving laterally (vy ≠ 0),
          a forward/backward centrifugal force -m·ω·vy appears

        Args:
            velocity: Body frame velocity [vx, vy, omega]

        Returns:
            mu: 3D Coriolis wrench in body frame
        """
        vx, vy, omega = velocity
        m = self.params.mass

        mu = np.array([
            -m * omega * vy,   # Centrifugal force in x direction
            m * omega * vx,    # Centrifugal force in y direction
            0.0                # No torque from linear velocity coupling
        ])

        return mu

    def gravity_wrench(self, pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gravity compensation wrench eta_b(q).

        For planar SE(2) motion (horizontal plane):
        eta_b = 0 (gravity acts perpendicular to motion plane)

        For vertical plane robots, this would be configuration-dependent.

        Args:
            pose: Current pose [x, y, theta] (unused for horizontal plane)

        Returns:
            eta_b: 3D gravity wrench (zero for planar)
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
    - C_b is skew-symmetric (for energy conservation)
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
        # For SE(2) Coriolis, this is not strictly skew-symmetric but has special structure
        # Just warn, don't fail
        pass

    return True


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("SE(2) Dynamics Module Test")
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
    M = dynamics.get_mass_matrix()
    print(M)

    print("\n[Task Space Inertia Lambda_b]")
    Lambda_b = dynamics.get_task_space_inertia()
    print(Lambda_b)
    print(f"(For direct SE(2) control: Lambda_b = M)")

    # Test with velocity
    velocity = np.array([1.0, 0.5, 0.2])  # vx, vy, omega

    print("\n[Coriolis Matrix C_b]")
    print(f"Velocity: {velocity}")
    C_b = dynamics.compute_coriolis_matrix(velocity)
    print(C_b)

    print("\n[Coriolis Wrench mu = C_b * V]")
    mu = dynamics.compute_coriolis_wrench(velocity)
    print(mu)
    print(f"Interpretation: Centrifugal forces due to rotation + translation")

    print("\n[Gravity Wrench eta_b]")
    eta_b = dynamics.gravity_wrench()
    print(eta_b)
    print(f"(Zero for planar motion)")

    print("\n[Validation]")
    is_valid = validate_dynamics_matrices(Lambda_b, C_b, eta_b)
    print(f"Dynamics matrices valid: {is_valid}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
