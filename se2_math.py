"""
SE(2) Lie Group Mathematics Utilities

This module provides mathematical operations for SE(2) (Special Euclidean Group in 2D),
including transformations, logarithms, exponentials, and Adjoint representations.

Author: Robotics Lab
Date: 2025-11-18
"""

import numpy as np
from typing import Tuple, Union


class SE2Math:
    """SE(2) Lie Group mathematical operations"""
    
    EPSILON = 1e-10  # Small number for numerical stability
    
    @staticmethod
    def rotation_matrix(theta: float) -> np.ndarray:
        """
        Create 2D rotation matrix from angle
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            R: 2x2 rotation matrix
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s],
                        [s,  c]])
    
    @staticmethod
    def transformation_matrix(x: float, y: float, theta: float) -> np.ndarray:
        """
        Create SE(2) transformation matrix from position and orientation
        
        Args:
            x: X position
            y: Y position
            theta: Orientation angle in radians
            
        Returns:
            T: 3x3 SE(2) transformation matrix
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s, x],
                        [s,  c, y],
                        [0,  0, 1]])
    
    @staticmethod
    def from_xyt(xyt: np.ndarray) -> np.ndarray:
        """
        Create SE(2) matrix from [x, y, theta] vector
        
        Args:
            xyt: 3D vector [x, y, theta]
            
        Returns:
            T: 3x3 SE(2) transformation matrix
        """
        return SE2Math.transformation_matrix(xyt[0], xyt[1], xyt[2])
    
    @staticmethod
    def to_xyt(T: np.ndarray) -> np.ndarray:
        """
        Extract [x, y, theta] from SE(2) matrix
        
        Args:
            T: 3x3 SE(2) transformation matrix
            
        Returns:
            xyt: 3D vector [x, y, theta]
        """
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return np.array([x, y, theta])
    
    @staticmethod
    def inverse(T: np.ndarray) -> np.ndarray:
        """
        Compute inverse of SE(2) matrix
        
        Args:
            T: 3x3 SE(2) transformation matrix
            
        Returns:
            T_inv: Inverse transformation
        """
        R = T[:2, :2]
        p = T[:2, 2]
        
        R_inv = R.T
        p_inv = -R_inv @ p
        
        T_inv = np.eye(3)
        T_inv[:2, :2] = R_inv
        T_inv[:2, 2] = p_inv
        
        return T_inv
    
    @staticmethod
    def hat(xi: np.ndarray) -> np.ndarray:
        """
        Hat operator: R³ -> se(2)
        Maps twist coordinates to se(2) matrix
        
        Args:
            xi: 3D twist vector [omega, v_x, v_y]
            
        Returns:
            xi_hat: 3x3 se(2) matrix
        """
        omega = xi[0]
        v_x = xi[1]
        v_y = xi[2]
        
        return np.array([[0,     -omega, v_x],
                        [omega,  0,     v_y],
                        [0,      0,     0  ]])
    
    @staticmethod
    def vee(xi_hat: np.ndarray) -> np.ndarray:
        """
        Vee operator: se(2) -> R³
        Maps se(2) matrix to twist coordinates
        
        Args:
            xi_hat: 3x3 se(2) matrix
            
        Returns:
            xi: 3D twist vector [omega, v_x, v_y]
        """
        omega = xi_hat[1, 0]
        v_x = xi_hat[0, 2]
        v_y = xi_hat[1, 2]
        
        return np.array([omega, v_x, v_y])
    
    @staticmethod
    def V_matrix(theta: float) -> np.ndarray:
        """
        Compute V matrix for SE(2) exponential map
        
        V(θ) = (sin(θ)/θ) * I + ((1-cos(θ))/θ) * [0 -1; 1 0]
        
        Args:
            theta: Rotation angle
            
        Returns:
            V: 2x2 V matrix
        """
        if abs(theta) < SE2Math.EPSILON:
            # Small angle approximation: V ≈ I
            return np.eye(2)
        
        s = np.sin(theta) / theta
        c = (1 - np.cos(theta)) / theta
        
        return np.array([[s, -c],
                        [c,  s]])
    
    @staticmethod
    def V_matrix_inv(theta: float) -> np.ndarray:
        """
        Compute inverse of V matrix
        
        Args:
            theta: Rotation angle
            
        Returns:
            V_inv: 2x2 inverse V matrix
        """
        if abs(theta) < SE2Math.EPSILON:
            # Small angle approximation: V^-1 ≈ I
            return np.eye(2)
        
        half_theta = theta / 2
        cot_half = 1.0 / np.tan(half_theta)
        
        return 0.5 * theta * np.array([[cot_half, 1],
                                       [-1,       cot_half]])
    
    @staticmethod
    def exp(xi: np.ndarray) -> np.ndarray:
        """
        Exponential map: R³ -> SE(2)
        Maps twist to SE(2) transformation
        
        Args:
            xi: 3D twist vector [omega, v_x, v_y]
            
        Returns:
            T: 3x3 SE(2) transformation matrix
        """
        omega = xi[0]
        v = xi[1:3]
        
        R = SE2Math.rotation_matrix(omega)
        V = SE2Math.V_matrix(omega)
        p = V @ v
        
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = p
        
        return T
    
    @staticmethod
    def log(T: np.ndarray) -> np.ndarray:
        """
        Logarithm map: SE(2) -> R³
        Maps SE(2) transformation to twist vector
        
        Args:
            T: 3x3 SE(2) transformation matrix
            
        Returns:
            xi: 3D twist vector [omega, v_x, v_y] (error vector)
        """
        # Extract rotation angle
        theta = np.arctan2(T[1, 0], T[0, 0])
        
        # Extract position
        p = T[:2, 2]
        
        # Compute V^-1
        V_inv = SE2Math.V_matrix_inv(theta)
        
        # Compute velocity part
        v = V_inv @ p
        
        return np.array([theta, v[0], v[1]])
    
    @staticmethod
    def adjoint(T: np.ndarray) -> np.ndarray:
        """
        Compute Adjoint matrix [Ad_T] for SE(2)
        
        [Ad_T] = [1    0;
                  p_hat*R  R]
        
        where p_hat = [0 -1; 1 0] * p = [-p_y; p_x]
        
        Args:
            T: 3x3 SE(2) transformation matrix
            
        Returns:
            Ad: 3x3 Adjoint matrix
        """
        R = T[:2, :2]
        p = T[:2, 2]
        
        # p_hat = [0 -1; 1 0] * [p_x; p_y] = [-p_y; p_x]
        p_hat = np.array([-p[1], p[0]])
        
        Ad = np.zeros((3, 3))
        Ad[0, 0] = 1.0
        Ad[1:3, 0] = p_hat @ R  # This is a scalar times first column of R
        Ad[1:3, 1:3] = R
        
        # More explicit form:
        Ad = np.array([[1.0,     0.0,        0.0      ],
                       [-p[1],   R[0, 0],   R[0, 1]  ],
                       [p[0],    R[1, 0],   R[1, 1]  ]])
        
        return Ad
    
    @staticmethod
    def adjoint_inv_transpose(T: np.ndarray) -> np.ndarray:
        """
        Compute [Ad_T]^(-T) for wrench transformation
        
        This is used to transform wrenches from spatial to body frame
        
        Args:
            T: 3x3 SE(2) transformation matrix
            
        Returns:
            Ad_inv_T: 3x3 matrix
        """
        T_inv = SE2Math.inverse(T)
        Ad_inv = SE2Math.adjoint(T_inv)
        return Ad_inv.T


class SE2Kinematics:
    """SE(2) Kinematics utilities for coordinate transformations"""
    
    @staticmethod
    def spatial_to_body_twist(T_sb: np.ndarray, spatial_twist: np.ndarray) -> np.ndarray:
        """
        Transform spatial twist to body twist
        
        b_V_b = [Ad_T_sb]^(-1) * s_V_b
        
        Args:
            T_sb: SE(2) transformation from space to body
            spatial_twist: Spatial twist s_V_b
            
        Returns:
            body_twist: Body twist b_V_b
        """
        T_bs = SE2Math.inverse(T_sb)
        Ad_T_bs = SE2Math.adjoint(T_bs)
        return Ad_T_bs @ spatial_twist
    
    @staticmethod
    def body_to_spatial_twist(T_sb: np.ndarray, body_twist: np.ndarray) -> np.ndarray:
        """
        Transform body twist to spatial twist
        
        s_V_b = [Ad_T_sb] * b_V_b
        
        Args:
            T_sb: SE(2) transformation from space to body
            body_twist: Body twist b_V_b
            
        Returns:
            spatial_twist: Spatial twist s_V_b
        """
        Ad_T_sb = SE2Math.adjoint(T_sb)
        return Ad_T_sb @ body_twist
    
    @staticmethod
    def spatial_to_body_wrench(T_sb: np.ndarray, spatial_wrench: np.ndarray) -> np.ndarray:
        """
        Transform spatial wrench to body wrench
        
        b_F = [Ad_T_sb]^(-T) * s_F
        
        Args:
            T_sb: SE(2) transformation from space to body
            spatial_wrench: Spatial wrench s_F
            
        Returns:
            body_wrench: Body wrench b_F
        """
        Ad_inv_T = SE2Math.adjoint_inv_transpose(T_sb)
        return Ad_inv_T @ spatial_wrench
    
    @staticmethod
    def body_to_spatial_wrench(T_sb: np.ndarray, body_wrench: np.ndarray) -> np.ndarray:
        """
        Transform body wrench to spatial wrench
        
        s_F = [Ad_T_sb]^T * b_F
        
        Args:
            T_sb: SE(2) transformation from space to body
            body_wrench: Body wrench b_F
            
        Returns:
            spatial_wrench: Spatial wrench s_F
        """
        Ad_T_sb = SE2Math.adjoint(T_sb)
        return Ad_T_sb.T @ body_wrench


def test_se2_math():
    """Test SE(2) math utilities"""
    print("="*60)
    print("Testing SE(2) Mathematics Utilities")
    print("="*60)
    
    # Test 1: Transformation matrix
    print("\n[Test 1] Transformation Matrix")
    x, y, theta = 1.0, 2.0, np.pi/4
    T = SE2Math.transformation_matrix(x, y, theta)
    print(f"T(x={x}, y={y}, θ={theta:.4f}):")
    print(T)
    
    xyt = SE2Math.to_xyt(T)
    print(f"Extracted [x, y, θ]: {xyt}")
    
    # Test 2: Inverse
    print("\n[Test 2] Inverse Transformation")
    T_inv = SE2Math.inverse(T)
    print("T_inv:")
    print(T_inv)
    print("T @ T_inv:")
    print(T @ T_inv)
    
    # Test 3: Log and Exp
    print("\n[Test 3] Logarithm and Exponential")
    xi = np.array([np.pi/6, 0.5, 1.0])
    T_exp = SE2Math.exp(xi)
    print(f"exp({xi}):")
    print(T_exp)
    
    xi_log = SE2Math.log(T_exp)
    print(f"log(exp(xi)): {xi_log}")
    print(f"Original xi:  {xi}")
    print(f"Difference:   {np.linalg.norm(xi - xi_log):.2e}")
    
    # Test 4: Adjoint
    print("\n[Test 4] Adjoint Matrix")
    Ad = SE2Math.adjoint(T)
    print("Ad_T:")
    print(Ad)
    
    # Test 5: Twist transformation
    print("\n[Test 5] Twist Transformation")
    body_twist = np.array([1.0, 0.5, 0.2])
    spatial_twist = SE2Kinematics.body_to_spatial_twist(T, body_twist)
    print(f"Body twist:   {body_twist}")
    print(f"Spatial twist: {spatial_twist}")
    
    body_twist_recovered = SE2Kinematics.spatial_to_body_twist(T, spatial_twist)
    print(f"Recovered:    {body_twist_recovered}")
    print(f"Error:        {np.linalg.norm(body_twist - body_twist_recovered):.2e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_se2_math()