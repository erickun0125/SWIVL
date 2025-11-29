"""
SWIVL Screw-Decomposed Impedance Control Demo

Demonstrates the SE2ScrewDecomposedImpedanceController for bimanual manipulation
with automatic screw axis detection and object-aware motion decomposition.

This demo implements:
- Layer 2: Reference Twist Field Generator (integrated in controller)
- Layer 4: Screw Axes-Decomposed Impedance Controller

Impedance variables are set manually (would come from RL policy in full SWIVL).
"""

import os
import sys

# Add project root to path for direct script execution
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from typing import Dict, Optional, Tuple

from scripts.demos.demo_base import BaseBimanualDemo, run_demo
from src.ll_controllers.se2_screw_decomposed_impedance import (
    MultiGripperSE2ScrewDecomposedImpedanceController
)
from src.se2_dynamics import SE2RobotParams


class ScrewImpedanceDemo(BaseBimanualDemo):
    """
    Demo for SWIVL Screw-Decomposed Impedance Control.
    
    Features:
    - Automatic screw axis detection from object
    - G-orthogonal projection for bulk/internal motion decomposition
    - Twist-driven impedance control
    - Fighting force measurement via wrench decomposition
    """
    
    TITLE_COLOR = (0, 100, 0)  # Dark green
    
    @property
    def controller_name(self) -> str:
        return "SWIVL Screw-Decomposed Impedance"
    
    def _create_controller(self):
        """Create screw-decomposed impedance controller."""
        robot_params = SE2RobotParams(
            mass=1.2,
            inertia=97.6
        )
        
        # Initial screw axes (will be updated from environment)
        initial_screw_axes = np.array([
            [1.0, 0.0, 0.0],  # Revolute default
            [1.0, 0.0, 0.0],
        ])
        
        controller = MultiGripperSE2ScrewDecomposedImpedanceController(
            num_grippers=2,
            screw_axes=initial_screw_axes,
            robot_params=robot_params,
            max_force=100.0,
            max_torque=500.0
        )
        
        # Set default impedance variables
        controller.set_impedance_variables(
            d_l_parallel=10.0,    # Low damping for internal motion (compliant)
            d_r_parallel=10.0,
            d_l_perp=100.0,       # High damping for bulk motion (stiff)
            d_r_perp=100.0,
            k_p_l=3.0,            # Stiffness for pose error correction
            k_p_r=3.0,
            alpha=10.0            # Characteristic length (pixels)
        )
        
        return controller
    
    def _on_reset(self):
        """Update screw axes from object after reset."""
        screw_axes = self.env.get_joint_axis_screws()
        if screw_axes is not None:
            B_left, B_right = screw_axes
            self.controller.set_screw_axes(np.array([B_left, B_right]))
            print(f"Screw axes from object:")
            print(f"  Left:  {B_left}")
            print(f"  Right: {B_right}")
        else:
            print("No screw axes available (no object or fixed joint)")
    
    def _compute_wrenches(
        self,
        state: Dict,
        desired_poses: np.ndarray,
        desired_velocities: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Compute SWIVL screw-decomposed impedance wrenches."""
        current_poses = state['ee_poses']
        
        # Use body twists if available, otherwise convert
        if 'ee_body_twists' in state:
            current_twists = state['ee_body_twists']
        else:
            current_twists = np.array([
                self.point_velocity_to_body_twist(current_poses[i], state['ee_velocities'][i])
                for i in range(2)
            ])
        
        # Convert desired velocities to body twists
        desired_twists = np.array([
            self.point_velocity_to_body_twist(current_poses[i], desired_velocities[i])
            for i in range(2)
        ])
        
        wrenches, info = self.controller.compute_wrenches(
            current_poses,
            desired_poses,
            current_twists,
            desired_twists,
            external_wrenches=state['external_wrenches']
        )
        
        return wrenches, info
    
    def compute_fighting_force(self, external_wrenches: np.ndarray) -> float:
        """
        Compute fighting force using screw decomposition.
        
        Uses controller's wrench decomposition to get F_perp.
        """
        F_par, F_perp = self.controller.decompose_external_wrenches(external_wrenches)
        return np.sum([np.linalg.norm(f) for f in F_perp])
    
    def _get_controller_info(self, info: Optional[Dict]) -> Dict[str, str]:
        """Get SWIVL controller info for display."""
        if info is None:
            return {}
        
        result = {}
        
        # Get info from left arm
        if 'left' in info and info['left']:
            left = info['left']
            result['L: Pose Err'] = f"{left.get('pose_error_norm', 0):.2f}"
            result['L: Vel Err'] = f"{left.get('velocity_error_norm', 0):.2f}"
        
        # Get info from right arm
        if 'right' in info and info['right']:
            right = info['right']
            result['R: Pose Err'] = f"{right.get('pose_error_norm', 0):.2f}"
            result['R: Vel Err'] = f"{right.get('velocity_error_norm', 0):.2f}"
        
        # Impedance parameters (from first arm)
        if 'left' in info and info['left']:
            left = info['left']
            result['d_∥/d_⊥'] = f"{left.get('d_parallel', 0):.1f} / {left.get('d_perp', 0):.1f}"
            result['k_p / α'] = f"{left.get('k_p', 0):.1f} / {left.get('alpha', 0):.1f}"
        
        return result


def main():
    run_demo(ScrewImpedanceDemo, "SWIVL Screw-Decomposed Impedance Demo")


if __name__ == "__main__":
    main()
