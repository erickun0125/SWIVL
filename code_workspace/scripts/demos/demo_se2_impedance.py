"""
SE(2) Impedance Control Demo

Demonstrates bimanual manipulation using classical SE(2) impedance control.
Provides compliant behavior with configurable mass-spring-damper dynamics.
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
from src.ll_controllers.se2_impedance_controller import MultiGripperSE2ImpedanceController
from src.se2_dynamics import SE2RobotParams


class SE2ImpedanceDemo(BaseBimanualDemo):
    """
    Demo for SE(2) Impedance Controller.
    
    Uses classical impedance control with configurable M, D, K matrices.
    Provides compliant behavior but without object-aware decomposition.
    """
    
    TITLE_COLOR = (0, 0, 128)  # Dark blue
    
    @property
    def controller_name(self) -> str:
        return "SE(2) Impedance Demo"
    
    def _create_controller(self):
        """Create SE(2) impedance controller."""
        # Robot Parameters (from GripperConfig calculations)
        robot_params = SE2RobotParams(
            mass=1.2,
            inertia=97.6
        )
        
        controller = MultiGripperSE2ImpedanceController(
            num_grippers=2,
            robot_params=robot_params,
            M_d=np.diag([97.6, 1.2, 1.2]),     # Match robot inertia
            D_d=np.diag([50.0, 10.0, 10.0]),   # Damping
            K_d=np.diag([200.0, 50.0, 50.0]),  # Stiffness
            model_matching=True,
            max_force=100.0,
            max_torque=500.0
        )
        return controller
    
    def _compute_wrenches(
        self,
        state: Dict,
        desired_poses: np.ndarray,
        desired_velocities: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Compute SE(2) impedance control wrenches."""
        current_poses = state['ee_poses']
        
        # Convert velocities to body twists
        current_twists = []
        desired_twists = []
        
        for i in range(2):
            current_twists.append(
                self.point_velocity_to_body_twist(current_poses[i], state['ee_velocities'][i])
            )
            desired_twists.append(
                self.point_velocity_to_body_twist(current_poses[i], desired_velocities[i])
            )
        
        current_twists = np.array(current_twists)
        desired_twists = np.array(desired_twists)
        
        wrenches = self.controller.compute_wrenches(
            current_poses,
            desired_poses,
            current_twists,
            desired_twists,
            desired_accels=None,
            external_wrenches=state['external_wrenches']
        )
        
        # Compute pose errors for info
        pos_errors = np.linalg.norm(current_poses[:, :2] - desired_poses[:, :2], axis=1)
        vel_errors = np.linalg.norm(current_twists - desired_twists, axis=1)
        
        info = {
            'pose_error_left': pos_errors[0],
            'pose_error_right': pos_errors[1],
            'vel_error_left': vel_errors[0],
            'vel_error_right': vel_errors[1],
        }
        
        return wrenches, info
    
    def _get_controller_info(self, info: Optional[Dict]) -> Dict[str, str]:
        """Get SE(2) impedance info for display."""
        if info is None:
            return {}
        return {
            'Pose Error (L/R)': f"{info['pose_error_left']:.1f} / {info['pose_error_right']:.1f} px",
            'Vel Error (L/R)': f"{info['vel_error_left']:.2f} / {info['vel_error_right']:.2f}",
        }


def main():
    run_demo(SE2ImpedanceDemo, "SE(2) Impedance Control Demo")


if __name__ == "__main__":
    main()
