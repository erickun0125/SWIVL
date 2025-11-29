"""
Rule-based Demo with PD Controller

Demonstrates bimanual manipulation using position decomposed PD control.
Automatically moves object from initial pose to desired pose.
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
from src.ll_controllers.pd_controller import MultiGripperPDController, PDGains


class PDControllerDemo(BaseBimanualDemo):
    """
    Demo for PD Controller (position decomposed control).
    
    Uses high-stiffness position control for trajectory tracking.
    This represents the standard deployment of imitation learning policies.
    """
    
    TITLE_COLOR = (128, 0, 0)  # Dark red
    
    @property
    def controller_name(self) -> str:
        return "PD Controller Demo"
    
    def _create_controller(self):
        """Create PD controller with tuned gains."""
        gains = PDGains(
            kp_linear=1500.0,
            kd_linear=1500.0,
            kp_angular=800.0,
            kd_angular=1000.0
        )
        controller = MultiGripperPDController(num_grippers=2, gains=gains, max_force=100.0, max_torque=500)
        controller.set_timestep(self.env.dt)
        return controller
    
    def _compute_wrenches(
        self,
        state: Dict,
        desired_poses: np.ndarray,
        desired_velocities: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Compute PD control wrenches."""
        wrenches = self.controller.compute_wrenches(
            state['ee_poses'],
            desired_poses,
            desired_velocities,
            current_velocities=state['ee_velocities']
        )
        
        # Compute tracking errors for info
        pos_errors = np.linalg.norm(state['ee_poses'][:, :2] - desired_poses[:, :2], axis=1)
        
        info = {
            'tracking_error_left': pos_errors[0],
            'tracking_error_right': pos_errors[1],
        }
        
        return wrenches, info
    
    def _get_controller_info(self, info: Optional[Dict]) -> Dict[str, str]:
        """Get PD controller info for display."""
        if info is None:
            return {}
        return {
            'Tracking Error (L/R)': f"{info['tracking_error_left']:.1f} / {info['tracking_error_right']:.1f} px"
        }


def main():
    run_demo(PDControllerDemo, "PD Controller Bimanual Demo")


if __name__ == "__main__":
    main()
