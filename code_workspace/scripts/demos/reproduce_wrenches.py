
import sys
import os
import numpy as np
import pymunk

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.envs.end_effector_manager import EndEffectorManager, GripperConfig
from src.envs.object_manager import ObjectManager, ObjectConfig

def test_collision_wrenches():
    print(f"Pymunk version: {pymunk.version}")
    
    space = pymunk.Space()
    dt = 0.01
    
    # Create managers
    ee_manager = EndEffectorManager(space, dt=dt)
    object_manager = ObjectManager(space)
    
    # Initialize grippers and object
    # Place object at center
    object_manager.reset(np.array([256, 256, 0]))
    
    # Place left gripper colliding with object
    # Object link1 is at (256, 256). Link width is 11.
    # Place gripper slightly inside the object to force collision
    ee_poses = np.array([
        [256, 256, 0],  # Left gripper at object center (collision!)
        [400, 400, 0]   # Right gripper far away
    ])
    ee_manager.reset(ee_poses)
    
    # Step simulation
    print("Stepping simulation...")
    for _ in range(10):
        space.step(dt)
        ee_manager.update_external_wrenches(dt)
        
    wrenches = ee_manager.get_external_wrenches()
    print(f"External wrenches:\n{wrenches}")
    
    if np.allclose(wrenches, 0):
        print("FAIL: Wrenches are all zero despite collision.")
    else:
        print("SUCCESS: Wrenches are non-zero.")

if __name__ == "__main__":
    test_collision_wrenches()
