import argparse
import h5py
import cv2
import numpy as np
import time
import os

def visualize_data(file_path, fps=30):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    with h5py.File(file_path, 'r') as f:
        # Load data
        images = f['obs/images'][:]
        ee_poses = f['obs/ee_poses'][:]
        external_wrenches = f['obs/external_wrenches'][:]
        desired_poses = f['action/desired_poses'][:]
        
        num_steps = f.attrs['num_steps']
        print(f"Total steps: {num_steps}")
        print(f"Images shape: {images.shape}")
        
    # Playback
    delay = 1.0 / fps
    
    print("\nControls:")
    print("  Space: Pause/Resume")
    print("  ESC/Q: Quit")
    print("  Any other key: Next frame (when paused)")
    
    paused = False
    i = 0
    
    while i < num_steps:
        # Get data for current step
        img = images[i]
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize for better visibility (96x96 is small)
        display_img = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Add text info
        pose = ee_poses[i, 0] # Left EE
        wrench = external_wrenches[i, 0]
        wrench_mag = np.linalg.norm(wrench[1:3])
        
        text_lines = [
            f"Step: {i}/{num_steps}",
            f"EE0 Pose: x={pose[0]:.1f} y={pose[1]:.1f}",
            f"EE0 Wrench: |F|={wrench_mag:.1f}",
        ]
        
        for idx, line in enumerate(text_lines):
            cv2.putText(display_img, line, (10, 30 + idx*25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Data Visualization", display_img)
        
        key = cv2.waitKey(int(delay * 1000) if not paused else 0) & 0xFF
        
        if key == ord('q') or key == 27: # ESC
            break
        elif key == ord(' '):
            paused = not paused
        
        if not paused:
            i += 1
        elif key != 255 and key != ord(' '):
            # Advance one frame if paused and key pressed (other than space)
            i += 1
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize collected HDF5 data")
    parser.add_argument("file", help="Path to HDF5 file")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    
    args = parser.parse_args()
    
    visualize_data(args.file, args.fps)
