# Fixes and Improvements

## Fixed Issues

### 1. Bouncing Problem
**Problem**: Objects and grippers were bouncing and unstable.

**Solutions**:
- Increased space damping from 0.95 to 0.98
- Reduced link width from 15.0 to 12.0 to fit better between jaws
- Reduced grip force from 30N to 25N for stability
- Increased settling steps from 20 to 50 with additional damping
- Added velocity damping during settling (0.95 multiplier)

### 2. Frame Alignment
**Problem**: EE body frame and object grasping frame were not aligned.

**Solution**: Complete redesign of `_position_gripper_to_grasp()`:
- Gripper angle = link angle + 90°
- Gripper x-axis perpendicular to link (jaw opening direction)
- Gripper y-axis parallel to link (jaw extension direction)
- Link positioned exactly between jaws
- Initial jaw offset = link_width/2 + 1.0 (slightly wider than link)

**Frame Convention**:
```
Link frame:
  x-axis: along link length
  y-axis: perpendicular to link

Gripper frame (aligned at 90° to link):
  x-axis: jaw opening/closing direction (perpendicular to link)
  y-axis: jaw extension direction (along link)

Perfect alignment ensures stable grasping!
```

### 3. V-Shaped Revolute Joint
**Problem**: Revolute joint objects were nearly straight, not V-shaped.

**Solution**:
- Changed joint angle range from [-π/6, π/6] to [-2π/3, -π/3] (-120° to -60°)
- Link2 now forms clear V-shape opening downward
- Connection point at end of link1 (link_length/2 offset)
- Better visualization of revolute joint behavior

## New Features

### Static Pose Holding Demo (`demo_static_hold.py`)

A new demo for verifying grasping without movement:

```bash
python demo_static_hold.py revolute   # Revolute joint
python demo_static_hold.py prismatic  # Prismatic joint
python demo_static_hold.py fixed      # Fixed joint
```

**Features**:
- Holds initial pose using PD control
- Real-time visualization of:
  - Current vs desired poses
  - Position/angle errors
  - External wrenches
  - Joint configuration
- Useful for debugging stability and grasping
- Controls: R (reset), ESC (exit)

**Display Information**:
- Current and desired pose for each EE
- Position error magnitude
- External wrench forces
- Object state and joint angles

## Parameter Changes

### Gripper Parameters
```python
gripper_max_opening = 20.0   # Was 18.0
grip_force = 25.0            # Was 30.0 (reduced for stability)
```

### Object Parameters
```python
link_width = 12.0            # Was 15.0 (to fit between jaws)
```

### Physics Parameters
```python
space.damping = 0.98         # Was 0.95 (higher damping)
```

### Settling
```python
settling_steps = 50          # Was 20
velocity_damping = 0.95      # Applied each settling step
```

## Verification

Run tests to verify all improvements:

```bash
# Smoke test
python test_demo_smoke.py

# Static holding (no bouncing)
python demo_static_hold.py revolute

# Interactive teleoperation
python demo_teleoperation.py revolute
```

## Expected Behavior

### Good Grasping Indicators:
- Position error < 2.0 pixels
- Wrench magnitude stable (< 50N)
- No oscillations after settling
- EE frames aligned with object grasping frames
- V-shaped configuration for revolute joints

### If Still Bouncing:
1. Check gripper-link alignment in static demo
2. Reduce grip_force further (try 20N or 15N)
3. Increase settling steps to 100
4. Check that link_width < gripper_max_opening

## Visual Debugging

Use `demo_static_hold.py` to visually verify:
1. ✅ Jaws straddle the link symmetrically
2. ✅ Link is centered between jaws
3. ✅ Gripper orientation perpendicular to link
4. ✅ No drift or oscillation
5. ✅ Wrenches stable over time
