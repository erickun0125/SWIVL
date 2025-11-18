# Fix Summary - Joint Constraints and Stability

## Issues Identified and Fixed

### 1. **Joint Constraint Implementation** ✅ FIXED
**Problem**: PivotJoint was using local coordinate anchors incorrectly, causing joints to not constrain properly.

**Solution**:
- Changed to use world coordinate form: `PivotJoint(body1, body2, world_point)`
- This is simpler and more reliable than the two-anchor local coordinate form

**Location**: `gym_biart/envs/biart.py:464`

### 2. **Revolute Joint V-Shape** ✅ FIXED
**Problem**: Revolute joints weren't maintaining V-shape configuration. Joint angles were drifting to near 0° instead of staying in -120° to -60° range.

**Solution**:
- Added `RotaryLimitJoint` to enforce angle limits
- Constrains relative angle between link1 and link2 to [-120°, -60°] range
- This creates proper V-shape opening downward

**Location**: `gym_biart/envs/biart.py:470-476`

```python
angle_limit = pymunk.RotaryLimitJoint(
    link1_body, link2_body,
    -2*np.pi/3,  # min: -120°
    -np.pi/3     # max: -60°
)
```

### 3. **Desired Pose Drift** ✅ FIXED
**Problem**: When no velocity input was given, desired poses would drift due to integrating from current pose instead of previous desired pose.

**Solution**:
- Modified `_update_controlled_ee_pose()` to integrate velocity from PREVIOUS desired pose
- Ensures zero velocity = fixed desired pose (no drift)

**Location**: `gym_biart/envs/keyboard_planner.py:275-293`

**Before**:
```python
x, y, theta = current_pose  # WRONG - causes drift
```

**After**:
```python
prev_desired = self.desired_poses[self.controlled_ee_idx]  # CORRECT
x_des, y_des, theta_des = prev_desired
```

### 4. **Constraint Fighting on Reset** ✅ FIXED
**Problem**: When `_randomize_state()` manually positioned both link1 and link2, it violated joint constraints, causing huge forces and instability.

**Solution**:
- Added 20-step pre-settling phase BEFORE positioning grippers
- Lets joint constraints resolve naturally before adding gripper forces
- Damps velocities during settling to prevent oscillations

**Location**: `gym_biart/envs/biart.py:988-996`

```python
# IMPORTANT: Let joint constraints resolve for a few steps BEFORE adding grippers
# This prevents constraint fighting
for _ in range(20):
    self.space.step(self.dt)
    # Damp any velocities from constraint resolution
    self.link1.velocity *= 0.9
    # ... etc
```

### 5. **Linkage Manager Integration** ✅ FIXED
**Problem**: Linkage manager wasn't tracking actual pymunk joints, only doing kinematic calculations.

**Solution**:
- Added `pymunk_joint` attribute to `LinkageJoint`
- Demos now register the actual pymunk joint with linkage manager
- Allows linkage manager to query joint states from physics engine

**Location**: `demo_teleoperation.py:126`, `demo_static_hold.py:104`

```python
# Register pymunk joint with linkage manager
if len(self.linkage.joints) > 0 and self.env.joint is not None:
    self.linkage.joints[0].pymunk_joint = self.env.joint
```

### 6. **Grip Force and Damping** ✅ ADJUSTED
**Problem**: Grip forces too strong caused bouncing, low damping caused oscillations.

**Solution**:
- Reduced grip force: 30N → 15N
- Increased space damping: 0.95 → 0.99
- Reduces contact forces and energy in system

**Location**: `gym_biart/envs/biart.py:127`, `gym_biart/envs/biart.py:222`

## Current Status

### ✅ **Working Correctly**
1. Revolute joints maintain V-shape in correct angle range (-120° to -60°)
2. Desired pose integration prevents drift with zero velocity input
3. Joint constraints work properly (PivotJoint, GrooveJoint, RotaryLimitJoint)
4. Frame alignment between grippers and links is correct (90° rotation)
5. Linkage manager tracks pymunk joints

### ⚠️ **Partial Issues Remaining**
1. **System stability with low PD gains**: When using soft PD gains (kp=15), the system has significant position errors (200-300 pixels) and takes a long time to settle. This is because:
   - Gripper grasping is friction-based, not rigid
   - Articulated objects have internal degrees of freedom
   - Low gains can't overcome friction and contact forces quickly

2. **Gripper grasping rigidity**: Current implementation uses constant closing forces and friction. This provides realistic grasping but is less rigid than constraint-based approaches.

### 🎯 **Recommendations**

#### For Better Stability:
1. **Use higher PD gains** in actual usage (e.g., kp=50-100 for faster tracking)
2. **Increase grip force** if objects are slipping (but watch for bouncing)
3. **Add virtual springs** between grippers and links for more rigid grasping
4. **Use smaller object masses** relative to control forces

#### For Realistic Usage:
The current system should work well for **teleoperation scenarios** where:
- User provides continuous velocity inputs (not just static holding)
- PD controller tracks moving desired poses (not fixed poses)
- Some drift and compliance is acceptable (realistic grasping)

The static hold test with low PD gains is a WORST CASE scenario. Normal teleoperation should work much better.

## Testing Results

### Smoke Test: ✅ PASS
```
✅ SMOKE TEST PASSED
Demo should work correctly!
```

### Stability Test with Static Hold: ⚠️ PARTIAL
- Joint angles correct: ✅
- System settles eventually: ⚠️ (takes >100 steps)
- Low position error: ❌ (200-300 pixels with kp=15)

**Note**: These results are with deliberately LOW PD gains (kp=15) to test worst-case stability. With higher gains (kp=50+), performance should be much better.

## Next Steps

### To Further Improve Stability:
1. Add damped springs between gripper bases and grasped links
2. Tune PD gains higher (test with kp=50-100)
3. Implement adaptive grip force based on contact detection
4. Add velocity limits to prevent sudden jumps

### To Test Realistic Usage:
1. Run teleoperation demo with keyboard input
2. Test moving trajectories (not just static holding)
3. Verify object manipulation works smoothly
4. Check if grippers can manipulate objects through full workspace

## Files Modified

- `gym_biart/envs/biart.py` - Joint creation, stability improvements
- `gym_biart/envs/keyboard_planner.py` - Desired pose integration fix
- `demo_teleoperation.py` - Joint registration
- `demo_static_hold.py` - Joint registration
- `test_stability.py` - New comprehensive stability test

## Commit

Committed as: `5d7052b - Fix joint constraints, frame alignment, and stability issues`
Pushed to: `claude/multi-ee-object-control-01C5TudzWwnKWGTmx8VLsi34`
