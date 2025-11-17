# SWIVL - Screw and Wrench informed Impedance Variable Learning

Bimanual manipulation of articulated objects with inter-force interaction using reinforcement learning.

## 🎯 Project Overview

This repository contains the implementation of SWIVL, a framework for learning impedance parameters for bimanual manipulation of articulated objects. The system uses:

- **Task-space impedance control** for robot manipulation
- **Low-level policy** that learns impedance variables (stiffness, damping) via RL
- **High-level policy** that provides desired trajectories
- **Dual-arm coordination** for manipulating shared 1-DOF linkage objects

## 🚀 Quick Start

### 30 Second Setup

```bash
# 1. Setup conda environment and install dependencies
bash setup_conda.sh

# 2. Activate environment
conda activate swivl

# 3. Run visualization!
python run_visualization.py
```

See [QUICKSTART.md](QUICKSTART.md) for more options.

## 📁 Repository Structure

```
SWIVL/
├── gym_pusht/              # Reference PushT environment
├── gym_biart/              # BiArt SE(2) environment (main)
│   ├── envs/
│   │   ├── biart.py       # Main environment implementation
│   │   └── pymunk_override.py
│   ├── example.py          # Usage examples
│   └── README.md           # Detailed documentation
├── run_visualization.py    # Visualization script
├── test_biart_simple.py    # Test suite
├── setup_conda.sh          # Automated setup script
├── QUICKSTART.md           # Quick start guide
├── SETUP_GUIDE.md          # Detailed setup instructions
└── requirements.txt        # Python dependencies
```

## 🤖 BiArt Environment

**BiArt** (Bimanual Articulated object manipulation) is our SE(2) environment featuring:

- ✅ Dual dynamic robot grippers with U-shaped (ㄷ) design
- ✅ Wrench command control (force + moment in body frame)
- ✅ Articulated objects (revolute, prismatic, fixed joints)
- ✅ External wrench sensing
- ✅ Physics-based simulation with Pymunk
- ✅ Gymnasium-compatible API

### Environment Features

**Action Space** (6D):
```python
[left_fx, left_fy, left_tau, right_fx, right_fy, right_tau]
```

**Observation Space** (18D):
```python
[left_gripper(x,y,θ), right_gripper(x,y,θ),
 link1(x,y,θ), link2(x,y,θ),
 external_wrench_left(fx,fy,τ), external_wrench_right(fx,fy,τ)]
```

**Reward Function**:
- Tracking reward: exponential reward based on position/orientation error
- Safety penalty: penalizes excessive contact forces

## 📖 Usage Examples

### Basic Usage

```python
import gymnasium as gym
import gym_biart

# Create environment
env = gym.make("gym_biart/BiArt-v0",
               render_mode="human",
               joint_type="revolute")  # or "prismatic", "fixed"

# Run episode
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Visualization Options

```bash
# Visual mode with different joint types
python run_visualization.py --mode visual --joint revolute
python run_visualization.py --mode visual --joint prismatic
python run_visualization.py --mode visual --joint fixed

# Slow motion for detailed observation
python run_visualization.py --mode visual --joint revolute --slow

# Test specific control patterns
python run_visualization.py --mode controlled --joint revolute

# Compare all joint types
python run_visualization.py --mode compare

# Headless mode (for servers) with image saving
python run_visualization.py --mode headless --save-images
```

## 🔧 Installation

### Requirements

- Python 3.11+
- Conda or Miniconda (recommended)

### Method 1: Automated Setup (Recommended)

```bash
bash setup_conda.sh
conda activate swivl
```

### Method 2: Manual Setup

```bash
# Create conda environment
conda create -n swivl python=3.11 -y
conda activate swivl

# Install packages
conda install -c conda-forge pygame pymunk -y
pip install gymnasium numpy opencv-python shapely
```

### Method 3: Using pip only

```bash
pip install -r requirements.txt
```

## 🧪 Testing

```bash
# Quick test (no display)
python test_biart_simple.py

# Visual test
python gym_biart/example.py

# Full test suite
python run_visualization.py --mode visual --steps 500
```

## 📚 Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Get started in 30 seconds
- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed installation and troubleshooting
- **BiArt Docs**: [gym_biart/README.md](gym_biart/README.md) - Environment API and details

## 🔬 Research Context

This work is part of research on impedance control for bimanual manipulation:

- **Problem**: Bimanual manipulation of articulated objects with inter-force interaction
- **Approach**: Learn impedance variables (stiffness, damping) via RL
- **Framework**: Task-space impedance control with dual-arm coordination
- **Environments**:
  - SE(2): BiArt (Pymunk-based, implemented ✅)
  - SE(3): Franka FR3 dual-arm (IsaacLab, planned 🚧)

## 🛣️ Roadmap

### SE(2) Environment (BiArt) - ✅ Complete
- [x] Dual dynamic grippers with U-shape
- [x] Wrench command control
- [x] Articulated objects (3 joint types)
- [x] External wrench sensing (basic)
- [x] Reward function (tracking + safety)
- [ ] Proper collision-based wrench sensing
- [ ] High-level policy interface
- [ ] Impedance control integration

### SE(3) Environment - 🚧 Planned
- [ ] IsaacLab integration
- [ ] Franka FR3 dual-arm setup
- [ ] 3D articulated objects
- [ ] 6-DOF manipulation

### Learning Pipeline - 🚧 Planned
- [ ] PPO/SAC implementation
- [ ] Impedance parameter learning
- [ ] Trajectory optimization
- [ ] Transfer learning SE(2) → SE(3)

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue for bugs or feature requests
- Check documentation before asking questions

## 📄 License

Research project - License TBD

## 🙏 Acknowledgments

- PushT environment as reference implementation
- Pymunk physics engine
- Gymnasium framework

## 📧 Contact

For research inquiries, please open an issue on this repository.

---

**Status**: SE(2) environment complete and tested ✅
**Next**: Implement RL training pipeline and SE(3) environment
