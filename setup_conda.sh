#!/bin/bash
# SWIVL Environment Setup Script for Conda
# Supports Ubuntu 22.04 with NVIDIA GPUs (CUDA 12.x)

set -e  # Exit on error

echo "======================================"
echo "SWIVL Environment Setup"
echo "======================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="swivl"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Skipping environment creation. Activating existing environment..."
        echo "Run: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.11 -y

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install conda packages (pygame and pymunk work better from conda-forge)
echo ""
echo "Installing packages from conda-forge..."
conda install -c conda-forge pygame pymunk -y

# Install core pip packages
echo ""
echo "Installing core pip packages..."
pip install gymnasium numpy opencv-python shapely scipy

# Install PyTorch with CUDA support
# For CUDA 12.x compatibility (works with driver 580.x and CUDA 13.0)
echo ""
echo "Installing PyTorch with CUDA support..."
echo "Note: Using CUDA 12.8 wheels (compatible with CUDA 12.x+ drivers)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install RL libraries
echo ""
echo "Installing RL libraries..."
pip install stable-baselines3 tensorboard

# Install the SWIVL package in development mode
echo ""
echo "Installing SWIVL package in development mode..."
pip install -e .

# Verify installation
echo ""
echo "======================================"
echo "Verifying installation..."
echo "======================================"

echo ""
echo "Checking core packages..."
python -c "import gymnasium; print(f'✓ gymnasium {gymnasium.__version__}')"
python -c "import pygame; print(f'✓ pygame {pygame.__version__}')"
python -c "import pymunk; print(f'✓ pymunk {pymunk.version}')"
python -c "import cv2; print(f'✓ opencv {cv2.__version__}')"
python -c "import shapely; print(f'✓ shapely {shapely.__version__}')"
python -c "import numpy; print(f'✓ numpy {numpy.__version__}')"
python -c "import scipy; print(f'✓ scipy {scipy.__version__}')"

echo ""
echo "Checking PyTorch and CUDA..."
python -c "
import torch
print(f'✓ torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Checking RL packages..."
python -c "import stable_baselines3; print(f'✓ stable-baselines3 {stable_baselines3.__version__}')"
python -c "import tensorboard; print(f'✓ tensorboard')"

echo ""
echo "Testing SWIVL package..."
python -c "
from src.envs import BiArtEnv
from src.ll_controllers import SE2ImpedanceController, SE2ScrewDecomposedImpedanceController
from src.se2_math import se2_exp, se2_log
from src.se2_dynamics import SE2Dynamics, SE2RobotParams
print('✓ SWIVL package imported successfully')
"

# Quick functional test
echo ""
echo "Running quick functional test..."
python -c "
import warnings
warnings.filterwarnings('ignore')
from src.envs import BiArtEnv
import numpy as np

env = BiArtEnv(render_mode='rgb_array', joint_type='revolute')
obs, info = env.reset()
action = np.zeros(6)
obs2, reward, _, _, _ = env.step(action)
env.close()
print('✓ BiArtEnv functional test passed')
"

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Quick start commands:"
echo "  # Run teleoperation demo"
echo "  python scripts/demos/demo_teleoperation.py revolute"
echo ""
echo "  # Run screw decomposition example"
echo "  python examples/screw_decomposed_bimanual_control.py"
echo ""
echo "  # Run tests"
echo "  python scripts/tests/test_controllers.py"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
