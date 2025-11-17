#!/bin/bash
# BiArt Environment Setup Script for Conda

set -e  # Exit on error

echo "======================================"
echo "BiArt Environment Setup"
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

# Install conda packages
echo ""
echo "Installing packages from conda-forge..."
conda install -c conda-forge pygame pymunk -y

# Install pip packages
echo ""
echo "Installing packages from pip..."
pip install gymnasium numpy opencv-python shapely

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import gymnasium; print(f'✓ gymnasium {gymnasium.__version__}')"
python -c "import pygame; print(f'✓ pygame {pygame.__version__}')"
python -c "import pymunk; print(f'✓ pymunk {pymunk.version}')"
python -c "import cv2; print(f'✓ opencv {cv2.__version__}')"
python -c "import shapely; print(f'✓ shapely {shapely.__version__}')"
python -c "import numpy; print(f'✓ numpy {numpy.__version__}')"

# Test BiArt environment
echo ""
echo "Testing BiArt environment..."
python -c "import sys; sys.path.insert(0, '.'); import gym_biart; print('✓ BiArt environment imported successfully')"

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test the environment visually, run:"
echo "  python gym_biart/example.py"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
