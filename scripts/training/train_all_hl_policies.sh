#!/bin/bash
# Train all three HL policies sequentially
# Usage: ./scripts/training/train_all_policies.sh

set -e  # Exit on error

DATASET="data/demos"
EPOCHS=2000
EXTRA_ARGS="--no_wrench"

echo "========================================"
echo "Training All High-Level Policies"
echo "========================================"
echo "Dataset: $DATASET"
echo "Epochs per policy: $EPOCHS"
echo "Extra args: $EXTRA_ARGS"
echo "========================================"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swivl

cd "$(dirname "$0")/../.."

# 1. Train Diffusion Policy
echo ""
echo "========================================"
echo "[1/3] Training DIFFUSION Policy"
echo "========================================"
echo "Start time: $(date)"
python scripts/training/train_hl_policy.py \
    --policy diffusion \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    $EXTRA_ARGS

echo "Diffusion training completed at: $(date)"

# 2. Train ACT Policy
echo ""
echo "========================================"
echo "[2/3] Training ACT Policy"
echo "========================================"
echo "Start time: $(date)"
python scripts/training/train_hl_policy.py \
    --policy act \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    $EXTRA_ARGS

echo "ACT training completed at: $(date)"

# 3. Train Flow Matching Policy
echo ""
echo "========================================"
echo "[3/3] Training FLOW MATCHING Policy"
echo "========================================"
echo "Start time: $(date)"
python scripts/training/train_hl_policy.py \
    --policy flow_matching \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    $EXTRA_ARGS

echo "Flow Matching training completed at: $(date)"

echo ""
echo "========================================"
echo "All Training Complete!"
echo "========================================"
echo "Checkpoints saved in: checkpoints/"
echo "  - diffusion_best.pth"
echo "  - act_best.pth"
echo "  - flow_matching_best.pth"
echo "========================================"

