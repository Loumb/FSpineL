#!/bin/bash
# ============================================================
# DDN Sparse Coefficient Alignment Model -- Multi-GPU Launch Script
# ============================================================
# Usage:
#   bash scripts/run_train_ddn_multi_gpu.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR]
#
# Example:
#   bash scripts/run_train_ddn_multi_gpu.sh
#   bash scripts/run_train_ddn_multi_gpu.sh chexfound/configs/train/lista16_ibot333_highres640.yaml 4 ./outputs/ddn_train
#
# Requirements:
#   - PyTorch 2.x + CUDA
#   - Multiple GPUs on the same machine (at least 2, recommend 4-8)
#   - torchrun (bundled with PyTorch)
#
# Notes:
#   1. Uses torchrun to set RANK/WORLD_SIZE/LOCAL_RANK env vars automatically
#   2. Each GPU runs one process; data is sharded via ShardedInfiniteSampler
#   3. Model uses FSDP (SHARD_GRAD_OP) for memory optimization
#   4. Total batch size = batch_size_per_gpu x NUM_GPUS
# ============================================================

set -e

# --- Default params ---
CONFIG_FILE="${1:-chexfound/configs/train/lista16_ibot333_highres640.yaml}"
NUM_GPUS="${2:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
OUTPUT_DIR="${3:-./outputs/ddn_multi_gpu}"
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Pre-check ---
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "WARNING: NUM_GPUS=$NUM_GPUS, at least 2 GPUs recommended for multi-GPU training"
fi

NGPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NGPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPU detected, check nvidia-smi"
    exit 1
fi
if [ "$NUM_GPUS" -gt "$NGPU_COUNT" ]; then
    echo "WARNING: Requested $NUM_GPUS GPUs but only $NGPU_COUNT available, using $NGPU_COUNT"
    NUM_GPUS=$NGPU_COUNT
fi

# --- Recommended num_workers ---
# num_workers is set in config; here we just report the recommendation
RECOMMENDED_NUM_WORKERS=$((NUM_GPUS * 5))
echo "============================================"
echo "  DDN Multi-GPU Training"
echo "  Config:      $CONFIG_FILE"
echo "  Num GPUs:    $NUM_GPUS"
echo "  Output Dir:  $OUTPUT_DIR"
echo "  Recommended num_workers: $RECOMMENDED_NUM_WORKERS (total, ~5 per GPU)"
echo "============================================"

# --- Environment (optional) ---
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

# --- Launch training ---
# torchrun args:
#   --nproc_per_node  : processes per node (= GPU count)
#   --master_port     : auto-assign port to avoid conflicts
#   --rdzv_backend    : c10d for single-node
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    chexfound/train/train.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "Training finished. Logs and checkpoints saved to: $OUTPUT_DIR"
