#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data1/wzy/WorldModel_NuScenes_HF}"
DATA_DIR="${DATA_DIR:-/mnt/data1/wzy/processed/womd_bev_r1_train100}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data1/wzy/outputs/bev_diffusion_world_model_r2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export CUDA_VISIBLE_DEVICES

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${PROJECT_DIR}/src/07_train_bev_diffusion_world_model.py" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 1 \
  --epochs 20 \
  --lr 5e-5 \
  --hidden_dim 192 \
  --num_heads 8 \
  --prediction_type sample \
  --predict_flow \
  --occ_dilate_kernel 1 \
  --flow_loss_weight 0.25 \
  --far_loss_weight 1.5 \
  --enhanced_map_condition \
  --num_workers 4 \
  --amp \
  --save_steps 100
