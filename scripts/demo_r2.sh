#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data1/wzy/WorldModel_NuScenes_HF}"
DATA_DIR="${DATA_DIR:-/mnt/data1/wzy/processed/womd_bev_r1_train100}"
RUN_DIR="${RUN_DIR:-/mnt/data1/wzy/outputs/bev_diffusion_world_model_r2}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/epoch_016.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/demo_epoch016_planner_v1_hd}"
INDICES="${INDICES:-10,50,100,300,500,1000}"

python "${PROJECT_DIR}/src/11_make_enterprise_bev_diffusion_demo.py" \
  --data_dir "${DATA_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --split validation \
  --indices "${INDICES}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples 4 \
  --num_inference_steps 50 \
  --threshold 0.35 \
  --fps 4 \
  --scale 6 \
  --video_crf 14 \
  --video_preset slow \
  --video_pix_fmt yuv420p \
  --seed 7 \
  --vis_erode_pred 0
