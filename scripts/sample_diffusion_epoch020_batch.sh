#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data1/wzy/WorldModel_NuScenes_HF}"
DATA_DIR="${DATA_DIR:-/mnt/data1/wzy/processed/womd_bev_r1_train100}"
RUN_DIR="${RUN_DIR:-/mnt/data1/wzy/outputs/bev_diffusion_world_model}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/epoch_020.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/sample_epoch020_vis_erode1}"
SPLIT="${SPLIT:-validation}"
INDICES="${INDICES:-10 50 100 300 500 1000}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
VIS_ERODE_PRED="${VIS_ERODE_PRED:-1}"
VIS_THRESHOLD="${VIS_THRESHOLD:-0.12}"

mkdir -p "${OUTPUT_DIR}"

for index in ${INDICES}; do
  echo "[sample] index=${index}"
  python "${PROJECT_DIR}/src/08_sample_bev_diffusion_world_model.py" \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --index "${index}" \
    --output "${OUTPUT_DIR}/sample_${index}.png" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --vis_erode_pred "${VIS_ERODE_PRED}" \
    --vis_threshold "${VIS_THRESHOLD}"
done

echo "[done] samples saved to ${OUTPUT_DIR}"
