#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data1/wzy/WorldModel_NuScenes_HF}"
DATA_DIR="${DATA_DIR:-/mnt/data1/wzy/processed/womd_bev_r1_train100}"
RUN_DIR="${RUN_DIR:-/mnt/data1/wzy/outputs/bev_diffusion_world_model_r4_x0_sample_pred}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/full_eval_epoch_select}"
DEMO_DIR="${DEMO_DIR:-${RUN_DIR}/enterprise_demo_best_full_eval}"

EPOCHS="${EPOCHS:-012 013 018 020}"
SPLIT="${SPLIT:-validation}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
THRESHOLD="${THRESHOLD:-0.35}"
SEED="${SEED:-7}"

DEMO_INDICES="${DEMO_INDICES:-10,50,100,300,500,1000}"
DEMO_NUM_SAMPLES="${DEMO_NUM_SAMPLES:-4}"
DEMO_NUM_INFERENCE_STEPS="${DEMO_NUM_INFERENCE_STEPS:-50}"
DEMO_FPS="${DEMO_FPS:-2}"
VIS_ERODE_PRED="${VIS_ERODE_PRED:-1}"

mkdir -p "${EVAL_DIR}" "${DEMO_DIR}"

for epoch in ${EPOCHS}; do
  checkpoint="${RUN_DIR}/epoch_${epoch}.pt"
  output_json="${EVAL_DIR}/epoch_${epoch}.json"
  echo "[eval] epoch_${epoch}: ${checkpoint}"
  python "${PROJECT_DIR}/src/09_eval_bev_diffusion_world_model.py" \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${checkpoint}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --threshold "${THRESHOLD}" \
    --seed "${SEED}" \
    --output_json "${output_json}"
done

python - "${EVAL_DIR}" ${EPOCHS} <<'PY'
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
epochs = sys.argv[2:]
rows = []
for epoch in epochs:
    path = eval_dir / f"epoch_{epoch}.json"
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows.append(
        {
            "epoch": epoch,
            "checkpoint": f"epoch_{epoch}.pt",
            "occ_iou": metrics["occ_iou"],
            "occ_iou_near": metrics["occ_iou_near"],
            "occ_iou_mid": metrics["occ_iou_mid"],
            "occ_iou_far": metrics["occ_iou_far"],
            "pred_pos_ratio": metrics["pred_pos_ratio"],
            "true_pos_ratio": metrics["true_pos_ratio"],
            "pred_prob_mean": metrics["pred_prob_mean"],
            "evaluated_batches": metrics.get("evaluated_batches"),
            "evaluated_samples": metrics.get("evaluated_samples"),
        }
    )

best = max(rows, key=lambda row: row["occ_iou"])
summary = {"selection_metric": "occ_iou", "best": best, "candidates": rows}
summary_path = eval_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("[summary]")
for row in rows:
    print(
        f"epoch_{row['epoch']} "
        f"occ={row['occ_iou']:.6f} "
        f"near={row['occ_iou_near']:.6f} "
        f"mid={row['occ_iou_mid']:.6f} "
        f"far={row['occ_iou_far']:.6f} "
        f"pred_pos={row['pred_pos_ratio']:.6f}"
    )
print(f"[best] epoch_{best['epoch']} by occ_iou={best['occ_iou']:.6f}")
print(summary_path)
PY

BEST_EPOCH="$(
  python - "${EVAL_DIR}/summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(summary["best"]["epoch"])
PY
)"
BEST_CKPT="${RUN_DIR}/epoch_${BEST_EPOCH}.pt"
BEST_DEMO_DIR="${DEMO_DIR}/epoch_${BEST_EPOCH}"

echo "[demo] best checkpoint: ${BEST_CKPT}"
python "${PROJECT_DIR}/src/11_make_enterprise_bev_diffusion_demo.py" \
  --data_dir "${DATA_DIR}" \
  --checkpoint "${BEST_CKPT}" \
  --split "${SPLIT}" \
  --indices "${DEMO_INDICES}" \
  --output_dir "${BEST_DEMO_DIR}" \
  --num_samples "${DEMO_NUM_SAMPLES}" \
  --num_inference_steps "${DEMO_NUM_INFERENCE_STEPS}" \
  --threshold "${THRESHOLD}" \
  --fps "${DEMO_FPS}" \
  --seed "${SEED}" \
  --vis_erode_pred "${VIS_ERODE_PRED}"

echo "[done] eval summary: ${EVAL_DIR}/summary.json"
echo "[done] demo output: ${BEST_DEMO_DIR}"
