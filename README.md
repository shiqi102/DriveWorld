# DriveWorld

BEV Occupancy Diffusion World Model for autonomous-driving scene forecasting and planning-on-prediction demos.

DriveWorld predicts future BEV occupancy and occupancy flow from historical BEV scenes, agent states, vector maps, raster maps and traffic-light states. The latest r2 model uses map-enhanced raster conditions, temporal BEV attention and cross-attention over lane graph / agent / traffic-light tokens. A lane-graph planner then searches candidate ego plans in the predicted future world and visualizes risk.

## Highlights

- **Generative BEV world model**: DDIM conditional diffusion over future BEV occupancy / flow.
- **Map-aware conditioning**: lane masks, drivable priors, lane-distance features and vector-map tokens.
- **Temporal BEV attention**: history BEV frames are encoded as motion-aware temporal features.
- **Cross attention**: denoising BEV features attend to lane graph, traffic-light and dynamic-agent tokens.
- **Planning on prediction**: lane-following ego candidates are scored by collision, uncertainty, off-road, smoothness and progress costs.
- **Portfolio demo**: exports PNG overview and MP4 rollout with current scene, GT future, predicted future, multi-sample uncertainty, flow and ego-plan risk.

## Model Inputs

```text
past_bev:             historical BEV occupancy
agent_features:       historical agent states
map_bev:              lane / road-line / road-edge raster map
map_vectors:          vectorized lane graph
traffic_lights:       historical traffic-light states
sensor_context:       placeholder for future camera / LiDAR features
```

## Model Outputs

```text
future_occ:           future BEV occupancy
future_flow:          future occupancy motion field
sample_uncertainty:   multi-sample diffusion uncertainty
ego_plan_risk:        lane-graph candidate route risk
```

## Latest r2 Result

Full validation evaluation on the processed WOMD BEV validation split:

```text
occ_iou       0.6169
near_iou      0.8670
mid_iou       0.6403
far_iou       0.4772
pred_pos      0.00487
true_pos      0.00498
```

## Repository Layout

```text
DriveWorld/
  configs/                         training configs
  scripts/
    train_r2.sh                    latest r2 training entry
    demo_r2.sh                     latest r2 demo entry
  src/
    womd_bev.py                    WOMD -> BEV preprocessing
    bev_diffusion_world_model.py   latest diffusion world model
    07_train_bev_diffusion_world_model.py
    08_sample_bev_diffusion_world_model.py
    09_eval_bev_diffusion_world_model.py
    11_make_enterprise_bev_diffusion_demo.py
```

Large data, checkpoints and generated videos are intentionally ignored by git.

## Training

```bash
bash DriveWorld/scripts/train_r2.sh
```

Override runtime paths when needed:

```bash
PROJECT_DIR=/mnt/data1/wzy/WorldModel_NuScenes_HF \
DATA_DIR=/mnt/data1/wzy/processed/womd_bev_r1_train100 \
OUTPUT_DIR=/mnt/data1/wzy/outputs/bev_diffusion_world_model_r2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash DriveWorld/scripts/train_r2.sh
```

## Evaluation

```bash
python DriveWorld/src/09_eval_bev_diffusion_world_model.py \
  --data_dir /mnt/data1/wzy/processed/womd_bev_r1_train100 \
  --checkpoint /mnt/data1/wzy/outputs/bev_diffusion_world_model_r2/epoch_016.pt \
  --split validation \
  --batch_size 2 \
  --num_workers 4 \
  --num_inference_steps 25 \
  --threshold 0.35 \
  --seed 7
```

## Demo

```bash
bash DriveWorld/scripts/demo_r2.sh
```

The demo exports:

```text
enterprise_diffusion_{index}_overview.png
enterprise_diffusion_{index}_rollout.mp4
enterprise_diffusion_{index}_metrics.json
```

## Notes

This repository contains code only. Processed WOMD shards, checkpoints and demo videos should be stored externally or attached as release artifacts.
