# Results

The current released checkpoint is:

```text
outputs/model_param.pt
```

It corresponds to the previous best checkpoint `epoch_016.pt`.

## Evaluation

Run evaluation:

```bash
python src/eval.py --config configs/eval.yaml
```

The latest model version reached approximately:

| Metric | Value |
| --- | ---: |
| occ_iou | 0.6169 |
| occ_iou_near | 0.8670 |
| occ_iou_mid | 0.6403 |
| occ_iou_far | 0.4772 |
| pred_pos_ratio | 0.00487 |
| true_pos_ratio | 0.00498 |

These numbers are project validation results, not official SOTA benchmark claims.

## Qualitative Checks

A useful demo should show:

- predicted occupancy is not all black and not random noise;
- nearby occupancy quality is better than far-horizon occupancy;
- multi-sample diffusion futures have plausible diversity;
- lane-following ego candidates remain on map/lane priors;
- high-risk ego candidates overlap predicted future occupancy or uncertainty.

## Known Limitations

- The default pipeline uses track/map-derived WOMD BEV tensors, not raw camera/LiDAR features.
- Long-horizon occupancy prediction remains harder than near-horizon prediction.
- The planner is a demo-level lane-graph risk scorer, not a full vehicle planner.
- Checkpoints and processed datasets are intentionally not stored in Git.
