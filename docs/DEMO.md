# Demo

DriveWorld exports PNG overview images and MP4 rollout videos.

## Run

```bash
python src/_demo.py --config configs/demo.yaml
```

Override sample indices:

```bash
python src/_demo.py --config configs/demo.yaml --indices 10,50,100
```

## Outputs

```text
outputs/demo/demo_<index>.png
outputs/demo/demo_<index>.mp4
outputs/demo/demo_<index>.json
```

## Panel Meaning

The six-panel visualization contains:

```text
current scene          BEV history, map prior, current agents
GT occupancy           ground-truth future occupancy
pred occupancy         diffusion ensemble mean prediction
multi-sample futures   multiple diffusion modes / uncertainty
occupancy flow         predicted or derived motion direction
ego plan risk          lane-following candidates and risk heatmap
```

The planner samples lane-following candidate routes and scores them using predicted occupancy and uncertainty. Green routes are lower risk, red routes are higher risk.

## Release Assets

Use the export helper to collect representative artifacts:

```bash
python tools/export_release_demo.py --demo_dir outputs/demo --assets_dir assets
```

This copies representative PNG/MP4 files into `assets/` so the README can reference stable demo assets.
