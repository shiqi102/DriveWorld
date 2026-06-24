# Data

DriveWorld uses the Waymo Open Motion Dataset (WOMD) scenario TFRecords and converts them into compact BEV shard files.

## Raw Data

Download the Motion Dataset scenario files from the official Waymo Open Dataset download page:

https://waymo.com/open/download

Place the raw files under:

```text
data/raw/womd/
  training/
    *.tfrecord*
  validation/
    *.tfrecord*
  testing/
    *.tfrecord*
```

Only `training` and `validation` are required for the default training and evaluation workflow.

## Preprocessing

Convert raw WOMD scenarios into BEV tensors:

```bash
python scripts/prepare_womd_scenarios.py --config configs/prepare.yaml
```

Process a single split:

```bash
python scripts/prepare_womd_scenarios.py --config configs/prepare.yaml --split validation
```

Run a small smoke test:

```bash
python scripts/prepare_womd_scenarios.py --config configs/prepare.yaml --split training --max_files 2
```

## Processed Format

The processed directory should look like:

```text
data/womd/
  training_00000.pt
  training_00001.pt
  training_metadata.json
  validation_00000.pt
  validation_metadata.json
```

Each shard stores batched tensors such as:

```text
past_bev
map_bev
map_vectors
traffic_lights
agent_features
future_occ
future_flow
traj_target
ego_future
```

The default model uses `past_bev`, `map_bev`, `map_vectors`, `traffic_lights`, and `agent_features` as conditions, and predicts `future_occ` plus optional `future_flow`.
