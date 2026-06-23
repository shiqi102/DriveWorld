from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bev_world_model import BevOccupancyFlowWorldModel, metrics
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="WorldModel_NuScenes_HF/data/processed/womd_bev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BevShardDataset(args.data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ex = dataset[0]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model = BevOccupancyFlowWorldModel(
        history_steps=int(ex["past_bev"].shape[0]),
        future_steps=int(ex["future_occ"].shape[0]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        depth=int(ckpt_args.get("depth", 8)),
        num_heads=int(ckpt_args.get("num_heads", 8)),
        num_modes=int(ckpt_args.get("num_modes", 6)),
        sensor_dim=int(ckpt_args.get("sensor_dim", 256)),
        traj_agents=int(ex["agent_features"].shape[0]),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    sums = {
        "occ_iou": 0.0,
        "occ_iou_near": 0.0,
        "occ_iou_mid": 0.0,
        "occ_iou_far": 0.0,
        "min_ade_norm": 0.0,
        "min_fde_norm": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = {key: value.to(device).float() for key, value in batch.items()}
            pred = model(
                batch["past_bev"],
                batch["map_bev"],
                batch["agent_features"],
                batch["agent_mask"],
                map_vectors=batch.get("map_vectors"),
                map_vector_mask=batch.get("map_vector_mask"),
                traffic_lights=batch.get("traffic_lights"),
                traffic_light_mask=batch.get("traffic_light_mask"),
                sensor_context=batch.get("sensor_context"),
                camera_images=batch.get("camera_images"),
                lidar_points=batch.get("lidar_points"),
                lidar_mask=batch.get("lidar_mask"),
            )
            item = metrics(pred, batch)
            for key in sums:
                sums[key] += float(item[key])
            count += 1
    print({key: value / max(count, 1) for key, value in sums.items()})


if __name__ == "__main__":
    main()
