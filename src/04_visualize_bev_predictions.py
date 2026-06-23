from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from bev_world_model import BevOccupancyFlowWorldModel
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="WorldModel_NuScenes_HF/data/processed/womd_bev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="WorldModel_NuScenes_HF/outputs/bev_world_model/preview.png")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BevShardDataset(args.data_dir, args.split)
    sample = dataset[args.index]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model = BevOccupancyFlowWorldModel(
        history_steps=int(sample["past_bev"].shape[0]),
        future_steps=int(sample["future_occ"].shape[0]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        depth=int(ckpt_args.get("depth", 8)),
        num_heads=int(ckpt_args.get("num_heads", 8)),
        num_modes=int(ckpt_args.get("num_modes", 6)),
        sensor_dim=int(ckpt_args.get("sensor_dim", 256)),
        traj_agents=int(sample["agent_features"].shape[0]),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    batch = {key: value.unsqueeze(0).to(device).float() for key, value in sample.items()}
    with torch.no_grad():
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

    past = sample["past_bev"][-1].sum(0).clamp(0, 1).cpu()
    future_true = sample["future_occ"].sum(1).clamp(0, 1).cpu()
    future_pred = torch.sigmoid(pred["occ_logits"][0]).sum(1).clamp(0, 1).cpu()
    frames = [0, min(3, future_true.shape[0] - 1), min(7, future_true.shape[0] - 1), future_true.shape[0] - 1]

    fig, axes = plt.subplots(3, len(frames), figsize=(4 * len(frames), 10))
    for col, idx in enumerate(frames):
        axes[0, col].imshow(past, cmap="gray")
        axes[0, col].set_title("current occupancy")
        axes[1, col].imshow(future_true[idx], cmap="magma")
        axes[1, col].set_title(f"future gt {idx}")
        axes[2, col].imshow(future_pred[idx], cmap="magma")
        axes[2, col].set_title(f"future pred {idx}")
        for row in range(3):
            axes[row, col].axis("off")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
