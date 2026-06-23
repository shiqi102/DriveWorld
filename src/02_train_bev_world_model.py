from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bev_world_model import BevOccupancyFlowWorldModel, compute_losses, metrics
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="WorldModel_NuScenes_HF/data/processed/womd_bev")
    parser.add_argument("--output_dir", default="WorldModel_NuScenes_HF/outputs/bev_world_model")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument("--sensor_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return True, rank, world, torch.device(f"cuda:{local_rank}")


def main():
    args = parse_args()
    distributed, rank, world, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_set = BevShardDataset(args.data_dir, "training")
    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    example = train_set[0]
    history_steps = int(example["past_bev"].shape[0])
    future_steps = int(example["future_occ"].shape[0])
    model = BevOccupancyFlowWorldModel(
        history_steps=history_steps,
        future_steps=future_steps,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        traj_agents=int(example["agent_features"].shape[0]),
        num_modes=args.num_modes,
        sensor_dim=args.sensor_dim,
    ).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        iterator = train_loader
        if rank == 0:
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(iterator):
            batch = {key: value.to(device, non_blocking=True).float() for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
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
                losses = compute_losses(pred, batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(losses["loss"].detach())
            if rank == 0 and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{running / (step + 1):.4f}",
                    occ=f"{float(losses['occ_loss']):.3f}",
                    traj=f"{float(losses['traj_loss']):.3f}",
                )

        if rank == 0 and epoch % args.save_every == 0:
            state = model.module.state_dict() if distributed else model.state_dict()
            torch.save({"model": state, "epoch": epoch, "args": vars(args)}, output_dir / f"epoch_{epoch:03d}.pt")
            torch.save({"model": state, "epoch": epoch, "args": vars(args)}, output_dir / "last.pt")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
