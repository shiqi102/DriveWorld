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

from bev_diffusion_world_model import (
    ConditionalBevDenoiser,
    DiffusionTargetConfig,
    build_ddim_scheduler,
    diffusion_loss,
    make_diffusion_target,
)
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--output_dir", default="/mnt/data1/wzy/outputs/bev_diffusion_world_model_r1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--sensor_dim", type=int, default=256)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--prediction_type", choices=["sample", "epsilon"], default="sample")
    parser.add_argument("--predict_flow", action="store_true")
    parser.add_argument("--flow_scale", type=float, default=20.0)
    parser.add_argument("--flow_loss_weight", type=float, default=0.25)
    parser.add_argument("--far_loss_weight", type=float, default=2.0)
    parser.add_argument("--occ_dilate_kernel", type=int, default=3)
    parser.add_argument("--enhanced_map_condition", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    visible_device_count = torch.cuda.device_count()
    if visible_device_count == 0:
        raise RuntimeError("Distributed training requested, but no CUDA devices are visible.")
    if local_rank >= visible_device_count:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        raise RuntimeError(
            "Invalid distributed GPU configuration: "
            f"LOCAL_RANK={local_rank}, but only {visible_device_count} CUDA device(s) are visible. "
            f"CUDA_VISIBLE_DEVICES={visible}. "
            "Set --nproc_per_node to the number of visible GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return True, rank, world, torch.device(f"cuda:{local_rank}")


def move_optimizer_state_to_device(optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main():
    args = parse_args()
    distributed, rank, world, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_set = BevShardDataset(args.data_dir, "training")
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ex = train_set[0]
    target_cfg = DiffusionTargetConfig(
        future_steps=int(ex["future_occ"].shape[0]),
        num_classes=int(ex["future_occ"].shape[1]),
        predict_flow=bool(args.predict_flow),
        flow_scale=float(args.flow_scale),
        occ_dilate_kernel=int(args.occ_dilate_kernel),
        flow_loss_weight=float(args.flow_loss_weight),
        far_loss_weight=float(args.far_loss_weight),
    )
    if rank == 0:
        print(f"distributed={distributed} world_size={world} device={device}", flush=True)
        print(f"num_training_samples={len(train_set)}", flush=True)
        print(f"target_cfg={target_cfg}", flush=True)
    model = ConditionalBevDenoiser(
        history_steps=int(ex["past_bev"].shape[0]),
        future_steps=target_cfg.future_steps,
        hidden_dim=args.hidden_dim,
        num_classes=target_cfg.num_classes,
        predict_flow=target_cfg.predict_flow,
        num_heads=args.num_heads,
        sensor_dim=args.sensor_dim,
        enhanced_map_condition=bool(args.enhanced_map_condition),
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    scheduler = build_ddim_scheduler(args.num_train_timesteps, prediction_type=args.prediction_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 1
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        target_model = model.module if distributed else model
        target_model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        if "scaler" in ckpt and args.amp and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler"])
        global_step = int(ckpt.get("global_step", 0))
        saved_epoch = int(ckpt.get("epoch", 0))
        start_epoch = saved_epoch + 1 if ckpt.get("epoch_completed", False) else max(saved_epoch, 1)
        if rank == 0:
            print(f"resumed from {args.resume}: epoch={saved_epoch}, start_epoch={start_epoch}, global_step={global_step}", flush=True)

    def save_checkpoint(epoch: int, epoch_completed: bool, filename: str):
        if rank != 0:
            return
        state = model.module.state_dict() if distributed else model.state_dict()
        payload = {
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "epoch_completed": epoch_completed,
            "global_step": global_step,
            "args": vars(args),
            "target_cfg": target_cfg.__dict__,
            "scheduler_type": "DDIMScheduler",
            "scheduler_config": dict(scheduler.config),
        }
        torch.save(payload, output_dir / filename)
        print(f"saved checkpoint: {output_dir / filename}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        iterator = tqdm(loader, desc=f"diffusion epoch {epoch}/{args.epochs}") if rank == 0 else loader
        for step, batch in enumerate(iterator):
            batch = {key: value.to(device, non_blocking=True).float() for key, value in batch.items()}
            clean = make_diffusion_target(batch, target_cfg)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy = scheduler.add_noise(clean, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                model_pred = model(
                    noisy,
                    timesteps,
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
                denoise_target = clean if args.prediction_type == "sample" else noise
                loss = diffusion_loss(model_pred, denoise_target, target_cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            running += float(loss.detach())
            if rank == 0 and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{running / (step + 1):.4f}")
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(epoch, False, "step_last.pt")

        if rank == 0 and epoch % args.save_every == 0:
            save_checkpoint(epoch, True, f"epoch_{epoch:03d}.pt")
            save_checkpoint(epoch, True, "last.pt")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
