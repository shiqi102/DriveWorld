from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from config_utils import parse_args_with_config
from model import ConditionalBevDenoiser, DiffusionTargetConfig, build_ddim_scheduler, sample_bev_diffusion
from womd import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/womd")
    parser.add_argument("--checkpoint", default="outputs/model_param.pt")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", default="")
    return parse_args_with_config(parser)


def setup_distributed():
    if "RANK" not in os.environ:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed eval requested, but CUDA is not available.")
    if local_rank >= torch.cuda.device_count():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA device(s) are visible. "
            f"CUDA_VISIBLE_DEVICES={visible}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return True, rank, world, torch.device(f"cuda:{local_rank}")


def build_model(sample, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt.get("args", {})
    cfg_dict = ckpt.get("target_cfg", {})
    target_cfg = DiffusionTargetConfig(
        future_steps=int(cfg_dict.get("future_steps", sample["future_occ"].shape[0])),
        num_classes=int(cfg_dict.get("num_classes", sample["future_occ"].shape[1])),
        predict_flow=bool(cfg_dict.get("predict_flow", args.get("predict_flow", False))),
        flow_scale=float(cfg_dict.get("flow_scale", args.get("flow_scale", 20.0))),
        occ_dilate_kernel=int(cfg_dict.get("occ_dilate_kernel", args.get("occ_dilate_kernel", 1))),
        flow_loss_weight=float(cfg_dict.get("flow_loss_weight", args.get("flow_loss_weight", 0.25))),
        far_loss_weight=float(cfg_dict.get("far_loss_weight", args.get("far_loss_weight", 1.0))),
    )
    model = ConditionalBevDenoiser(
        history_steps=int(sample["past_bev"].shape[0]),
        future_steps=target_cfg.future_steps,
        hidden_dim=int(args.get("hidden_dim", 192)),
        num_classes=target_cfg.num_classes,
        predict_flow=target_cfg.predict_flow,
        num_heads=int(args.get("num_heads", 8)),
        sensor_dim=int(args.get("sensor_dim", 256)),
        enhanced_map_condition=bool(args.get("enhanced_map_condition", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scheduler = build_ddim_scheduler(int(args.get("num_train_timesteps", 1000)))
    if "scheduler_config" in ckpt:
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler.from_config(ckpt["scheduler_config"])
    return model, target_cfg, scheduler


def main():
    args = parse_args()
    distributed, rank, world, device = setup_distributed()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = BevShardDataset(args.data_dir, args.split)
    eval_size = len(dataset)
    if args.max_batches:
        eval_size = min(eval_size, int(args.max_batches) * int(args.batch_size))
    eval_indices = list(range(eval_size))
    if distributed:
        eval_indices = eval_indices[rank::world]
    eval_dataset = Subset(dataset, eval_indices)
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model, target_cfg, scheduler = build_model(dataset[0], args.checkpoint, device)
    sums = {
        "occ_iou": 0.0,
        "occ_iou_near": 0.0,
        "occ_iou_mid": 0.0,
        "occ_iou_far": 0.0,
        "pred_pos_ratio": 0.0,
        "true_pos_ratio": 0.0,
        "pred_prob_mean": 0.0,
    }
    count = 0
    sample_count = 0
    global_total_batches = math.ceil(eval_size / max(int(args.batch_size), 1))
    with torch.no_grad():
        iterator = tqdm(loader, desc="diffusion eval", total=len(loader), disable=rank != 0)
        for batch in iterator:
            batch = {key: value.to(device).float() for key, value in batch.items()}
            sample_count += int(batch["past_bev"].shape[0])
            pred = sample_bev_diffusion(model, scheduler, batch, target_cfg, args.num_inference_steps)
            occ_pred = (pred["occ_probs"] > args.threshold).float()
            occ_true = batch["future_occ"]
            sums["pred_pos_ratio"] += float(occ_pred.mean())
            sums["true_pos_ratio"] += float(occ_true.mean())
            sums["pred_prob_mean"] += float(pred["occ_probs"].mean())
            inter = (occ_pred * occ_true).sum()
            union = ((occ_pred + occ_true) > 0).float().sum().clamp_min(1.0)
            sums["occ_iou"] += float(inter / union)
            horizon = []
            for idx in range(occ_true.shape[1]):
                p = occ_pred[:, idx]
                t = occ_true[:, idx]
                horizon.append((p * t).sum() / ((p + t) > 0).float().sum().clamp_min(1.0))
            horizon = torch.stack(horizon)
            third = max(1, len(horizon) // 3)
            sums["occ_iou_near"] += float(horizon[:third].mean())
            sums["occ_iou_mid"] += float(horizon[third : 2 * third].mean())
            sums["occ_iou_far"] += float(horizon[2 * third :].mean())
            count += 1

    if distributed:
        values = torch.tensor(
            [
                sums["occ_iou"],
                sums["occ_iou_near"],
                sums["occ_iou_mid"],
                sums["occ_iou_far"],
                sums["pred_pos_ratio"],
                sums["true_pos_ratio"],
                sums["pred_prob_mean"],
                float(count),
                float(sample_count),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        keys = [
            "occ_iou",
            "occ_iou_near",
            "occ_iou_mid",
            "occ_iou_far",
            "pred_pos_ratio",
            "true_pos_ratio",
            "pred_prob_mean",
        ]
        for idx, key in enumerate(keys):
            sums[key] = float(values[idx].item())
        count = int(values[7].item())
        sample_count = int(values[8].item())

    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["evaluated_batches"] = count
    metrics["evaluated_samples"] = sample_count
    metrics["total_batches"] = global_total_batches
    metrics["max_batches"] = int(args.max_batches)
    metrics["seed"] = int(args.seed)
    metrics["distributed_world_size"] = int(world)
    if args.max_batches and rank == 0:
        print(f"stopped after --max_batches={args.max_batches}; evaluated {count}/{global_total_batches} batches.")
    if rank == 0:
        print(metrics)
    if args.output_json and rank == 0:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"saved metrics: {out}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
