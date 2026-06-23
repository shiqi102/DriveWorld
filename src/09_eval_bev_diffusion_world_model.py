from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bev_diffusion_world_model import ConditionalBevDenoiser, DiffusionTargetConfig, build_ddim_scheduler, sample_bev_diffusion
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BevShardDataset(args.data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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
    effective_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="diffusion eval", total=effective_batches)):
            if args.max_batches and batch_idx >= args.max_batches:
                break
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
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["evaluated_batches"] = count
    metrics["evaluated_samples"] = sample_count
    metrics["total_batches"] = len(loader)
    metrics["max_batches"] = int(args.max_batches)
    metrics["seed"] = int(args.seed)
    if args.max_batches:
        print(f"stopped after --max_batches={args.max_batches}; evaluated {count}/{len(loader)} batches.")
    print(metrics)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"saved metrics: {out}")


if __name__ == "__main__":
    main()
