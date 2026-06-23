from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from bev_diffusion_world_model import ConditionalBevDenoiser, DiffusionTargetConfig, build_ddim_scheduler, sample_bev_diffusion
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="/mnt/data1/wzy/outputs/bev_diffusion_world_model_r1/sample.png")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--vis_erode_pred", type=int, default=0)
    parser.add_argument("--vis_threshold", type=float, default=0.12)
    return parser.parse_args()


def refine_pred_for_display(heat: torch.Tensor, erode_iters: int = 1, threshold: float = 0.12) -> torch.Tensor:
    """Sharpen only the rendered occupancy; model outputs and metrics stay unchanged."""
    heat = heat.clamp(0.0, 1.0).float()
    if float(heat.max()) <= 0.0:
        return heat
    mask = (heat >= float(threshold)).float()
    if erode_iters <= 0:
        return heat * mask
    flat = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    for _ in range(int(erode_iters)):
        flat = 1.0 - F.max_pool2d(1.0 - flat, kernel_size=2, stride=1, padding=0)
        flat = F.pad(flat, (0, 1, 0, 1))
        flat = flat[..., : mask.shape[-2], : mask.shape[-1]]
    return heat * flat.reshape_as(mask)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BevShardDataset(args.data_dir, args.split)
    sample = dataset[args.index]
    model, target_cfg, scheduler = build_model(sample, args.checkpoint, device)
    batch = {key: value.unsqueeze(0).to(device).float() for key, value in sample.items()}
    pred = sample_bev_diffusion(model, scheduler, batch, target_cfg, args.num_inference_steps)
    pred_stats = pred["occ_probs"]
    print(
        "occ_probs stats:",
        f"min={float(pred_stats.min()):.4f}",
        f"max={float(pred_stats.max()):.4f}",
        f"mean={float(pred_stats.mean()):.4f}",
        f"pos@0.35={float((pred_stats > 0.35).float().mean()):.6f}",
    )

    past = sample["past_bev"][-1].sum(0).clamp(0, 1).cpu()
    gt = sample["future_occ"].sum(1).clamp(0, 1).cpu()
    pred_occ_raw = pred["occ_probs"][0].sum(1).clamp(0, 1).cpu()
    pred_occ = refine_pred_for_display(pred_occ_raw, args.vis_erode_pred, args.vis_threshold)
    frames = [0, min(3, gt.shape[0] - 1), min(7, gt.shape[0] - 1), gt.shape[0] - 1]

    fig, axes = plt.subplots(3, len(frames), figsize=(4 * len(frames), 10))
    for col, idx in enumerate(frames):
        axes[0, col].imshow(past, cmap="gray")
        axes[0, col].set_title("current BEV")
        axes[1, col].imshow(gt[idx], cmap="magma")
        axes[1, col].set_title(f"GT future {idx}")
        axes[2, col].imshow(pred_occ[idx], cmap="magma")
        axes[2, col].set_title(f"diffusion sample {idx}")
        for row in range(3):
            axes[row, col].axis("off")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
