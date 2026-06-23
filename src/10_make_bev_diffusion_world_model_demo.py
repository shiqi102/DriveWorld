from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from bev_diffusion_world_model import (
    ConditionalBevDenoiser,
    DiffusionTargetConfig,
    build_ddim_scheduler,
    sample_bev_diffusion,
)
from womd_bev import BevShardDataset


RGB = Tuple[int, int, int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create portfolio-grade PNG/MP4 visualizations for the conditional BEV diffusion world model."
    )
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="/mnt/data1/wzy/outputs/bev_diffusion_world_model_r1/demo")
    parser.add_argument("--prefix", default="diffusion")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep_frames", action="store_true")
    return parser.parse_args()


def load_bev_meta(data_dir: str | Path, split: str) -> Dict[str, float]:
    meta_path = Path(data_dir) / f"{split}_metadata.json"
    if not meta_path.exists():
        return {"x_min": -80.0, "x_max": 80.0, "y_min": -80.0, "y_max": 80.0, "resolution": 1.0}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("bev", {})


def load_model(sample: Dict[str, torch.Tensor], checkpoint: str | Path, device: torch.device):
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
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    scheduler = build_ddim_scheduler(int(args.get("num_train_timesteps", 1000)))
    if "scheduler_config" in ckpt:
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler.from_config(ckpt["scheduler_config"])
    return model, target_cfg, scheduler


def norm_xy_to_pixel(xy: np.ndarray, meta: Dict[str, float], h: int, w: int) -> np.ndarray:
    x_min = float(meta.get("x_min", -80.0))
    x_max = float(meta.get("x_max", 80.0))
    y_min = float(meta.get("y_min", -80.0))
    y_max = float(meta.get("y_max", 80.0))
    res = float(meta.get("resolution", 1.0))
    x = xy[:, 0] * x_max
    y = xy[:, 1] * y_max
    rows = ((x - x_min) / res).astype(np.int32)
    cols = ((y - y_min) / res).astype(np.int32)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return np.stack([cols[valid], rows[valid]], axis=-1)


def draw_poly(canvas: np.ndarray, xy: np.ndarray, meta: Dict[str, float], color: RGB, thickness: int = 1):
    if xy.shape[0] < 2:
        return
    pts = norm_xy_to_pixel(xy, meta, canvas.shape[0], canvas.shape[1])
    if pts.shape[0] >= 2:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_points(canvas: np.ndarray, xy: np.ndarray, meta: Dict[str, float], color: RGB, radius: int = 2):
    pts = norm_xy_to_pixel(xy, meta, canvas.shape[0], canvas.shape[1])
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def base_layer(sample: Dict[str, torch.Tensor]) -> np.ndarray:
    past = sample["past_bev"][-1].sum(0).clamp(0, 1).cpu().numpy()
    map_bev = sample["map_bev"].sum(0).clamp(0, 1).cpu().numpy()
    h, w = past.shape
    canvas = np.full((h, w, 3), 8, dtype=np.uint8)
    canvas[..., 1] = np.maximum(canvas[..., 1], (map_bev * 85).astype(np.uint8))
    canvas[..., 2] = np.maximum(canvas[..., 2], (map_bev * 105).astype(np.uint8))
    occ = (past * 245).astype(np.uint8)
    canvas = np.maximum(canvas, occ[..., None])
    return canvas


def overlay_heat(canvas: np.ndarray, heat: np.ndarray, color: RGB, alpha: float = 0.82) -> np.ndarray:
    heat = np.clip(heat, 0.0, 1.0)
    out = canvas.astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    out = out * (1.0 - alpha * heat[..., None]) + color_arr * (alpha * heat[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_history(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], max_agents: int = 48):
    features = sample["agent_features"].cpu().numpy()
    mask = sample["agent_mask"].cpu().numpy() > 0.5
    drawn = 0
    for i, valid in enumerate(mask):
        if not valid or drawn >= max_agents:
            continue
        xy = features[i, :, :2]
        draw_poly(canvas, xy, meta, (110, 185, 255), 1)
        draw_points(canvas, xy[-1:], meta, (245, 245, 245), 1)
        drawn += 1


def draw_gt_traj(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], end_t: int, max_agents: int = 48):
    if "traj_target" not in sample or "traj_mask" not in sample:
        return
    traj = sample["traj_target"].cpu().numpy()
    mask = sample["traj_mask"].cpu().numpy() > 0.5
    drawn = 0
    for i in range(traj.shape[0]):
        if drawn >= max_agents or not mask[i, : end_t + 1].any():
            continue
        xy = traj[i, : end_t + 1][mask[i, : end_t + 1]]
        draw_poly(canvas, xy, meta, (70, 255, 110), 2)
        draw_points(canvas, xy[-1:], meta, (200, 255, 205), 2)
        drawn += 1


def draw_flow(canvas: np.ndarray, flow: np.ndarray | None, occ_prob: np.ndarray, step: int = 10):
    if flow is None:
        return
    h, w = canvas.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            if occ_prob[y, x] < 0.25:
                continue
            dy = float(flow[0, y, x])
            dx = float(flow[1, y, x])
            end = (int(np.clip(x + dx * 2.0, 0, w - 1)), int(np.clip(y + dy * 2.0, 0, h - 1)))
            cv2.arrowedLine(canvas, (x, y), end, (255, 225, 70), 1, cv2.LINE_AA, tipLength=0.32)


def add_title(img: np.ndarray, title: str, subtitle: str | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    bar_h = 28 if subtitle else 22
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = img
    cv2.putText(out, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (248, 248, 248), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (188, 188, 188), 1, cv2.LINE_AA)
    return out


def add_legend(img: np.ndarray) -> np.ndarray:
    legend = np.zeros((58, img.shape[1], 3), dtype=np.uint8)
    entries = [
        ("white: current occupancy", (245, 245, 245)),
        ("cyan: map prior", (0, 135, 160)),
        ("blue: actor history", (110, 185, 255)),
        ("green: GT future", (70, 255, 110)),
        ("orange: diffusion sample", (255, 125, 35)),
        ("purple: ensemble mean", (200, 95, 255)),
        ("red: sample uncertainty", (255, 70, 85)),
    ]
    x, y = 12, 20
    for text, color in entries:
        cv2.circle(legend, (x, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(legend, text, (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
        x += 230
        if x > img.shape[1] - 230:
            x, y = 12, y + 25
    return np.concatenate([img, legend], axis=0)


def tile_panels(panels: List[np.ndarray], header_title: str, summary: str, scale: int) -> np.ndarray:
    resized = [cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for panel in panels]
    h = max(panel.shape[0] for panel in resized)
    w = max(panel.shape[1] for panel in resized)
    padded = []
    for panel in resized:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(out)
    gap = 12
    sep_v = np.full((h, gap, 3), 245, dtype=np.uint8)
    row1 = np.concatenate([padded[0], sep_v, padded[1], sep_v, padded[2]], axis=1)
    row2 = np.concatenate([padded[3], sep_v, padded[4], sep_v, padded[5]], axis=1)
    sep_h = np.full((gap, row1.shape[1], 3), 245, dtype=np.uint8)
    body = np.concatenate([row1, sep_h, row2], axis=0)
    header = np.zeros((58, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, header_title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, summary, (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
    return add_legend(np.concatenate([header, body], axis=0))


def save_h264_video(frames: List[np.ndarray], out_path: Path, fps: int) -> Path:
    frame_dir = out_path.parent / f".{out_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{idx:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frames[0].shape[1], frames[0].shape[0]),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    return frame_dir


def run_diffusion_ensemble(
    model: ConditionalBevDenoiser,
    scheduler,
    batch: Dict[str, torch.Tensor],
    target_cfg: DiffusionTargetConfig,
    num_samples: int,
    num_inference_steps: int,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    outputs = []
    for sample_idx in range(num_samples):
        torch.manual_seed(seed + sample_idx)
        if batch["past_bev"].device.type == "cuda":
            torch.cuda.manual_seed_all(seed + sample_idx)
        outputs.append(sample_bev_diffusion(model, scheduler, batch, target_cfg, num_inference_steps))
    return outputs


def summarize_metrics(
    outputs: List[Dict[str, torch.Tensor]],
    sample: Dict[str, torch.Tensor],
    threshold: float,
) -> Dict[str, float]:
    pred_stack = torch.stack([out["occ_probs"][0] for out in outputs])
    mean_occ = pred_stack.mean(0)
    pred_bin = (mean_occ > threshold).float()
    true_occ = sample["future_occ"].to(pred_stack.device).float()
    inter = (pred_bin * true_occ).sum()
    union = ((pred_bin + true_occ) > 0).float().sum().clamp_min(1.0)
    return {
        "ensemble_size": float(len(outputs)),
        "occupancy_iou": float((inter / union).cpu()),
        "pred_positive_ratio": float(pred_bin.mean().cpu()),
        "gt_positive_ratio": float(true_occ.mean().cpu()),
        "mean_probability": float(mean_occ.mean().cpu()),
        "mean_uncertainty": float(pred_stack.std(0, unbiased=False).mean().cpu()),
    }


def make_panels(
    sample: Dict[str, torch.Tensor],
    outputs: List[Dict[str, torch.Tensor]],
    meta: Dict[str, float],
    t: int,
) -> List[np.ndarray]:
    pred_stack = torch.stack([out["occ_probs"][0] for out in outputs])
    sample_a = pred_stack[0, t].sum(0).clamp(0, 1).cpu().numpy()
    sample_b = pred_stack[min(1, len(outputs) - 1), t].sum(0).clamp(0, 1).cpu().numpy()
    mean_occ = pred_stack[:, t].mean(0).sum(0).clamp(0, 1).cpu().numpy()
    uncertainty = pred_stack[:, t].sum(1).std(0, unbiased=False).clamp(0, 1).cpu().numpy()
    gt_occ = sample["future_occ"][t].sum(0).clamp(0, 1).cpu().numpy()
    flow = outputs[0].get("flow")
    flow_np = None if flow is None else flow[0, t].detach().cpu().numpy()

    current = base_layer(sample)
    draw_history(current, sample, meta)

    gt = overlay_heat(base_layer(sample), gt_occ, (70, 255, 110), 0.86)
    draw_gt_traj(gt, sample, meta, t)

    draw_a = overlay_heat(base_layer(sample), sample_a, (255, 125, 35), 0.82)
    draw_history(draw_a, sample, meta)

    draw_b = overlay_heat(base_layer(sample), sample_b, (255, 185, 45), 0.78)
    draw_history(draw_b, sample, meta)

    mean = overlay_heat(base_layer(sample), mean_occ, (200, 95, 255), 0.78)
    draw_flow(mean, flow_np, mean_occ)

    uncert = overlay_heat(base_layer(sample), uncertainty, (255, 70, 85), 0.88)
    draw_flow(uncert, flow_np, mean_occ, step=12)

    return [
        add_title(current, "Condition", "current BEV + history"),
        add_title(gt, f"GT Future t={t}"),
        add_title(draw_a, f"Sample A t={t}"),
        add_title(draw_b, f"Sample B t={t}"),
        add_title(mean, "Ensemble Mean", "probability + flow"),
        add_title(uncert, "Uncertainty", "std across samples"),
    ]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BevShardDataset(args.data_dir, args.split)
    sample = dataset[args.index]
    meta = load_bev_meta(args.data_dir, args.split)
    model, target_cfg, scheduler = load_model(sample, args.checkpoint, device)
    batch = {key: value.unsqueeze(0).to(device).float() for key, value in sample.items()}

    with torch.no_grad():
        outputs = run_diffusion_ensemble(
            model,
            scheduler,
            batch,
            target_cfg,
            max(1, args.num_samples),
            args.num_inference_steps,
            args.seed,
        )

    metrics = summarize_metrics(outputs, sample, args.threshold)
    future_count = int(sample["future_occ"].shape[0])
    summary = (
        f"IoU={metrics['occupancy_iou']:.3f}  "
        f"meanP={metrics['mean_probability']:.3f}  "
        f"uncertainty={metrics['mean_uncertainty']:.3f}  "
        f"samples={int(metrics['ensemble_size'])}"
    )

    frames = []
    title = "Conditional BEV Diffusion World Model Demo"
    for t in range(future_count):
        panels = make_panels(sample, outputs, meta, t)
        frames.append(tile_panels(panels, title, summary, args.scale))

    overview_path = out_dir / f"{args.prefix}_{args.index}_overview.png"
    video_path = out_dir / f"{args.prefix}_{args.index}_rollout.mp4"
    metrics_path = out_dir / f"{args.prefix}_{args.index}_metrics.json"

    overview_idx = min(future_count - 1, max(0, future_count // 2))
    cv2.imwrite(str(overview_path), cv2.cvtColor(frames[overview_idx], cv2.COLOR_RGB2BGR))
    frame_dir = save_h264_video(frames, video_path, args.fps)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.keep_frames:
        shutil.rmtree(frame_dir)

    print(f"saved overview: {overview_path}")
    print(f"saved video: {video_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
