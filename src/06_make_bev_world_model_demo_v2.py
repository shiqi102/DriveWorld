from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from bev_world_model import BevOccupancyFlowWorldModel
from womd_bev import BevShardDataset


RGB = Tuple[int, int, int]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--checkpoint", default="/mnt/data1/wzy/outputs/bev_world_model_r1_full/last.pt")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="/mnt/data1/wzy/outputs/bev_world_model_r1_full/demo_v2")
    parser.add_argument("--prefix", default="sample")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--keep_frames", action="store_true")
    return parser.parse_args()


def load_bev_meta(data_dir: str | Path, split: str) -> Dict[str, float]:
    meta_path = Path(data_dir) / f"{split}_metadata.json"
    if not meta_path.exists():
        return {"x_min": -80.0, "x_max": 80.0, "y_min": -80.0, "y_max": 80.0, "resolution": 1.0}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("bev", {})


def build_model(sample: Dict[str, torch.Tensor], checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu")
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
    return model


def norm_xy_to_pixel(xy: np.ndarray, meta: Dict[str, float], h: int, w: int) -> np.ndarray:
    x_max = float(meta.get("x_max", 80.0))
    y_max = float(meta.get("y_max", 80.0))
    x_min = float(meta.get("x_min", -80.0))
    y_min = float(meta.get("y_min", -80.0))
    res = float(meta.get("resolution", 1.0))
    x = xy[:, 0] * x_max
    y = xy[:, 1] * y_max
    rows = ((x - x_min) / res).astype(np.int32)
    cols = ((y - y_min) / res).astype(np.int32)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return np.stack([cols[valid], rows[valid]], axis=-1)


def draw_poly(canvas: np.ndarray, xy: np.ndarray, meta: Dict[str, float], color: RGB, thickness: int = 2):
    if xy.shape[0] < 2:
        return
    pts = norm_xy_to_pixel(xy, meta, canvas.shape[0], canvas.shape[1])
    if pts.shape[0] < 2:
        return
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_points(canvas: np.ndarray, xy: np.ndarray, meta: Dict[str, float], color: RGB, radius: int = 2):
    pts = norm_xy_to_pixel(xy, meta, canvas.shape[0], canvas.shape[1])
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def base_layer(sample: Dict[str, torch.Tensor]) -> np.ndarray:
    past = sample["past_bev"][-1].sum(0).clamp(0, 1).cpu().numpy()
    map_bev = sample["map_bev"].sum(0).clamp(0, 1).cpu().numpy()
    h, w = past.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[..., 1] = np.maximum(canvas[..., 1], (map_bev * 65).astype(np.uint8))
    canvas[..., 2] = np.maximum(canvas[..., 2], (map_bev * 85).astype(np.uint8))
    occ = (past * 245).astype(np.uint8)
    canvas[..., 0] = np.maximum(canvas[..., 0], occ)
    canvas[..., 1] = np.maximum(canvas[..., 1], occ)
    canvas[..., 2] = np.maximum(canvas[..., 2], occ)
    return canvas


def overlay_heat(canvas: np.ndarray, heat: np.ndarray, color: RGB, alpha: float = 0.85) -> np.ndarray:
    heat = np.clip(heat, 0.0, 1.0)
    out = canvas.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out = out * (1.0 - alpha * heat[..., None]) + color_arr * (alpha * heat[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_history(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], max_agents: int = 48):
    features = sample["agent_features"].cpu().numpy()
    mask = sample["agent_mask"].cpu().numpy() > 0.5
    count = 0
    for i, valid in enumerate(mask):
        if not valid or count >= max_agents:
            continue
        xy = features[i, :, :2]
        draw_poly(canvas, xy, meta, (120, 190, 255), 1)
        draw_points(canvas, xy[-1:], meta, (255, 255, 255), 1)
        count += 1


def draw_gt_traj(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], end_t: int, max_agents: int = 48):
    traj = sample["traj_target"].cpu().numpy()
    mask = sample["traj_mask"].cpu().numpy() > 0.5
    count = 0
    for i in range(traj.shape[0]):
        if count >= max_agents or not mask[i, : end_t + 1].any():
            continue
        xy = traj[i, : end_t + 1][mask[i, : end_t + 1]]
        draw_poly(canvas, xy, meta, (60, 255, 95), 2)
        draw_points(canvas, xy[-1:], meta, (190, 255, 190), 2)
        count += 1


def draw_pred_modes(
    canvas: np.ndarray,
    pred: Dict[str, torch.Tensor],
    sample: Dict[str, torch.Tensor],
    meta: Dict[str, float],
    end_t: int,
    max_agents: int = 24,
    top_k: int = 2,
):
    traj = pred["traj"][0].detach().cpu().numpy()
    logits = pred["traj_logits"][0].detach().cpu().numpy()
    agent_mask = sample["agent_mask"].cpu().numpy() > 0.5
    prob = np.exp(logits - logits.max(axis=-1, keepdims=True))
    prob = prob / np.maximum(prob.sum(axis=-1, keepdims=True), 1e-6)
    count = 0
    for agent_idx, valid in enumerate(agent_mask):
        if not valid or count >= max_agents:
            continue
        modes = np.argsort(-prob[agent_idx])[:top_k]
        for rank, mode_idx in enumerate(modes):
            color = (255, 175, 30) if rank == 0 else (255, 80, 210)
            draw_poly(canvas, traj[agent_idx, mode_idx, : end_t + 1], meta, color, 1)
        count += 1


def make_rule_ego_plans(sample: Dict[str, torch.Tensor], meta: Dict[str, float], future_count: int) -> np.ndarray:
    x_max = float(meta.get("x_max", 80.0))
    y_max = float(meta.get("y_max", 80.0))
    ego = sample.get("ego_future")
    ego_mask = sample.get("ego_future_mask")
    if ego is not None and ego_mask is not None and float(ego_mask.sum()) >= 2:
        ego_xy = ego.cpu().numpy()
        mask = ego_mask.cpu().numpy() > 0.5
        valid = ego_xy[mask]
        speed = float(np.linalg.norm(valid[-1] - valid[0]) / max(valid.shape[0] - 1, 1))
        speed = max(speed, 3.0 / x_max)
    else:
        speed = 6.0 / x_max
    ts = np.arange(1, future_count + 1, dtype=np.float32)
    curves = [-0.018, -0.010, 0.0, 0.010, 0.018]
    plans = []
    for curve in curves:
        x = speed * ts
        y = curve * (ts ** 1.45) * (x_max / y_max)
        plans.append(np.stack([x, y], axis=-1))
    stop = np.stack([np.minimum(speed * ts, 5.0 / x_max), np.zeros_like(ts)], axis=-1)
    plans.append(stop)
    return np.asarray(plans, dtype=np.float32)


def score_plan_risk(plan: np.ndarray, pred_prob: np.ndarray, meta: Dict[str, float]) -> float:
    h, w = pred_prob.shape[-2:]
    risks = []
    for t, xy in enumerate(plan):
        pts = norm_xy_to_pixel(xy[None], meta, h, w)
        if pts.shape[0] == 0:
            risks.append(0.2)
            continue
        col, row = int(pts[0, 0]), int(pts[0, 1])
        r0, r1 = max(row - 2, 0), min(row + 3, h)
        c0, c1 = max(col - 2, 0), min(col + 3, w)
        risks.append(float(pred_prob[min(t, pred_prob.shape[0] - 1), r0:r1, c0:c1].max()))
    return float(np.clip(max(risks), 0.0, 1.0))


def draw_ego(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], pred_prob_seq: np.ndarray, end_t: int):
    plans = make_rule_ego_plans(sample, meta, pred_prob_seq.shape[0])
    risks = np.asarray([score_plan_risk(plan, pred_prob_seq, meta) for plan in plans])
    for idx in np.argsort(risks):
        risk = float(risks[idx])
        color = (int(255 * risk), int(255 * (1 - risk)), 70)
        draw_poly(canvas, plans[idx, : end_t + 1], meta, color, 2)
    if "ego_future" in sample and "ego_future_mask" in sample:
        ego = sample["ego_future"].cpu().numpy()
        mask = sample["ego_future_mask"].cpu().numpy() > 0.5
        xy = ego[: end_t + 1][mask[: end_t + 1]]
        draw_poly(canvas, xy, meta, (255, 255, 255), 2)


def draw_flow(canvas: np.ndarray, flow: np.ndarray, occ_prob: np.ndarray, step: int = 12):
    h, w = canvas.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            if occ_prob[y, x] < 0.25:
                continue
            dy = float(flow[0, y, x])
            dx = float(flow[1, y, x])
            end = (int(np.clip(x + dx * 2.0, 0, w - 1)), int(np.clip(y + dy * 2.0, 0, h - 1)))
            cv2.arrowedLine(canvas, (x, y), end, (255, 230, 60), 1, cv2.LINE_AA, tipLength=0.35)


def add_title(img: np.ndarray, title: str, subtitle: str | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    bar_h = 30 if subtitle else 24
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = img
    cv2.putText(out, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (185, 185, 185), 1, cv2.LINE_AA)
    return out


def add_legend(img: np.ndarray) -> np.ndarray:
    legend = np.zeros((70, img.shape[1], 3), dtype=np.uint8)
    entries = [
        ("white: current boxes", (245, 245, 245)),
        ("cyan: map", (0, 120, 150)),
        ("blue: past tracks", (120, 190, 255)),
        ("green: GT future", (60, 255, 95)),
        ("orange: predicted occ / top mode", (255, 175, 30)),
        ("magenta: alternate modes", (255, 80, 210)),
        ("ego risk: green low, red high", (120, 220, 80)),
    ]
    x, y = 10, 20
    for text, color in entries:
        cv2.circle(legend, (x, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(legend, text, (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
        x += 220
        if x > img.shape[1] - 230:
            x, y = 10, y + 28
    return np.concatenate([img, legend], axis=0)


def sample_metrics(pred: Dict[str, torch.Tensor], sample: Dict[str, torch.Tensor], meta: Dict[str, float], threshold: float):
    with torch.no_grad():
        occ_pred = (torch.sigmoid(pred["occ_logits"][0]).cpu() > threshold).float()
        occ_true = sample["future_occ"].float()
        inter = (occ_pred * occ_true).sum()
        union = ((occ_pred + occ_true) > 0).float().sum().clamp_min(1.0)
        iou = float((inter / union).cpu())

        traj = pred["traj"][0].detach().cpu()
        target = sample["traj_target"][:, None]
        mask = sample["traj_mask"][:, None]
        dist_norm = ((traj - target) ** 2).sum(dim=-1).sqrt()
        ade = (dist_norm * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        min_ade = ade.min(dim=-1).values
        fde = dist_norm[..., -1].min(dim=-1).values
        valid = sample["traj_mask"].sum(dim=-1) > 0
        min_ade = float((min_ade * valid.float()).sum() / valid.float().sum().clamp_min(1.0))
        fde = float((fde * valid.float()).sum() / valid.float().sum().clamp_min(1.0))
    meter_scale = 0.5 * (float(meta.get("x_max", 80.0)) + float(meta.get("y_max", 80.0)))
    return iou, min_ade, fde, min_ade * meter_scale, fde * meter_scale


def make_panels(sample: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], meta: Dict[str, float], t: int, threshold: float):
    gt_occ = sample["future_occ"][t].sum(0).clamp(0, 1).cpu().numpy()
    pred_prob_seq = torch.sigmoid(pred["occ_logits"][0]).sum(1).clamp(0, 1).detach().cpu().numpy()
    pred_prob = pred_prob_seq[t]
    pred_flow = pred["flow"][0, t].detach().cpu().numpy()

    current = base_layer(sample)
    draw_history(current, sample, meta)

    gt = overlay_heat(base_layer(sample), gt_occ, (45, 245, 75), 0.85)
    draw_gt_traj(gt, sample, meta, t)

    pred_occ = overlay_heat(base_layer(sample), pred_prob, (255, 105, 35), 0.80)
    draw_flow(pred_occ, pred_flow, pred_prob)

    traj = base_layer(sample)
    draw_history(traj, sample, meta)
    draw_gt_traj(traj, sample, meta, t)
    draw_pred_modes(traj, pred, sample, meta, t)

    flow = overlay_heat(base_layer(sample), pred_prob, (200, 70, 255), 0.65)
    draw_flow(flow, pred_flow, pred_prob, step=8)

    ego = base_layer(sample)
    draw_ego(ego, sample, meta, pred_prob_seq, t)

    iou, ade_n, fde_n, ade_m, fde_m = sample_metrics(pred, sample, meta, threshold)
    panels = [
        add_title(current, "Current BEV"),
        add_title(gt, f"GT Future t={t}"),
        add_title(pred_occ, f"Pred Occ t={t}"),
        add_title(traj, "Traj Modes"),
        add_title(flow, "Flow Field"),
        add_title(ego, "Rule Ego Risk"),
    ]
    summary = f"IoU={iou:.3f}  minADE={ade_m:.2f}m ({ade_n:.3f} norm)  minFDE={fde_m:.2f}m ({fde_n:.3f} norm)"
    return panels, summary


def tile_panels(panels: list[np.ndarray], summary: str, scale: int) -> np.ndarray:
    resized = []
    for panel in panels:
        resized.append(cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    h = max(p.shape[0] for p in resized)
    w = max(p.shape[1] for p in resized)
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
    header = np.zeros((54, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "BEV Occupancy-Flow World Model Demo", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, summary, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    return add_legend(np.concatenate([header, body], axis=0))


def save_h264_video(frames: list[np.ndarray], out_path: Path, fps: int):
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
    subprocess.run(cmd, check=True)
    return frame_dir


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BevShardDataset(args.data_dir, args.split)
    sample = dataset[args.index]
    meta = load_bev_meta(args.data_dir, args.split)
    model = build_model(sample, args.checkpoint, device)

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

    frames = []
    future_count = int(sample["future_occ"].shape[0])
    for t in range(future_count):
        panels, summary = make_panels(sample, pred, meta, t, args.threshold)
        frames.append(tile_panels(panels, summary, args.scale))
    overview = frames[-1]
    overview_path = out_dir / f"{args.prefix}_{args.index}_overview_v3.png"
    video_path = out_dir / f"{args.prefix}_{args.index}_rollout_v3_h264.mp4"
    cv2.imwrite(str(overview_path), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    frame_dir = save_h264_video(frames, video_path, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    print(f"saved overview: {overview_path}")
    print(f"saved video: {video_path}")


if __name__ == "__main__":
    main()
