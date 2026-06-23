from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from bev_world_model import BevOccupancyFlowWorldModel
from womd_bev import BevShardDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--checkpoint", default="/mnt/data1/wzy/outputs/bev_world_model_r1_full/last.pt")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="/mnt/data1/wzy/outputs/bev_world_model_r1_full/demo")
    parser.add_argument("--prefix", default="demo")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--prob_threshold", type=float, default=0.35)
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


def metric_to_pixel_xy(x_norm: np.ndarray, y_norm: np.ndarray, meta: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    x_min = float(meta.get("x_min", -80.0))
    x_max = float(meta.get("x_max", 80.0))
    y_min = float(meta.get("y_min", -80.0))
    y_max = float(meta.get("y_max", 80.0))
    res = float(meta.get("resolution", 1.0))
    x = x_norm * x_max
    y = y_norm * y_max
    rows = ((x - x_min) / res).astype(np.int32)
    cols = ((y - y_min) / res).astype(np.int32)
    return cols, rows


def valid_points(cols: np.ndarray, rows: np.ndarray, h: int, w: int):
    mask = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return cols[mask], rows[mask], mask


def base_canvas(sample: Dict[str, torch.Tensor]) -> np.ndarray:
    past = sample["past_bev"][-1].sum(0).clamp(0, 1).cpu().numpy()
    map_bev = sample["map_bev"].sum(0).clamp(0, 1).cpu().numpy()
    h, w = past.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas[..., 1] += map_bev * 0.18
    canvas[..., 2] += map_bev * 0.18
    canvas[..., 0] += past * 0.90
    canvas[..., 1] += past * 0.90
    canvas[..., 2] += past * 0.90
    return np.clip(canvas, 0.0, 1.0)


def overlay_occ(canvas: np.ndarray, occ: np.ndarray, color: Tuple[float, float, float], alpha: float):
    occ = np.clip(occ, 0.0, 1.0)
    colored = np.zeros_like(canvas)
    colored[..., 0] = color[0] * occ
    colored[..., 1] = color[1] * occ
    colored[..., 2] = color[2] * occ
    return np.clip(canvas * (1.0 - alpha * occ[..., None]) + colored * alpha, 0.0, 1.0)


def draw_polyline(
    canvas: np.ndarray,
    xy_norm: np.ndarray,
    meta: Dict[str, float],
    color: Tuple[int, int, int],
    thickness: int = 1,
):
    if xy_norm.shape[0] < 2:
        return
    h, w = canvas.shape[:2]
    cols, rows = metric_to_pixel_xy(xy_norm[:, 0], xy_norm[:, 1], meta)
    cols, rows, mask = valid_points(cols, rows, h, w)
    if mask.sum() < 2:
        return
    pts = np.stack([cols, rows], axis=-1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_agent_tracks(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], max_agents: int = 24):
    features = sample["agent_features"].cpu().numpy()
    agent_mask = sample["agent_mask"].cpu().numpy() > 0.5
    drawn = 0
    for agent_idx, valid_agent in enumerate(agent_mask):
        if not valid_agent or drawn >= max_agents:
            continue
        hist_xy = features[agent_idx, :, :2]
        draw_polyline(canvas, hist_xy, meta, (120, 180, 255), thickness=1)
        drawn += 1


def draw_gt_trajs(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], t: int, max_agents: int = 24):
    targets = sample["traj_target"].cpu().numpy()
    masks = sample["traj_mask"].cpu().numpy() > 0.5
    drawn = 0
    for agent_idx in range(targets.shape[0]):
        if drawn >= max_agents or not masks[agent_idx, : t + 1].any():
            continue
        xy = targets[agent_idx, : t + 1][masks[agent_idx, : t + 1]]
        draw_polyline(canvas, xy, meta, (70, 255, 90), thickness=1)
        drawn += 1


def draw_pred_trajs(
    canvas: np.ndarray,
    pred: Dict[str, torch.Tensor],
    sample: Dict[str, torch.Tensor],
    meta: Dict[str, float],
    t: int,
    max_agents: int = 16,
):
    traj = pred["traj"][0].detach().cpu().numpy()
    logits = pred["traj_logits"][0].detach().cpu().numpy()
    agent_mask = sample["agent_mask"].cpu().numpy() > 0.5
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-6)
    drawn = 0
    for agent_idx, valid_agent in enumerate(agent_mask):
        if not valid_agent or drawn >= max_agents:
            continue
        best_modes = np.argsort(-probs[agent_idx])[:2]
        for mode_idx in best_modes:
            confidence = float(probs[agent_idx, mode_idx])
            color = (255, int(120 + 100 * confidence), 30)
            draw_polyline(canvas, traj[agent_idx, mode_idx, : t + 1], meta, color, thickness=1)
        drawn += 1


def draw_ego_plans(canvas: np.ndarray, pred: Dict[str, torch.Tensor], meta: Dict[str, float], t: int):
    plans = pred["ego_plan"][0].detach().cpu().numpy()
    risk = pred["ego_risk"][0].detach().cpu().numpy()
    risk_prob = 1.0 / (1.0 + np.exp(-risk))
    order = np.argsort(risk_prob)
    for mode_idx in order:
        r = float(risk_prob[mode_idx])
        color = (int(255 * r), int(255 * (1.0 - r)), 80)
        draw_polyline(canvas, plans[mode_idx, : t + 1], meta, color, thickness=2)


def draw_flow(canvas: np.ndarray, flow: np.ndarray, occ_prob: np.ndarray, step: int = 12):
    h, w = canvas.shape[:2]
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    for y, x in zip(ys.reshape(-1), xs.reshape(-1)):
        if occ_prob[y, x] < 0.25:
            continue
        dx = float(flow[1, y, x])
        dy = float(flow[0, y, x])
        end_x = int(np.clip(x + dx * 1.8, 0, w - 1))
        end_y = int(np.clip(y + dy * 1.8, 0, h - 1))
        cv2.arrowedLine(canvas, (int(x), int(y)), (end_x, end_y), (255, 220, 80), 1, tipLength=0.3)


def add_label(canvas: np.ndarray, text: str):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(canvas, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def make_panel(
    sample: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    meta: Dict[str, float],
    t: int,
) -> np.ndarray:
    gt_occ = sample["future_occ"][t].sum(0).clamp(0, 1).cpu().numpy()
    pred_prob = torch.sigmoid(pred["occ_logits"][0, t]).sum(0).clamp(0, 1).detach().cpu().numpy()
    pred_flow = pred["flow"][0, t].detach().cpu().numpy()

    current = (base_canvas(sample) * 255).astype(np.uint8)
    draw_agent_tracks(current, sample, meta)
    add_label(current, "current BEV + past tracks")

    gt = overlay_occ(base_canvas(sample), gt_occ, (0.2, 1.0, 0.25), 0.95)
    gt = (gt * 255).astype(np.uint8)
    draw_gt_trajs(gt, sample, meta, t)
    add_label(gt, f"future GT occupancy + tracks t={t}")

    pred_canvas = overlay_occ(base_canvas(sample), pred_prob, (1.0, 0.2, 0.05), 0.95)
    pred_canvas = (pred_canvas * 255).astype(np.uint8)
    draw_pred_trajs(pred_canvas, pred, sample, meta, t)
    draw_flow(pred_canvas, pred_flow, pred_prob)
    add_label(pred_canvas, f"pred occupancy + flow + traj modes t={t}")

    plan = overlay_occ(base_canvas(sample), pred_prob, (0.8, 0.15, 1.0), 0.70)
    plan = (plan * 255).astype(np.uint8)
    draw_ego_plans(plan, pred, meta, t)
    add_label(plan, "ego candidate plans colored by risk")

    spacer = np.full((current.shape[0], 8, 3), 255, dtype=np.uint8)
    top = np.concatenate([current, spacer, gt], axis=1)
    bottom = np.concatenate([pred_canvas, spacer, plan], axis=1)
    spacer_h = np.full((8, top.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([top, spacer_h, bottom], axis=0)


def save_video(frames: list[np.ndarray], output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    future_count = int(sample["future_occ"].shape[0])
    frames = [make_panel(sample, pred, meta, t) for t in range(future_count)]
    overview_path = output_dir / f"{args.prefix}_overview.png"
    video_path = output_dir / f"{args.prefix}_rollout.mp4"
    plt.imsave(overview_path, frames[-1])
    save_video(frames, video_path, args.fps)
    print(f"saved overview: {overview_path}")
    print(f"saved video: {video_path}")


if __name__ == "__main__":
    main()
