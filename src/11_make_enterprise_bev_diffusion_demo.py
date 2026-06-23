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


def text_factor_from_scale(scale: int) -> float:
    return float(np.clip(float(scale) / 2.0, 1.3, 3.1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create enterprise-style BEV diffusion world-model demos with ego-plan risk visualization."
    )
    parser.add_argument("--data_dir", default="/mnt/data1/wzy/processed/womd_bev_r1_train100")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--indices", default="0", help="Comma-separated validation indices, for example: 10,50,100")
    parser.add_argument("--output_dir", default="/mnt/data1/wzy/outputs/bev_diffusion_world_model_r4_x0_sample_pred/enterprise_demo")
    parser.add_argument("--prefix", default="enterprise_diffusion")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--video_crf", type=int, default=16)
    parser.add_argument("--video_preset", default="slow")
    parser.add_argument("--video_pix_fmt", default="yuv420p")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vis_erode_pred", type=int, default=1)
    parser.add_argument("--keep_frames", action="store_true")
    return parser.parse_args()


def parse_indices(text: str) -> List[int]:
    return [int(piece.strip()) for piece in text.split(",") if piece.strip()]


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
    rows = ((xy[:, 0] * x_max - x_min) / res).astype(np.int32)
    cols = ((xy[:, 1] * y_max - y_min) / res).astype(np.int32)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return np.stack([cols[valid], rows[valid]], axis=-1)


def norm_xy_to_pixel_with_valid(xy: np.ndarray, meta: Dict[str, float], h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    x_min = float(meta.get("x_min", -80.0))
    x_max = float(meta.get("x_max", 80.0))
    y_min = float(meta.get("y_min", -80.0))
    y_max = float(meta.get("y_max", 80.0))
    res = float(meta.get("resolution", 1.0))
    rows = ((xy[:, 0] * x_max - x_min) / res).astype(np.int32)
    cols = ((xy[:, 1] * y_max - y_min) / res).astype(np.int32)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return np.stack([cols, rows], axis=-1), valid


def pixel_to_norm_xy(points: np.ndarray, meta: Dict[str, float]) -> np.ndarray:
    x_min = float(meta.get("x_min", -80.0))
    x_max = float(meta.get("x_max", 80.0))
    y_min = float(meta.get("y_min", -80.0))
    y_max = float(meta.get("y_max", 80.0))
    res = float(meta.get("resolution", 1.0))
    cols = points[:, 0].astype(np.float32)
    rows = points[:, 1].astype(np.float32)
    x = (rows * res + x_min) / x_max
    y = (cols * res + y_min) / y_max
    return np.stack([x, y], axis=-1)


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
    canvas = np.maximum(canvas, (past * 245).astype(np.uint8)[..., None])
    return canvas


def overlay_heat(canvas: np.ndarray, heat: np.ndarray, color: RGB, alpha: float = 0.82) -> np.ndarray:
    heat = np.clip(heat, 0.0, 1.0)
    out = canvas.astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    out = out * (1.0 - alpha * heat[..., None]) + color_arr * (alpha * heat[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def refine_pred_for_display(heat: np.ndarray, erode_iters: int = 1) -> np.ndarray:
    """Sharpen only the rendered occupancy; training/eval tensors stay unchanged."""
    heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
    if erode_iters <= 0 or heat.max() <= 0:
        return heat
    soft = np.sqrt(heat)
    mask = (soft > 0.08).astype(np.uint8)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.erode(mask, kernel, iterations=int(erode_iters))
    return soft * mask.astype(np.float32)


def draw_history(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], max_agents: int = 64):
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


def draw_gt_traj(canvas: np.ndarray, sample: Dict[str, torch.Tensor], meta: Dict[str, float], end_t: int, max_agents: int = 64):
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


def map_lane_points(sample: Dict[str, torch.Tensor], dilation: int = 5) -> np.ndarray:
    map_mask = sample["map_bev"][0].clamp(0, 1).cpu().numpy()
    mask = (map_mask > 0.05).astype(np.uint8)
    if dilation > 1:
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([xs, ys], axis=-1).astype(np.int32)


def lane_distance_map(sample: Dict[str, torch.Tensor]) -> np.ndarray:
    lane = sample["map_bev"][0].clamp(0, 1).cpu().numpy()
    lane_mask = (lane > 0.05).astype(np.uint8)
    if lane_mask.max() == 0:
        return np.ones_like(lane, dtype=np.float32)
    dist = cv2.distanceTransform(1 - lane_mask, cv2.DIST_L2, 3)
    return (dist / max(float(max(lane.shape)), 1.0)).astype(np.float32)


def snap_plan_to_map(
    plan: np.ndarray,
    sample: Dict[str, torch.Tensor],
    meta: Dict[str, float],
    max_snap_px: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = sample["past_bev"].shape[-2:]
    lane_pts = map_lane_points(sample)
    if lane_pts.shape[0] == 0:
        return plan, np.ones(plan.shape[0], dtype=bool)
    pix, valid = norm_xy_to_pixel_with_valid(plan, meta, h, w)
    snapped = pix.astype(np.float32).copy()
    keep = valid.copy()
    for idx, is_valid in enumerate(valid):
        if not is_valid:
            keep[idx] = False
            continue
        delta = lane_pts.astype(np.float32) - pix[idx].astype(np.float32)
        dist2 = np.sum(delta * delta, axis=1)
        best = int(np.argmin(dist2))
        dist = float(np.sqrt(dist2[best]))
        if dist <= max_snap_px:
            snapped[idx] = lane_pts[best]
        else:
            keep[idx] = False
    if keep.sum() >= 2:
        valid_indices = np.where(keep)[0]
        first_valid = int(valid_indices[0])
        for idx in range(0, first_valid):
            snapped[idx] = snapped[first_valid]
        last_valid = first_valid
        for idx in range(first_valid + 1, snapped.shape[0]):
            if keep[idx]:
                last_valid = idx
            else:
                snapped[idx] = snapped[last_valid]
        return pixel_to_norm_xy(snapped, meta), keep
    return plan, keep


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((count, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points.astype(np.float32), count, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total < 1e-6:
        return np.repeat(points[:1].astype(np.float32), count, axis=0)
    query = np.linspace(0.0, total, count, dtype=np.float32)
    x = np.interp(query, dist, points[:, 0])
    y = np.interp(query, dist, points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def make_map_vector_ego_plans(
    sample: Dict[str, torch.Tensor],
    meta: Dict[str, float],
    future_count: int,
    max_plans: int = 8,
) -> List[np.ndarray]:
    if "map_vectors" not in sample or "map_vector_mask" not in sample:
        return []
    vectors = sample["map_vectors"].cpu().numpy()
    masks = sample["map_vector_mask"].cpu().numpy() > 0.5
    candidates: List[Tuple[float, np.ndarray]] = []
    for vector, active in zip(vectors, masks):
        if not active:
            continue
        point_mask = vector[:, 4] > 0.5
        pts = vector[point_mask, :2].astype(np.float32)
        dirs = vector[point_mask, 2:4].astype(np.float32)
        if pts.shape[0] < 3:
            continue
        d0 = np.linalg.norm(pts, axis=1)
        closest = int(np.argmin(d0))
        if float(d0[closest]) > 0.35:
            continue
        forward_dir = float(np.nanmean(dirs[:, 0])) if dirs.shape[0] else 0.0
        if forward_dir < 0.0:
            pts = pts[::-1]
            closest = pts.shape[0] - 1 - closest
        if closest > pts.shape[0] - 2:
            continue
        path = pts[closest:]
        if path.shape[0] < 2 or float(path[-1, 0] - path[0, 0]) < 0.02:
            continue
        forward_ratio = float((path[:, 0] > -0.05).mean())
        if forward_ratio < 0.6:
            continue
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        lateral = float(abs(path[0, 1]))
        score = float(d0[closest]) + 0.15 * lateral - 0.03 * min(length, 10.0)
        for speed_scale in (0.55, 0.75, 1.0):
            keep_count = max(2, int(round(2 + (path.shape[0] - 1) * speed_scale)))
            variant = path[:keep_count]
            candidates.append((score + 0.04 * (1.0 - speed_scale), resample_polyline(variant, future_count)))
    candidates.sort(key=lambda item: item[0])

    plans: List[np.ndarray] = []
    for _, plan in candidates:
        if all(float(np.mean(np.linalg.norm(plan - prev, axis=1))) > 0.008 for prev in plans):
            snapped, keep = snap_plan_to_map(plan, sample, meta, max_snap_px=10.0)
            if keep.sum() >= 2:
                plans.append(snapped)
        if len(plans) >= max_plans:
            break
    return plans


def make_rule_ego_plans(sample: Dict[str, torch.Tensor], meta: Dict[str, float], future_count: int) -> np.ndarray:
    lane_plans = make_map_vector_ego_plans(sample, meta, future_count)
    if lane_plans:
        return np.asarray(lane_plans, dtype=np.float32)

    x_max = float(meta.get("x_max", 80.0))
    speed = 6.0 / x_max
    ts = np.arange(1, future_count + 1, dtype=np.float32)
    centerline = np.stack([speed * ts, np.zeros_like(ts)], axis=-1)
    snapped_center, keep_center = snap_plan_to_map(centerline, sample, meta, max_snap_px=32.0)
    plan = snapped_center if keep_center.sum() >= 2 else centerline
    return np.asarray([plan], dtype=np.float32)


def _plan_local_values(plan: np.ndarray, field: np.ndarray, meta: Dict[str, float], radius: int = 2) -> List[float]:
    h, w = field.shape[-2:]
    values = []
    for t, xy in enumerate(plan):
        if t < 1:
            continue
        pts = norm_xy_to_pixel(xy[None], meta, h, w)
        if pts.shape[0] == 0:
            values.append(1.0)
            continue
        col, row = int(pts[0, 0]), int(pts[0, 1])
        r0, r1 = max(row - radius, 0), min(row + radius + 1, h)
        c0, c1 = max(col - radius, 0), min(col + radius + 1, w)
        local = field[min(t, field.shape[0] - 1), r0:r1, c0:c1] if field.ndim == 3 else field[r0:r1, c0:c1]
        values.append(float(np.percentile(local, 90)))
    return values


def score_plan_components(
    plan: np.ndarray,
    pred_prob: np.ndarray,
    uncertainty: np.ndarray,
    lane_dist: np.ndarray,
    meta: Dict[str, float],
) -> Dict[str, float]:
    collision_vals = _plan_local_values(plan, pred_prob, meta, radius=2)
    uncertainty_vals = _plan_local_values(plan, uncertainty, meta, radius=2)
    offroad_vals = _plan_local_values(plan, lane_dist, meta, radius=1)
    if plan.shape[0] >= 3:
        second_diff = plan[2:] - 2.0 * plan[1:-1] + plan[:-2]
        smoothness = float(np.mean(np.linalg.norm(second_diff, axis=-1)))
    else:
        smoothness = 0.0
    progress = max(float(plan[-1, 0] - plan[0, 0]), 0.0)
    collision = float(np.mean(collision_vals) + 0.7 * np.max(collision_vals)) if collision_vals else 0.0
    uncertainty_cost = float(np.mean(uncertainty_vals) + 0.5 * np.max(uncertainty_vals)) if uncertainty_vals else 0.0
    offroad = float(np.mean(offroad_vals) + 0.5 * np.max(offroad_vals)) if offroad_vals else 0.0
    total = (
        1.00 * collision
        + 0.65 * uncertainty_cost
        + 1.20 * offroad
        + 0.35 * smoothness
        - 0.35 * progress
    )
    return {
        "total": float(np.clip(total, 0.0, 1.0)),
        "collision": float(np.clip(collision, 0.0, 1.0)),
        "uncertainty": float(np.clip(uncertainty_cost, 0.0, 1.0)),
        "offroad": float(np.clip(offroad, 0.0, 1.0)),
        "smoothness": smoothness,
        "progress": progress,
    }


def score_plan_risk(plan: np.ndarray, pred_prob: np.ndarray, meta: Dict[str, float]) -> float:
    h, w = pred_prob.shape[-2:]
    risks = []
    for t, xy in enumerate(plan):
        if t < 2:
            continue
        pts = norm_xy_to_pixel(xy[None], meta, h, w)
        if pts.shape[0] == 0:
            risks.append(0.2)
            continue
        col, row = int(pts[0, 0]), int(pts[0, 1])
        r0, r1 = max(row - 2, 0), min(row + 3, h)
        c0, c1 = max(col - 2, 0), min(col + 3, w)
        local = pred_prob[min(t, pred_prob.shape[0] - 1), r0:r1, c0:c1]
        risks.append(float(np.percentile(local, 90)))
    if not risks:
        return 0.0
    return float(np.clip(np.mean(risks) + 0.5 * np.max(risks), 0.0, 1.0))


def draw_ego_risk(
    canvas: np.ndarray,
    sample: Dict[str, torch.Tensor],
    meta: Dict[str, float],
    pred_prob_seq: np.ndarray,
    uncertainty_seq: np.ndarray,
    end_t: int,
):
    risk_heat = np.maximum.reduce(pred_prob_seq[min(2, end_t) : end_t + 1])
    unc_heat = np.maximum.reduce(uncertainty_seq[min(2, end_t) : end_t + 1])
    heat_floor = 0.12
    risk_heat = np.maximum(risk_heat, 0.65 * unc_heat)
    risk_heat = np.where(risk_heat >= heat_floor, risk_heat, 0.0)
    heat_den = max(float(np.percentile(risk_heat, 99)), heat_floor)
    vis_heat = risk_heat / heat_den
    canvas[:] = overlay_heat(canvas, np.clip(vis_heat, 0.0, 1.0), (255, 55, 70), 0.55)
    plans = make_rule_ego_plans(sample, meta, pred_prob_seq.shape[0])
    lane_dist = lane_distance_map(sample)
    score_dicts = [score_plan_components(plan, pred_prob_seq, uncertainty_seq, lane_dist, meta) for plan in plans]
    risks = np.asarray([score["total"] for score in score_dicts], dtype=np.float32)
    risk_den = max(float(np.percentile(risks, 90)), 0.15)
    risk_vis = np.clip(risks / risk_den, 0.0, 1.0)
    best_idx = int(np.argmin(risks)) if risks.size else -1
    for idx in np.argsort(risks):
        risk = float(risk_vis[idx])
        if idx == best_idx:
            color = (80, 255, 95)
            thickness = 3
        else:
            color = (int(255 * risk), int(220 * (1.0 - risk)), 70)
            thickness = 2
        draw_poly(canvas, plans[idx, : end_t + 1], meta, color, thickness)
    if "ego_future" in sample and "ego_future_mask" in sample:
        ego = sample["ego_future"].cpu().numpy()
        mask = sample["ego_future_mask"].cpu().numpy() > 0.5
        xy = ego[: end_t + 1][mask[: end_t + 1]]
        draw_poly(canvas, xy, meta, (245, 245, 245), 2)
    return risks


def add_title(img: np.ndarray, title: str, subtitle: str | None = None, scale: int = 1) -> np.ndarray:
    scale = max(1, int(scale))
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    text_factor = text_factor_from_scale(scale)
    bar_h = int(round((28 if subtitle else 22) * text_factor))
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = img
    title_scale = 0.34 * text_factor
    subtitle_scale = 0.24 * text_factor
    title_thickness = max(1, int(round(0.9 * text_factor)))
    subtitle_thickness = max(1, int(round(0.65 * text_factor)))
    cv2.putText(
        out,
        title,
        (int(round(6 * text_factor)), int(round(15 * text_factor))),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (248, 248, 248),
        title_thickness,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            out,
            subtitle,
            (int(round(6 * text_factor)), int(round(26 * text_factor))),
            cv2.FONT_HERSHEY_SIMPLEX,
            subtitle_scale,
            (188, 188, 188),
            subtitle_thickness,
            cv2.LINE_AA,
        )
    return out


def add_legend(img: np.ndarray, scale: int) -> np.ndarray:
    scale = max(1, int(scale))
    text_factor = text_factor_from_scale(scale)
    legend_h = int(round(58 * text_factor))
    legend = np.zeros((legend_h, img.shape[1], 3), dtype=np.uint8)
    entries = [
        ("white: current / ego GT", (245, 245, 245)),
        ("cyan: map prior", (0, 135, 160)),
        ("blue: actor history", (110, 185, 255)),
        ("green: GT future", (70, 255, 110)),
        ("orange: diffusion occ", (255, 125, 35)),
        ("yellow: flow", (255, 225, 70)),
        ("ego plans: green low risk, red high risk", (180, 245, 80)),
    ]
    x, y = int(round(12 * text_factor)), int(round(20 * text_factor))
    font_scale = 0.36 * text_factor
    thickness = max(1, int(round(0.65 * text_factor)))
    for text, color in entries:
        cv2.circle(legend, (x, y - int(round(4 * text_factor))), int(round(5 * text_factor)), color, -1, cv2.LINE_AA)
        cv2.putText(
            legend,
            text,
            (x + int(round(14 * text_factor)), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (235, 235, 235),
            thickness,
            cv2.LINE_AA,
        )
        x += int(round(245 * text_factor))
        if x > img.shape[1] - int(round(245 * text_factor)):
            x, y = int(round(12 * text_factor)), y + int(round(25 * text_factor))
    return np.concatenate([img, legend], axis=0)


def tile_panels(panels: List[np.ndarray], title: str, summary: str, scale: int) -> np.ndarray:
    scale = max(1, int(scale))
    text_factor = text_factor_from_scale(scale)
    resized = panels
    h = max(panel.shape[0] for panel in resized)
    w = max(panel.shape[1] for panel in resized)
    padded = []
    for panel in resized:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(out)
    gap = 12 * scale
    sep_v = np.full((h, gap, 3), 245, dtype=np.uint8)
    row1 = np.concatenate([padded[0], sep_v, padded[1], sep_v, padded[2]], axis=1)
    row2 = np.concatenate([padded[3], sep_v, padded[4], sep_v, padded[5]], axis=1)
    sep_h = np.full((gap, row1.shape[1], 3), 245, dtype=np.uint8)
    body = np.concatenate([row1, sep_h, row2], axis=0)
    header_h = int(round(24 * text_factor))
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        summary,
        (int(round(10 * text_factor)), int(round(17 * text_factor))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * text_factor,
        (210, 210, 210),
        max(1, int(round(0.65 * text_factor))),
        cv2.LINE_AA,
    )
    return add_legend(np.concatenate([header, body], axis=0), scale)


def save_h264_video(frames: List[np.ndarray], out_path: Path, fps: int, crf: int, preset: str, pix_fmt: str) -> Path:
    if not frames:
        raise ValueError("No frames were provided for video export.")
    h, w = frames[0].shape[:2]
    even_h = h + (h % 2)
    even_w = w + (w % 2)
    padded_frames = []
    for frame in frames:
        if frame.shape[0] != h or frame.shape[1] != w:
            raise ValueError("All video frames must have the same shape.")
        if even_h != h or even_w != w:
            padded = np.zeros((even_h, even_w, 3), dtype=frame.dtype)
            padded[:h, :w] = frame
            padded_frames.append(padded)
        else:
            padded_frames.append(frame)

    frame_dir = out_path.parent / f".{out_path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    for idx, frame in enumerate(padded_frames):
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
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-pix_fmt",
        str(pix_fmt),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (even_w, even_h),
        )
        for frame in padded_frames:
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


def sample_metrics(outputs: List[Dict[str, torch.Tensor]], sample: Dict[str, torch.Tensor], threshold: float) -> Dict[str, float]:
    pred_stack = torch.stack([out["occ_probs"][0] for out in outputs])
    mean_occ = pred_stack.mean(0)
    pred_bin = (mean_occ > threshold).float()
    true_occ = sample["future_occ"].to(pred_stack.device).float()
    inter = (pred_bin * true_occ).sum()
    union = ((pred_bin + true_occ) > 0).float().sum().clamp_min(1.0)
    horizon_iou = []
    for t in range(true_occ.shape[0]):
        p = pred_bin[t]
        g = true_occ[t]
        horizon_iou.append(float(((p * g).sum() / ((p + g) > 0).float().sum().clamp_min(1.0)).cpu()))
    third = max(1, len(horizon_iou) // 3)
    diversity = float(pred_stack.flatten(1).std(0, unbiased=False).mean().cpu())
    return {
        "occupancy_iou": float((inter / union).cpu()),
        "occ_iou_near": float(np.mean(horizon_iou[:third])),
        "occ_iou_mid": float(np.mean(horizon_iou[third : 2 * third])),
        "occ_iou_far": float(np.mean(horizon_iou[2 * third :])),
        "pred_positive_ratio": float(pred_bin.mean().cpu()),
        "gt_positive_ratio": float(true_occ.mean().cpu()),
        "mean_probability": float(mean_occ.mean().cpu()),
        "sample_diversity": diversity,
    }


def make_panels(
    sample: Dict[str, torch.Tensor],
    outputs: List[Dict[str, torch.Tensor]],
    meta: Dict[str, float],
    t: int,
    vis_erode_pred: int,
    render_scale: int,
):
    pred_stack = torch.stack([out["occ_probs"][0] for out in outputs])
    mean_probs = pred_stack.mean(0)
    uncertainty_probs = pred_stack.std(0, unbiased=False)
    mean_occ_seq = mean_probs.sum(1).clamp(0, 1).detach().cpu().numpy()
    risk_prob_seq = mean_probs.amax(1).detach().cpu().numpy()
    uncertainty_seq = uncertainty_probs.amax(1).detach().cpu().numpy()
    pred_occ = mean_occ_seq[t]
    pred_occ_vis = refine_pred_for_display(pred_occ, vis_erode_pred)
    gt_occ = sample["future_occ"][t].sum(0).clamp(0, 1).cpu().numpy()
    flow = outputs[0].get("flow")
    flow_np = None if flow is None else flow[0, t].detach().cpu().numpy()

    condition = base_layer(sample)
    draw_history(condition, sample, meta)

    gt = overlay_heat(base_layer(sample), gt_occ, (70, 255, 110), 0.86)
    draw_gt_traj(gt, sample, meta, t)

    pred = overlay_heat(base_layer(sample), pred_occ_vis, (255, 125, 35), 0.82)

    modes = base_layer(sample)
    draw_history(modes, sample, meta)
    mode_colors = [(255, 125, 35), (255, 205, 55), (200, 95, 255), (255, 80, 180)]
    for sample_idx, out in enumerate(outputs[: len(mode_colors)]):
        occ = out["occ_probs"][0, t].sum(0).clamp(0, 1).detach().cpu().numpy()
        occ_vis = refine_pred_for_display(occ, vis_erode_pred)
        modes = overlay_heat(modes, occ_vis, mode_colors[sample_idx], 0.46)

    flow_panel = overlay_heat(base_layer(sample), pred_occ_vis, (200, 95, 255), 0.55)
    draw_flow(flow_panel, flow_np, pred_occ, step=8)

    ego = base_layer(sample)
    risks = draw_ego_risk(ego, sample, meta, risk_prob_seq, uncertainty_seq, t)

    return [
        add_title(condition, "Current Scene", "BEV + map + history", render_scale),
        add_title(gt, f"GT Occupancy t={t}", None, render_scale),
        add_title(pred, f"Pred Occupancy t={t}", "diffusion ensemble mean", render_scale),
        add_title(modes, "Multi-sample Futures", "colored diffusion modes", render_scale),
        add_title(flow_panel, "Occupancy Flow", "motion direction", render_scale),
        add_title(ego, "Ego Plan Risk", f"max risk={float(risks.max()):.3f}", render_scale),
    ], risks


def render_one_index(
    dataset: BevShardDataset,
    model: ConditionalBevDenoiser,
    scheduler,
    target_cfg: DiffusionTargetConfig,
    meta: Dict[str, float],
    index: int,
    args,
    device: torch.device,
) -> Dict[str, float]:
    sample = dataset[index]
    batch = {key: value.unsqueeze(0).to(device).float() for key, value in sample.items()}
    with torch.no_grad():
        outputs = run_diffusion_ensemble(
            model,
            scheduler,
            batch,
            target_cfg,
            max(1, args.num_samples),
            args.num_inference_steps,
            args.seed + index * 1000,
        )

    metrics = sample_metrics(outputs, sample, args.threshold)
    summary = (
        f"idx={index}  IoU={metrics['occupancy_iou']:.3f}  "
        f"n/m/f={metrics['occ_iou_near']:.3f}/{metrics['occ_iou_mid']:.3f}/{metrics['occ_iou_far']:.3f}  "
        f"div={metrics['sample_diversity']:.4f}"
    )

    frames = []
    future_count = int(sample["future_occ"].shape[0])
    risk_values = []
    for t in range(future_count):
        panels, risks = make_panels(sample, outputs, meta, t, args.vis_erode_pred, args.scale)
        risk_values.append(float(risks.max()))
        frames.append(tile_panels(panels, "Enterprise BEV Diffusion World Model Demo", summary, args.scale))

    out_dir = Path(args.output_dir)
    overview_idx = min(future_count - 1, max(0, future_count // 2))
    overview_path = out_dir / f"{args.prefix}_{index}_overview.png"
    video_path = out_dir / f"{args.prefix}_{index}_rollout.mp4"
    metrics_path = out_dir / f"{args.prefix}_{index}_metrics.json"
    cv2.imwrite(str(overview_path), cv2.cvtColor(frames[overview_idx], cv2.COLOR_RGB2BGR))
    frame_dir = save_h264_video(frames, video_path, args.fps, args.video_crf, args.video_preset, args.video_pix_fmt)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)

    metrics.update(
        {
            "index": float(index),
            "num_samples": float(max(1, args.num_samples)),
            "max_ego_risk": float(np.max(risk_values)),
            "mean_ego_risk": float(np.mean(risk_values)),
            "overview": str(overview_path),
            "video": str(video_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved overview: {overview_path}")
    print(f"saved video: {video_path}")
    print(f"saved metrics: {metrics_path}")
    return metrics


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices = parse_indices(args.indices)
    dataset = BevShardDataset(args.data_dir, args.split)
    meta = load_bev_meta(args.data_dir, args.split)
    model, target_cfg, scheduler = load_model(dataset[indices[0]], args.checkpoint, device)

    all_metrics = []
    for index in indices:
        all_metrics.append(render_one_index(dataset, model, scheduler, target_cfg, meta, index, args, device))
    report_path = out_dir / f"{args.prefix}_report.json"
    report_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
