from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class BevConfig:
    x_min: float = -80.0
    x_max: float = 80.0
    y_min: float = -80.0
    y_max: float = 80.0
    resolution: float = 0.5
    history_steps: int = 10
    future_steps: int = 80
    future_stride: int = 5
    max_agents: int = 128
    max_map_points: int = 4096
    max_map_vectors: int = 512
    max_polyline_points: int = 20
    max_traffic_lights: int = 64
    sensor_dim: int = 256

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def future_bins(self) -> List[int]:
        return list(range(1, self.future_steps + 1, self.future_stride))


CLASS_TO_CHANNEL = {
    1: 0,  # vehicle
    2: 1,  # pedestrian
    3: 2,  # cyclist
}


def import_womd_runtime():
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise ImportError(
            "WOMD parsing needs tensorflow and waymo-open-dataset. Install requirements.txt "
            "in the Ubuntu training environment before running preprocessing."
        ) from exc
    return tf, scenario_pb2


def find_tfrecords(root: Path, split: str, max_files: int = 0) -> List[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing WOMD split directory: {split_dir}")
    files = sorted(split_dir.glob("*.tfrecord*"))
    files = [path for path in files if ".gstmp" not in path.name]
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No TFRecord shards found under {split_dir}")
    return files


def iter_scenarios(shards: Sequence[Path]) -> Iterator[object]:
    tf, scenario_pb2 = import_womd_runtime()
    for path in shards:
        dataset = tf.data.TFRecordDataset(str(path), compression_type="")
        try:
            for record in dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytes(record.numpy()))
                yield scenario
        except tf.errors.DataLossError as exc:
            print(f"[warn] skip corrupted TFRecord shard: {path} ({exc})", flush=True)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def global_to_ego(x: float, y: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
    dx = x - ego_x
    dy = y - ego_y
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    return c * dx - s * dy, s * dx + c * dy


def metric_to_pixel(x: float, y: float, cfg: BevConfig) -> Optional[Tuple[int, int]]:
    row = int((x - cfg.x_min) / cfg.resolution)
    col = int((y - cfg.y_min) / cfg.resolution)
    if 0 <= row < cfg.height and 0 <= col < cfg.width:
        return row, col
    return None


def draw_box(grid: np.ndarray, x: float, y: float, length: float, width: float, yaw: float, cfg: BevConfig, value: float = 1.0):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for BEV box rasterization.") from exc

    half_l = max(length, 0.5) * 0.5
    half_w = max(width, 0.5) * 0.5
    corners = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float32,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rot.T + np.array([x, y], dtype=np.float32)
    pixels = []
    for cx, cy in corners:
        pix = metric_to_pixel(float(cx), float(cy), cfg)
        if pix is None:
            return
        row, col = pix
        pixels.append([col, row])
    cv2.fillConvexPoly(grid, np.asarray(pixels, dtype=np.int32), value)


def draw_polyline(grid: np.ndarray, points: Sequence[Tuple[float, float]], cfg: BevConfig, value: float = 1.0):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for BEV map rasterization.") from exc

    pixels = []
    for x, y in points:
        pix = metric_to_pixel(x, y, cfg)
        if pix is not None:
            row, col = pix
            pixels.append([col, row])
    if len(pixels) >= 2:
        cv2.polylines(grid, [np.asarray(pixels, dtype=np.int32)], False, value, thickness=1)


def state_to_ego(state, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float, float]:
    x, y = global_to_ego(float(state.center_x), float(state.center_y), ego_x, ego_y, ego_yaw)
    yaw = normalize_angle(float(state.heading) - ego_yaw)
    return x, y, yaw


def get_sdc_pose(scenario) -> Optional[Tuple[float, float, float]]:
    cur = int(scenario.current_time_index)
    if scenario.sdc_track_index < 0 or scenario.sdc_track_index >= len(scenario.tracks):
        return None
    sdc = scenario.tracks[scenario.sdc_track_index]
    if cur >= len(sdc.states) or not sdc.states[cur].valid:
        return None
    state = sdc.states[cur]
    return float(state.center_x), float(state.center_y), float(state.heading)


def _polyline_points(polyline, ego_x: float, ego_y: float, ego_yaw: float, limit: int) -> List[Tuple[float, float]]:
    pts = []
    for point in polyline:
        x, y = global_to_ego(float(point.x), float(point.y), ego_x, ego_y, ego_yaw)
        pts.append((x, y))
        if len(pts) >= limit:
            break
    return pts


def extract_map(scenario, ego_x: float, ego_y: float, ego_yaw: float, cfg: BevConfig):
    grid = np.zeros((3, cfg.height, cfg.width), dtype=np.float32)
    map_vectors = np.zeros((cfg.max_map_vectors, cfg.max_polyline_points, 8), dtype=np.float32)
    map_vector_mask = np.zeros((cfg.max_map_vectors,), dtype=np.float32)
    total_points = 0
    vector_idx = 0
    for feature in scenario.map_features:
        if total_points >= cfg.max_map_points:
            break
        channel = None
        polyline = None
        type_id = 0
        if feature.HasField("lane"):
            channel = 0
            polyline = feature.lane.polyline
            type_id = 0
        elif feature.HasField("road_line"):
            channel = 1
            polyline = feature.road_line.polyline
            type_id = 1
        elif feature.HasField("road_edge"):
            channel = 2
            polyline = feature.road_edge.polyline
            type_id = 2
        if channel is None or not polyline:
            continue
        pts = _polyline_points(polyline, ego_x, ego_y, ego_yaw, cfg.max_polyline_points)
        total_points += len(pts)
        draw_polyline(grid[channel], pts, cfg)
        if vector_idx < cfg.max_map_vectors and len(pts) >= 2:
            map_vector_mask[vector_idx] = 1.0
            for p_idx, (x, y) in enumerate(pts):
                prev_x, prev_y = pts[max(p_idx - 1, 0)]
                next_x, next_y = pts[min(p_idx + 1, len(pts) - 1)]
                dx = next_x - prev_x
                dy = next_y - prev_y
                norm = math.hypot(dx, dy) + 1e-6
                map_vectors[vector_idx, p_idx, 0] = x / cfg.x_max
                map_vectors[vector_idx, p_idx, 1] = y / cfg.y_max
                map_vectors[vector_idx, p_idx, 2] = dx / norm
                map_vectors[vector_idx, p_idx, 3] = dy / norm
                map_vectors[vector_idx, p_idx, 4 + type_id] = 1.0
                map_vectors[vector_idx, p_idx, 7] = p_idx / max(cfg.max_polyline_points - 1, 1)
            vector_idx += 1
    return grid, map_vectors, map_vector_mask


def extract_traffic_lights(scenario, ego_x: float, ego_y: float, ego_yaw: float, cfg: BevConfig) -> Tuple[np.ndarray, np.ndarray]:
    cur = int(scenario.current_time_index)
    past_indices = list(range(cur - cfg.history_steps + 1, cur + 1))
    traffic = np.zeros((cfg.history_steps, cfg.max_traffic_lights, 8), dtype=np.float32)
    mask = np.zeros((cfg.history_steps, cfg.max_traffic_lights), dtype=np.float32)
    dynamic_states = getattr(scenario, "dynamic_map_states", [])
    for t_pos, idx in enumerate(past_indices):
        if idx < 0 or idx >= len(dynamic_states):
            continue
        lane_states = getattr(dynamic_states[idx], "lane_states", [])
        light_idx = 0
        for lane_state in lane_states:
            if light_idx >= cfg.max_traffic_lights:
                break
            stop_point = getattr(lane_state, "stop_point", None)
            if stop_point is None:
                continue
            x, y = global_to_ego(float(stop_point.x), float(stop_point.y), ego_x, ego_y, ego_yaw)
            state = int(getattr(lane_state, "state", 0))
            traffic[t_pos, light_idx, 0] = x / cfg.x_max
            traffic[t_pos, light_idx, 1] = y / cfg.y_max
            traffic[t_pos, light_idx, 2] = 1.0 if state == 1 else 0.0
            traffic[t_pos, light_idx, 3] = 1.0 if state == 2 else 0.0
            traffic[t_pos, light_idx, 4] = 1.0 if state == 3 else 0.0
            traffic[t_pos, light_idx, 5] = min(max(state, 0), 10) / 10.0
            traffic[t_pos, light_idx, 6] = t_pos / max(cfg.history_steps - 1, 1)
            traffic[t_pos, light_idx, 7] = 1.0
            mask[t_pos, light_idx] = 1.0
            light_idx += 1
    return traffic, mask


def scenario_to_sample(scenario, cfg: BevConfig) -> Optional[Dict[str, torch.Tensor]]:
    ego_pose = get_sdc_pose(scenario)
    if ego_pose is None:
        return None
    ego_x, ego_y, ego_yaw = ego_pose
    cur = int(scenario.current_time_index)
    past_indices = list(range(cur - cfg.history_steps + 1, cur + 1))
    future_indices = [cur + offset for offset in cfg.future_bins]
    if past_indices[0] < 0:
        return None

    past = np.zeros((cfg.history_steps, 4, cfg.height, cfg.width), dtype=np.float32)
    future_occ = np.zeros((len(future_indices), 4, cfg.height, cfg.width), dtype=np.float32)
    future_flow = np.zeros((len(future_indices), 2, cfg.height, cfg.width), dtype=np.float32)
    agent_features = np.zeros((cfg.max_agents, cfg.history_steps, 8), dtype=np.float32)
    agent_mask = np.zeros((cfg.max_agents,), dtype=np.float32)
    traj_target = np.zeros((cfg.max_agents, len(future_indices), 2), dtype=np.float32)
    traj_mask = np.zeros((cfg.max_agents, len(future_indices)), dtype=np.float32)
    ego_future = np.zeros((len(future_indices), 2), dtype=np.float32)
    ego_future_mask = np.zeros((len(future_indices),), dtype=np.float32)

    kept_agents = 0
    for track_idx, track in enumerate(scenario.tracks):
        channel = CLASS_TO_CHANNEL.get(int(track.object_type), 3)
        states = track.states
        if cur >= len(states) or not states[cur].valid:
            continue
        if kept_agents < cfg.max_agents:
            agent_mask[kept_agents] = 1.0
        prev_future_xy = None
        for t_pos, idx in enumerate(past_indices):
            if 0 <= idx < len(states) and states[idx].valid:
                st = states[idx]
                x, y, yaw = state_to_ego(st, ego_x, ego_y, ego_yaw)
                draw_box(past[t_pos, channel], x, y, float(st.length), float(st.width), yaw, cfg)
                if kept_agents < cfg.max_agents:
                    agent_features[kept_agents, t_pos] = np.array(
                        [
                            x / cfg.x_max,
                            y / cfg.y_max,
                            math.sin(yaw),
                            math.cos(yaw),
                            float(st.velocity_x) / 30.0,
                            float(st.velocity_y) / 30.0,
                            float(st.length) / 20.0,
                            float(st.width) / 8.0,
                        ],
                        dtype=np.float32,
                    )
        for f_pos, idx in enumerate(future_indices):
            if 0 <= idx < len(states) and states[idx].valid:
                st = states[idx]
                x, y, yaw = state_to_ego(st, ego_x, ego_y, ego_yaw)
                draw_box(future_occ[f_pos, channel], x, y, float(st.length), float(st.width), yaw, cfg)
                if prev_future_xy is not None:
                    prev_x, prev_y = prev_future_xy
                    pix = metric_to_pixel(x, y, cfg)
                    if pix is not None:
                        row, col = pix
                        future_flow[f_pos, 0, row, col] = (x - prev_x) / cfg.resolution
                        future_flow[f_pos, 1, row, col] = (y - prev_y) / cfg.resolution
                prev_future_xy = (x, y)
                if kept_agents < cfg.max_agents:
                    traj_target[kept_agents, f_pos] = np.array([x / cfg.x_max, y / cfg.y_max], dtype=np.float32)
                    traj_mask[kept_agents, f_pos] = 1.0
                if track_idx == int(scenario.sdc_track_index):
                    ego_future[f_pos] = np.array([x / cfg.x_max, y / cfg.y_max], dtype=np.float32)
                    ego_future_mask[f_pos] = 1.0
        kept_agents += 1

    if agent_mask.sum() < 2:
        return None

    map_bev, map_vectors, map_vector_mask = extract_map(scenario, ego_x, ego_y, ego_yaw, cfg)
    traffic_lights, traffic_light_mask = extract_traffic_lights(scenario, ego_x, ego_y, ego_yaw, cfg)
    sensor_context = np.zeros((1, cfg.sensor_dim), dtype=np.float32)
    occ3d_future = np.zeros_like(future_occ)
    occ3d_mask = np.zeros((1,), dtype=np.float32)
    sample = {
        "past_bev": torch.from_numpy(past),
        "map_bev": torch.from_numpy(map_bev),
        "map_vectors": torch.from_numpy(map_vectors),
        "map_vector_mask": torch.from_numpy(map_vector_mask),
        "traffic_lights": torch.from_numpy(traffic_lights),
        "traffic_light_mask": torch.from_numpy(traffic_light_mask),
        "sensor_context": torch.from_numpy(sensor_context),
        "future_occ": torch.from_numpy(future_occ),
        "future_flow": torch.from_numpy(future_flow),
        "occ3d_future": torch.from_numpy(occ3d_future),
        "occ3d_mask": torch.from_numpy(occ3d_mask),
        "agent_features": torch.from_numpy(agent_features),
        "agent_mask": torch.from_numpy(agent_mask),
        "traj_target": torch.from_numpy(traj_target),
        "traj_mask": torch.from_numpy(traj_mask),
        "ego_future": torch.from_numpy(ego_future),
        "ego_future_mask": torch.from_numpy(ego_future_mask),
        "scenario_id": scenario.scenario_id,
    }
    return sample


def save_shard(samples: List[Dict[str, torch.Tensor]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_keys = [
        "past_bev",
        "map_bev",
        "map_vectors",
        "map_vector_mask",
        "traffic_lights",
        "traffic_light_mask",
        "sensor_context",
        "future_occ",
        "future_flow",
        "occ3d_future",
        "occ3d_mask",
        "agent_features",
        "agent_mask",
        "traj_target",
        "traj_mask",
        "ego_future",
        "ego_future_mask",
        "camera_images",
        "lidar_points",
        "lidar_mask",
    ]
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys if key in samples[0]}
    batch["scenario_id"] = [sample["scenario_id"] for sample in samples]
    torch.save(batch, output_path)


def write_metadata(output_dir: Path, cfg: BevConfig, split: str, count: int):
    meta = {"split": split, "num_samples": count, "bev": asdict(cfg)}
    (output_dir / f"{split}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class BevShardDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str | Path, split: str):
        self.paths = sorted(Path(data_dir).glob(f"{split}_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No processed shards matching {split}_*.pt under {data_dir}")
        self.index: List[Tuple[int, int]] = []
        self._sizes: List[int] = []
        for shard_idx, path in enumerate(self.paths):
            data = torch.load(path, map_location="cpu")
            size = int(data["past_bev"].shape[0])
            self._sizes.append(size)
            for row in range(size):
                self.index.append((shard_idx, row))
        self._cache_idx = -1
        self._cache = None

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, shard_idx: int):
        if shard_idx != self._cache_idx:
            self._cache = torch.load(self.paths[shard_idx], map_location="cpu")
            self._cache_idx = shard_idx
        return self._cache

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, row = self.index[idx]
        data = self._load(shard_idx)
        item = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                item[key] = value[row]
        return item
