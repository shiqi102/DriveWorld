from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sensor_encoders import MultiSensorFusionEncoder


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinePositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        freqs = torch.arange(self.dim // 4, device=device).float()
        freqs = 1.0 / (10000 ** (freqs / max(1, self.dim // 4 - 1)))
        enc = torch.cat(
            [
                torch.sin(x[..., None] * freqs),
                torch.cos(x[..., None] * freqs),
                torch.sin(y[..., None] * freqs),
                torch.cos(y[..., None] * freqs),
            ],
            dim=-1,
        )
        if enc.shape[-1] < self.dim:
            enc = F.pad(enc, (0, self.dim - enc.shape[-1]))
        return enc[..., : self.dim].reshape(1, h * w, self.dim)


class VectorMapEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, depth: int = 2):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.polyline_encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, map_vectors: torch.Tensor, map_vector_mask: torch.Tensor) -> torch.Tensor:
        b, m, p, _ = map_vectors.shape
        point_tokens = self.point_mlp(map_vectors)
        point_valid = (map_vectors[..., :2].abs().sum(dim=-1) > 0).float()
        denom = point_valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        poly_tokens = (point_tokens * point_valid[..., None]).sum(dim=2) / denom
        empty = map_vector_mask.sum(dim=1) < 0.5
        if empty.any():
            map_vector_mask = map_vector_mask.clone()
            map_vector_mask[empty, 0] = 1.0
            poly_tokens = poly_tokens.clone()
            poly_tokens[empty, 0] = 0.0
        key_padding_mask = map_vector_mask < 0.5
        return self.polyline_encoder(poly_tokens, src_key_padding_mask=key_padding_mask)


class TrafficLightEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, traffic_lights: torch.Tensor, traffic_light_mask: torch.Tensor) -> torch.Tensor:
        b, t, l, d = traffic_lights.shape
        tokens = self.mlp(traffic_lights.reshape(b, t * l, d))
        mask = traffic_light_mask.reshape(b, t * l) < 0.5
        empty = (~mask).float().sum(dim=1) < 0.5
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0.0
        valid = (~mask).float().sum(dim=1, keepdim=True).clamp_min(1.0)
        query = (tokens * (~mask)[..., None].float()).sum(dim=1, keepdim=True) / valid[..., None]
        context, _ = self.attn(query, tokens, tokens, key_padding_mask=mask)
        return context


class SensorTokenAdapter(nn.Module):
    """Adapter for future camera/lidar tokens from Waymo Perception or nuScenes.

    WOMD has no raw camera/lidar tensors, so training currently feeds a zero
    placeholder. When raw sensor encoders are added, their tokens enter here.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sensor_context: torch.Tensor) -> torch.Tensor:
        return self.proj(sensor_context)


class MultiModalTrajectoryHead(nn.Module):
    def __init__(self, hidden_dim: int, future_steps: int, num_modes: int):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.traj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modes * future_steps * 2),
        )
        self.logits = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_modes))

    def forward(self, agent_context: torch.Tensor):
        b, a, _ = agent_context.shape
        traj = self.traj(agent_context).reshape(b, a, self.num_modes, self.future_steps, 2)
        logits = self.logits(agent_context)
        return traj, logits


class PlannerHead(nn.Module):
    def __init__(self, hidden_dim: int, future_steps: int, num_modes: int):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.plan = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modes * future_steps * 2),
        )
        self.risk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_modes))

    def forward(self, scene_token: torch.Tensor):
        plan = self.plan(scene_token).reshape(scene_token.shape[0], self.num_modes, self.future_steps, 2)
        risk = self.risk(scene_token)
        return plan, risk


class BevOccupancyFlowWorldModel(nn.Module):
    def __init__(
        self,
        history_steps: int,
        future_steps: int,
        hidden_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        num_classes: int = 4,
        traj_agents: int = 128,
        num_modes: int = 6,
        sensor_dim: int = 256,
        map_dim: int = 8,
        traffic_dim: int = 8,
    ):
        super().__init__()
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_classes = num_classes
        self.traj_agents = traj_agents
        self.num_modes = num_modes
        self.sensor_dim = sensor_dim

        bev_in_channels = history_steps * num_classes + 3
        self.bev_stem = ConvStem(bev_in_channels, hidden_dim)
        self.pos = SinePositionEncoding(hidden_dim)
        self.map_encoder = VectorMapEncoder(map_dim, hidden_dim, num_heads)
        self.traffic_encoder = TrafficLightEncoder(traffic_dim, hidden_dim, num_heads)
        self.sensor_adapter = SensorTokenAdapter(sensor_dim, hidden_dim)
        self.raw_sensor_encoder = MultiSensorFusionEncoder(hidden_dim=hidden_dim, num_heads=num_heads)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(layer, num_layers=depth)

        self.agent_encoder = nn.Sequential(
            nn.Linear(history_steps * 8 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.agent_scene_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.occ_head = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, future_steps * num_classes, 1),
        )
        self.flow_head = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, future_steps * 2, 1),
        )
        self.dense_occ_head = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, future_steps * num_classes, 1),
        )
        self.traj_head = MultiModalTrajectoryHead(hidden_dim, future_steps, num_modes)
        self.planner_head = PlannerHead(hidden_dim, future_steps, num_modes)

    def forward(
        self,
        past_bev: torch.Tensor,
        map_bev: torch.Tensor,
        agent_features: torch.Tensor,
        agent_mask: torch.Tensor,
        map_vectors: torch.Tensor | None = None,
        map_vector_mask: torch.Tensor | None = None,
        traffic_lights: torch.Tensor | None = None,
        traffic_light_mask: torch.Tensor | None = None,
        sensor_context: torch.Tensor | None = None,
        camera_images: torch.Tensor | None = None,
        lidar_points: torch.Tensor | None = None,
        lidar_mask: torch.Tensor | None = None,
    ):
        b, hist, cls, h, w = past_bev.shape
        bev = torch.cat([past_bev.reshape(b, hist * cls, h, w), map_bev], dim=1)
        feat = self.bev_stem(bev)
        _, c, hh, ww = feat.shape
        bev_tokens = feat.flatten(2).transpose(1, 2)
        bev_tokens = bev_tokens + self.pos(hh, ww, bev_tokens.device)

        extra_tokens = []
        if map_vectors is not None and map_vector_mask is not None:
            extra_tokens.append(self.map_encoder(map_vectors, map_vector_mask))
        if traffic_lights is not None and traffic_light_mask is not None:
            extra_tokens.append(self.traffic_encoder(traffic_lights, traffic_light_mask))
        if camera_images is not None or lidar_points is not None:
            extra_tokens.append(
                self.raw_sensor_encoder(
                    images=camera_images,
                    lidar_points=lidar_points,
                    lidar_mask=lidar_mask,
                )
            )
        if sensor_context is not None:
            extra_tokens.append(self.sensor_adapter(sensor_context))
        fused_tokens = torch.cat([bev_tokens] + extra_tokens, dim=1) if extra_tokens else bev_tokens
        encoded_tokens = self.scene_encoder(fused_tokens)
        bev_encoded = encoded_tokens[:, : bev_tokens.shape[1]]
        global_scene = encoded_tokens.mean(dim=1)
        scene_feat = bev_encoded.transpose(1, 2).reshape(b, c, hh, ww)

        agent_in = torch.cat(
            [agent_features.reshape(b, agent_features.shape[1], -1), agent_mask[..., None]],
            dim=-1,
        )
        agent_tokens = self.agent_encoder(agent_in)
        agent_context, _ = self.agent_scene_attn(
            query=agent_tokens,
            key=encoded_tokens,
            value=encoded_tokens,
            key_padding_mask=None,
        )
        traj, traj_logits = self.traj_head(agent_context)
        ego_plan, ego_risk = self.planner_head(global_scene)
        occ = self.occ_head(scene_feat).reshape(b, self.future_steps, self.num_classes, h, w)
        flow = self.flow_head(scene_feat).reshape(b, self.future_steps, 2, h, w)
        dense_occ = self.dense_occ_head(scene_feat).reshape(b, self.future_steps, self.num_classes, h, w)
        return {
            "occ_logits": occ,
            "flow": flow,
            "dense_occ_logits": dense_occ,
            "traj": traj,
            "traj_logits": traj_logits,
            "ego_plan": ego_plan,
            "ego_risk": ego_risk,
        }


def _multimodal_traj_loss(traj: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    expanded_target = target[:, :, None].expand_as(traj)
    expanded_mask = mask[:, :, None, :, None]
    per_step = F.smooth_l1_loss(traj, expanded_target, reduction="none").sum(dim=-1, keepdim=True)
    per_mode = (per_step * expanded_mask).sum(dim=(3, 4)) / expanded_mask.sum(dim=(3, 4)).clamp_min(1.0)
    valid_agent = mask.sum(dim=-1) > 0
    best_mode = per_mode.argmin(dim=-1)
    best_loss = per_mode.gather(-1, best_mode[..., None]).squeeze(-1)
    reg_loss = (best_loss * valid_agent.float()).sum() / valid_agent.float().sum().clamp_min(1.0)
    cls_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), best_mode.reshape(-1), reduction="none")
    cls_loss = (cls_loss.reshape_as(best_mode) * valid_agent.float()).sum() / valid_agent.float().sum().clamp_min(1.0)
    return reg_loss + 0.2 * cls_loss


def _horizon_weights(length: int, device: torch.device, far_weight: float = 1.5) -> torch.Tensor:
    return torch.linspace(1.0, far_weight, length, device=device).reshape(1, length, 1, 1, 1)


def _balanced_occ_loss(logits: torch.Tensor, target: torch.Tensor, far_weight: float = 1.5) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    pos_weight = ((1.0 - pos) / pos).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    weights = _horizon_weights(target.shape[1], target.device, far_weight)
    return (bce * weights).mean()


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (2, 3, 4)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _sample_occ_at_norm_xy(occ_prob: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    b, t, _, h, w = occ_prob.shape
    grid = torch.stack([xy_norm[..., 1], xy_norm[..., 0]], dim=-1)
    grid = grid.clamp(-1.0, 1.0).reshape(b * t, -1, 1, 2)
    occ = occ_prob.max(dim=2).values.reshape(b * t, 1, h, w)
    sampled = F.grid_sample(occ, grid, mode="bilinear", align_corners=True).reshape(b, t, -1)
    return sampled


def _traj_occ_consistency_loss(pred, batch) -> torch.Tensor:
    occ_prob = torch.sigmoid(pred["occ_logits"])
    mode = pred["traj_logits"].argmax(dim=-1)
    gather_idx = mode[:, :, None, None, None].expand(-1, -1, 1, pred["traj"].shape[3], 2)
    top_traj = pred["traj"].gather(2, gather_idx).squeeze(2)
    b, agents, t, _ = top_traj.shape
    xy = top_traj.permute(0, 2, 1, 3).reshape(b, t, agents, 2)
    sampled = _sample_occ_at_norm_xy(occ_prob, xy)
    valid = batch["agent_mask"][:, None, :].float() * batch["traj_mask"].permute(0, 2, 1).float()
    return ((1.0 - sampled) * valid).sum() / valid.sum().clamp_min(1.0)


def _ego_collision_regularizer(pred, batch) -> torch.Tensor:
    if "ego_future_mask" not in batch:
        return pred["occ_logits"].new_tensor(0.0)
    occ_prob = torch.sigmoid(pred["occ_logits"])
    safest_mode = pred["ego_risk"].argmin(dim=-1)
    gather_idx = safest_mode[:, None, None, None].expand(-1, 1, pred["ego_plan"].shape[2], 2)
    ego_plan = pred["ego_plan"].gather(1, gather_idx).squeeze(1)
    xy = ego_plan[:, :, None]
    sampled = _sample_occ_at_norm_xy(occ_prob, xy).squeeze(-1)
    mask = batch["ego_future_mask"].float()
    return (sampled * mask).sum() / mask.sum().clamp_min(1.0)


def compute_losses(pred, batch):
    occ_loss = _balanced_occ_loss(pred["occ_logits"], batch["future_occ"]) + 0.5 * _dice_loss(
        pred["occ_logits"], batch["future_occ"]
    )
    occ_mask = (batch["future_occ"].sum(dim=2, keepdim=True) > 0).float()
    flow_raw = F.smooth_l1_loss(pred["flow"], batch["future_flow"], reduction="none")
    flow_weights = _horizon_weights(batch["future_flow"].shape[1], batch["future_flow"].device, 1.5)
    flow_loss = (flow_raw * occ_mask * flow_weights).sum()
    flow_loss = flow_loss / (occ_mask * flow_weights).sum().clamp_min(1.0)
    traj_loss = _multimodal_traj_loss(pred["traj"], pred["traj_logits"], batch["traj_target"], batch["traj_mask"])

    ego_loss = pred["ego_plan"].new_tensor(0.0)
    if "ego_future" in batch and "ego_future_mask" in batch:
        ego_target = batch["ego_future"][:, None].expand_as(pred["ego_plan"])
        ego_mask = batch["ego_future_mask"][:, None, :, None]
        ego_per_mode = (F.smooth_l1_loss(pred["ego_plan"], ego_target, reduction="none") * ego_mask).sum(dim=(2, 3))
        ego_per_mode = ego_per_mode / ego_mask.sum(dim=(2, 3)).clamp_min(1.0)
        ego_loss = ego_per_mode.min(dim=-1).values.mean()

    dense_occ_loss = pred["dense_occ_logits"].new_tensor(0.0)
    if "occ3d_future" in batch and "occ3d_mask" in batch and batch["occ3d_mask"].sum() > 0:
        dense_occ_loss = F.binary_cross_entropy_with_logits(pred["dense_occ_logits"], batch["occ3d_future"])

    traj_occ_loss = _traj_occ_consistency_loss(pred, batch)
    ego_collision_loss = _ego_collision_regularizer(pred, batch)
    loss = (
        occ_loss
        + 0.2 * flow_loss
        + traj_loss
        + 0.5 * ego_loss
        + 0.5 * dense_occ_loss
        + 0.05 * traj_occ_loss
        + 0.05 * ego_collision_loss
    )
    return {
        "loss": loss,
        "occ_loss": occ_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "traj_loss": traj_loss.detach(),
        "ego_loss": ego_loss.detach(),
        "dense_occ_loss": dense_occ_loss.detach(),
        "traj_occ_loss": traj_occ_loss.detach(),
        "ego_collision_loss": ego_collision_loss.detach(),
    }


@torch.no_grad()
def metrics(pred, batch):
    occ_pred = (torch.sigmoid(pred["occ_logits"]) > 0.35).float()
    occ_true = batch["future_occ"]
    inter = (occ_pred * occ_true).sum()
    union = ((occ_pred + occ_true) > 0).float().sum().clamp_min(1.0)
    iou = inter / union

    traj = pred["traj"]
    target = batch["traj_target"][:, :, None]
    mask = batch["traj_mask"][:, :, None]
    dist = ((traj - target) ** 2).sum(dim=-1).sqrt()
    ade_per_mode = (dist * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    min_ade = ade_per_mode.min(dim=-1).values
    fde_per_mode = dist[..., -1]
    min_fde = fde_per_mode.min(dim=-1).values
    valid_agent = batch["traj_mask"].sum(dim=-1) > 0
    min_ade = (min_ade * valid_agent.float()).sum() / valid_agent.float().sum().clamp_min(1.0)
    min_fde = (min_fde * valid_agent.float()).sum() / valid_agent.float().sum().clamp_min(1.0)
    occ_prob = torch.sigmoid(pred["occ_logits"])
    per_horizon_iou = []
    for idx in range(occ_true.shape[1]):
        pred_t = (occ_prob[:, idx] > 0.35).float()
        true_t = occ_true[:, idx]
        inter_t = (pred_t * true_t).sum()
        union_t = ((pred_t + true_t) > 0).float().sum().clamp_min(1.0)
        per_horizon_iou.append(inter_t / union_t)
    horizon_iou = torch.stack(per_horizon_iou)
    return {
        "occ_iou": iou,
        "occ_iou_near": horizon_iou[: max(1, len(horizon_iou) // 3)].mean(),
        "occ_iou_mid": horizon_iou[len(horizon_iou) // 3 : max(len(horizon_iou) // 3 + 1, 2 * len(horizon_iou) // 3)].mean(),
        "occ_iou_far": horizon_iou[2 * len(horizon_iou) // 3 :].mean(),
        "min_ade_norm": min_ade,
        "min_fde_norm": min_fde,
    }
