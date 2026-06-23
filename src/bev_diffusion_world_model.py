from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from bev_world_model import SensorTokenAdapter, TrafficLightEncoder, VectorMapEncoder
from sensor_encoders import MultiSensorFusionEncoder


def _same_size_max_pool(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _multi_scale_proximity(mask: torch.Tensor, kernels: tuple[int, ...] = (5, 11, 21, 41)) -> torch.Tensor:
    """Cheap differentiable map proximity proxy built from existing raster maps."""
    weights = mask.new_tensor([1.0, 0.75, 0.5, 0.25])[: len(kernels)]
    prox = mask.new_zeros(mask.shape)
    for weight, kernel in zip(weights, kernels):
        prox = torch.maximum(prox, _same_size_max_pool(mask, kernel) * weight)
    return prox.clamp(0.0, 1.0)


def build_condition_image(
    past_bev: torch.Tensor,
    map_bev: torch.Tensor,
    enhanced_map_condition: bool = False,
) -> torch.Tensor:
    """Build raster conditions from safe inputs only.

    Enhanced channels are derived from current preprocessed tensors, so no raw
    WOMD files or future labels are required.
    """
    b, hist, cls, h, w = past_bev.shape
    base = [past_bev.reshape(b, hist * cls, h, w), map_bev]
    if not enhanced_map_condition:
        return torch.cat(base, dim=1)

    lane = map_bev[:, 0:1]
    road_line = map_bev[:, 1:2]
    road_edge = map_bev[:, 2:3]
    lane_proximity = _multi_scale_proximity(lane)
    edge_proximity = _multi_scale_proximity(road_edge)
    drivable_prior = torch.maximum(lane_proximity, _multi_scale_proximity(road_line, kernels=(5, 9, 17, 33)) * 0.35)
    distance_to_lane = 1.0 - lane_proximity
    distance_to_road_edge = 1.0 - edge_proximity

    occ_hist = past_bev.sum(dim=2, keepdim=False).clamp(0.0, 1.0)
    time_weights = torch.linspace(0.2, 1.0, hist, device=past_bev.device, dtype=past_bev.dtype).view(1, hist, 1, 1)
    history_motion_hint = (occ_hist * time_weights).amax(dim=1, keepdim=True)
    current_agent_occupancy = occ_hist[:, -1:].clamp(0.0, 1.0)

    enhanced = [
        drivable_prior,
        distance_to_lane,
        distance_to_road_edge,
        history_motion_hint,
        current_agent_occupancy,
    ]
    return torch.cat(base + enhanced, dim=1)


@dataclass
class DiffusionTargetConfig:
    future_steps: int
    num_classes: int = 4
    predict_flow: bool = True
    flow_scale: float = 20.0
    occ_dilate_kernel: int = 1
    flow_loss_weight: float = 0.25
    far_loss_weight: float = 2.0

    @property
    def channels_per_step(self) -> int:
        return self.num_classes + (2 if self.predict_flow else 0)

    @property
    def target_channels(self) -> int:
        return self.future_steps * self.channels_per_step


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ):
        device = device or torch.device("cpu")
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        beta = self.betas[timestep]
        alpha = self.alphas[timestep]
        alpha_bar = self.alphas_cumprod[timestep]
        coef = beta / (1.0 - alpha_bar).sqrt()
        mean = (sample - coef * noise_pred) / alpha.sqrt()
        if timestep == 0:
            return mean
        noise = torch.randn_like(sample)
        return mean + beta.sqrt() * noise


def build_ddim_scheduler(num_train_timesteps: int = 1000, prediction_type: str = "sample"):
    from diffusers import DDIMScheduler

    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_type=prediction_type,
        clip_sample=prediction_type == "sample",
        set_alpha_to_one=True,
    )


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.hidden_dim:
            emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
        return self.mlp(emb)


class FiLMConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.emb = nn.Linear(emb_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.emb(emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TemporalBEVEncoder(nn.Module):
    """Encodes BEV history with explicit temporal attention at each BEV cell."""

    def __init__(self, num_classes: int, hidden_dim: int, history_steps: int, num_heads: int):
        super().__init__()
        self.history_steps = history_steps
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_steps, hidden_dim))
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, past_bev: torch.Tensor) -> torch.Tensor:
        b, hist, cls, h, w = past_bev.shape
        frames = past_bev.reshape(b * hist, cls, h, w)
        feat = self.frame_encoder(frames).reshape(b, hist, -1, h, w)
        feat = feat.permute(0, 3, 4, 1, 2).reshape(b * h * w, hist, -1)
        feat = feat + self.temporal_pos[:, :hist]
        attn, _ = self.temporal_attn(feat, feat, feat, need_weights=False)
        feat = self.temporal_norm(feat + attn)
        latest = feat[:, -1].reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.out(latest)


class BevCrossAttentionBlock(nn.Module):
    """Cross-attends BEV features to vector map, agent and traffic-light tokens."""

    def __init__(self, channels: int, token_dim: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.token_proj = nn.Linear(token_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, bev: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = bev.shape
        if tokens.shape[1] == 0:
            return bev
        query = bev.flatten(2).transpose(1, 2)
        key_value = self.token_proj(tokens)
        attn, _ = self.attn(
            self.query_norm(query),
            key_value,
            key_value,
            key_padding_mask=token_mask,
            need_weights=False,
        )
        query = query + attn
        query = query + self.ffn(query)
        return query.transpose(1, 2).reshape(b, c, h, w)


class ConditionalBevDenoiser(nn.Module):
    """Denoises future BEV occupancy/flow conditioned on past BEV, map and actors."""

    def __init__(
        self,
        history_steps: int,
        future_steps: int,
        hidden_dim: int = 256,
        num_classes: int = 4,
        predict_flow: bool = True,
        num_heads: int = 8,
        sensor_dim: int = 256,
        map_dim: int = 8,
        traffic_dim: int = 8,
        enhanced_map_condition: bool = False,
    ):
        super().__init__()
        self.target_config = DiffusionTargetConfig(future_steps, num_classes, predict_flow)
        self.history_steps = history_steps
        self.num_classes = num_classes
        self.enhanced_map_condition = bool(enhanced_map_condition)
        self.map_condition_channels = 8 if self.enhanced_map_condition else 3
        cond_channels = history_steps * num_classes + self.map_condition_channels
        target_channels = self.target_config.target_channels
        self.time = TimeEmbedding(hidden_dim)
        self.cond_stem = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_dim, 3, stride=4, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.raster_condition = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.temporal_bev = TemporalBEVEncoder(num_classes, hidden_dim, history_steps, num_heads)
        self.agent_encoder = nn.Sequential(
            nn.Linear(history_steps * 8 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.map_encoder = VectorMapEncoder(map_dim, hidden_dim, num_heads)
        self.traffic_encoder = TrafficLightEncoder(traffic_dim, hidden_dim, num_heads)
        self.sensor_adapter = SensorTokenAdapter(sensor_dim, hidden_dim)
        self.raw_sensor_encoder = MultiSensorFusionEncoder(hidden_dim=hidden_dim, num_heads=num_heads)
        self.cond_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        in_channels = target_channels
        self.down1 = FiLMConvBlock(in_channels, hidden_dim, hidden_dim)
        self.down2 = FiLMConvBlock(hidden_dim, hidden_dim * 2, hidden_dim)
        self.mid = FiLMConvBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim)
        self.cross_attn = BevCrossAttentionBlock(hidden_dim * 2, hidden_dim, num_heads)
        self.up1 = FiLMConvBlock(hidden_dim * 3, hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, target_channels, 3, padding=1),
        )

    def encode_condition_tokens(
        self,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = agent_features.shape[0]
        agent_in = torch.cat([agent_features.reshape(b, agent_features.shape[1], -1), agent_mask[..., None]], dim=-1)
        agent_tokens = self.agent_encoder(agent_in)
        agent_key_mask = agent_mask < 0.5
        token_list = [agent_tokens]
        mask_list = [agent_key_mask]

        if map_vectors is not None and map_vector_mask is not None:
            map_tokens = self.map_encoder(map_vectors, map_vector_mask)
            token_list.append(map_tokens)
            mask_list.append(map_vector_mask < 0.5)

        if traffic_lights is not None and traffic_light_mask is not None:
            traffic_tokens = self.traffic_encoder(traffic_lights, traffic_light_mask)
            token_list.append(traffic_tokens)
            mask_list.append(torch.zeros((b, traffic_tokens.shape[1]), dtype=torch.bool, device=traffic_tokens.device))

        if camera_images is not None or lidar_points is not None:
            sensor_tokens = self.raw_sensor_encoder(camera_images, lidar_points, lidar_mask)
            token_list.append(sensor_tokens)
            mask_list.append(torch.zeros((b, sensor_tokens.shape[1]), dtype=torch.bool, device=sensor_tokens.device))
        elif sensor_context is not None:
            sensor_tokens = self.sensor_adapter(sensor_context)
            token_list.append(sensor_tokens)
            mask_list.append(torch.zeros((b, sensor_tokens.shape[1]), dtype=torch.bool, device=sensor_tokens.device))

        tokens = torch.cat(token_list, dim=1)
        mask = torch.cat(mask_list, dim=1)
        empty = (~mask).float().sum(dim=1) < 0.5
        if empty.any():
            mask = mask.clone()
            tokens = tokens.clone()
            mask[empty, 0] = False
            tokens[empty, 0] = 0.0
        return tokens, mask

    def encode_condition(
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
    ) -> torch.Tensor:
        b, hist, cls, h, w = past_bev.shape
        bev = build_condition_image(past_bev, map_bev, self.enhanced_map_condition)
        bev_token = self.cond_stem(bev).flatten(1)

        tokens, token_mask = self.encode_condition_tokens(
            agent_features,
            agent_mask,
            map_vectors,
            map_vector_mask,
            traffic_lights,
            traffic_light_mask,
            sensor_context,
            camera_images,
            lidar_points,
            lidar_mask,
        )
        valid = (~token_mask).float()
        token_mean = (tokens * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        zero = bev_token.new_zeros(b, bev_token.shape[-1])
        return self.cond_fuse(torch.cat([bev_token, token_mean, zero, zero, zero], dim=-1))

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
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
    ) -> torch.Tensor:
        b, hist, cls, h, w = past_bev.shape
        cond_image = build_condition_image(past_bev, map_bev, self.enhanced_map_condition)
        cond_raster = self.raster_condition(cond_image) + self.temporal_bev(past_bev)
        cond_tokens, cond_token_mask = self.encode_condition_tokens(
            agent_features,
            agent_mask,
            map_vectors,
            map_vector_mask,
            traffic_lights,
            traffic_light_mask,
            sensor_context,
            camera_images,
            lidar_points,
            lidar_mask,
        )
        cond = self.encode_condition(
            past_bev,
            map_bev,
            agent_features,
            agent_mask,
            map_vectors,
            map_vector_mask,
            traffic_lights,
            traffic_light_mask,
            sensor_context,
            camera_images,
            lidar_points,
            lidar_mask,
        )
        emb = self.time(timesteps) + cond
        d1 = self.down1(noisy_target, emb)
        d1 = d1 + cond_raster
        d2 = self.down2(F.avg_pool2d(d1, 2), emb)
        mid = self.mid(d2, emb)
        mid = self.cross_attn(mid, cond_tokens, cond_token_mask)
        up = F.interpolate(mid, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up1(torch.cat([up, d1], dim=1), emb)
        return self.out(up)


def make_diffusion_target(batch: Dict[str, torch.Tensor], cfg: DiffusionTargetConfig) -> torch.Tensor:
    occ = batch["future_occ"]
    if cfg.occ_dilate_kernel > 1:
        kernel = int(cfg.occ_dilate_kernel)
        if kernel % 2 == 0:
            raise ValueError("occ_dilate_kernel must be odd so the BEV target stays aligned.")
        b, t, c, h, w = occ.shape
        occ = F.max_pool2d(
            occ.reshape(b * t * c, 1, h, w),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).reshape(b, t, c, h, w)
    occ = occ * 2.0 - 1.0
    pieces = [occ]
    if cfg.predict_flow:
        pieces.append((batch["future_flow"] / cfg.flow_scale).clamp(-1.0, 1.0))
    target = torch.cat(pieces, dim=2)
    b, t, c, h, w = target.shape
    return target.reshape(b, t * c, h, w)


def split_diffusion_target(target: torch.Tensor, cfg: DiffusionTargetConfig) -> Dict[str, torch.Tensor]:
    b, _, h, w = target.shape
    target = target.reshape(b, cfg.future_steps, cfg.channels_per_step, h, w)
    occ = ((target[:, :, : cfg.num_classes] + 1.0) * 0.5).clamp(0.0, 1.0)
    out = {"occ_probs": occ, "occ_logits": torch.logit(occ.clamp(1e-4, 1.0 - 1e-4))}
    if cfg.predict_flow:
        out["flow"] = target[:, :, cfg.num_classes : cfg.num_classes + 2] * cfg.flow_scale
    return out


def diffusion_loss(prediction: torch.Tensor, target: torch.Tensor, target_cfg: DiffusionTargetConfig) -> torch.Tensor:
    b, _, h, w = target.shape
    pred = prediction.reshape(b, target_cfg.future_steps, target_cfg.channels_per_step, h, w)
    true = target.reshape_as(pred)
    horizon_weight = torch.linspace(
        1.0,
        float(target_cfg.far_loss_weight),
        target_cfg.future_steps,
        device=target.device,
        dtype=target.dtype,
    ).view(1, target_cfg.future_steps, 1, 1, 1)
    occ_mse = (pred[:, :, : target_cfg.num_classes] - true[:, :, : target_cfg.num_classes]) ** 2
    occ_loss = (occ_mse * horizon_weight).mean()
    if not target_cfg.predict_flow:
        return occ_loss
    flow_mse = (pred[:, :, target_cfg.num_classes :] - true[:, :, target_cfg.num_classes :]) ** 2
    flow_loss = (flow_mse * horizon_weight).mean()
    return occ_loss + target_cfg.flow_loss_weight * flow_loss


@torch.no_grad()
def sample_bev_diffusion(
    model: ConditionalBevDenoiser,
    scheduler,
    batch: Dict[str, torch.Tensor],
    target_cfg: DiffusionTargetConfig,
    num_inference_steps: int = 50,
) -> Dict[str, torch.Tensor]:
    device = batch["past_bev"].device
    b, _, _, h, w = batch["past_bev"].shape
    sample = torch.randn((b, target_cfg.target_channels, h, w), device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)
    for timestep in scheduler.timesteps:
        t = torch.full((b,), int(timestep), device=device, dtype=torch.long)
        noise_pred = model(
            sample,
            t,
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
        sample = scheduler.step(noise_pred, timestep, sample).prev_sample
    return split_diffusion_target(sample, target_cfg)
