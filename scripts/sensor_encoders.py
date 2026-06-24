from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CameraImageEncoder(nn.Module):
    """Lightweight multi-camera encoder.

    Input:
        images: [B, num_cameras, 3, H, W]
    Output:
        camera tokens: [B, num_cameras, hidden_dim]

    In a larger run this class can be replaced with ResNet/Swin/ViT pretrained
    encoders while keeping the same output contract.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        feat = self.backbone(images.reshape(b * n, c, h, w))
        feat = self.pool(feat).flatten(1)
        return feat.reshape(b, n, -1)


class LidarPointEncoder(nn.Module):
    """Point-level lidar encoder.

    Input:
        points: [B, num_points, point_dim], usually xyz/intensity/time/ring
        point_mask: [B, num_points]
    Output:
        lidar token: [B, 1, hidden_dim]
    """

    def __init__(self, point_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.point_mlp(points)
        mask = point_mask[..., None].float()
        pooled = (tokens * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled


class MultiSensorFusionEncoder(nn.Module):
    """Camera/lidar fusion front-end for future raw-sensor training."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, point_dim: int = 5):
        super().__init__()
        self.camera_encoder = CameraImageEncoder(hidden_dim)
        self.lidar_encoder = LidarPointEncoder(point_dim, hidden_dim)
        self.type_embedding = nn.Embedding(2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)

    def forward(
        self,
        images: torch.Tensor | None = None,
        lidar_points: torch.Tensor | None = None,
        lidar_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = []
        if images is not None:
            cam = self.camera_encoder(images)
            cam = cam + self.type_embedding.weight[0].reshape(1, 1, -1)
            tokens.append(cam)
        if lidar_points is not None and lidar_mask is not None:
            lidar = self.lidar_encoder(lidar_points, lidar_mask)
            lidar = lidar + self.type_embedding.weight[1].reshape(1, 1, -1)
            tokens.append(lidar)
        if not tokens:
            raise ValueError("At least one of images or lidar_points must be provided.")
        return self.fusion(torch.cat(tokens, dim=1))


def build_zero_sensor_context(batch_size: int, sensor_dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch_size, 1, sensor_dim), device=device)
