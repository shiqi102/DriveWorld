from __future__ import annotations

from pathlib import Path

import torch


class Occ3DAdapter:
    """Adapter boundary for dense occupancy supervision.

    WOMD alone gives track-derived occupancy, not dense semantic 3D occupancy.
    When Occ3D-Waymo or Occ3D-nuScenes annotations are available, convert them
    into the same future BEV contract used by the model:

        occ3d_future: [T, C, H, W]
        occ3d_mask:   [1]
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def available(self) -> bool:
        return self.root.exists() and any(self.root.rglob("*"))

    def load_bev_future(self, scenario_id: str, future_steps: int, num_classes: int, height: int, width: int):
        path = self.root / f"{scenario_id}.pt"
        if not path.exists():
            return torch.zeros(future_steps, num_classes, height, width), torch.zeros(1)
        data = torch.load(path, map_location="cpu")
        occ = data["occ3d_future"].float()
        if occ.shape != (future_steps, num_classes, height, width):
            raise ValueError(f"Unexpected Occ3D shape for {path}: {tuple(occ.shape)}")
        return occ, torch.ones(1)
