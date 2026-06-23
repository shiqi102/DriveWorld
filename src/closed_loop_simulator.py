from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClosedLoopConfig:
    collision_threshold_m: float = 2.0
    norm_to_meter: float = 80.0


@torch.no_grad()
def select_lowest_risk_plan(pred: dict) -> torch.Tensor:
    """Select one ego plan from model output.

    Returns:
        [B, T, 2] normalized BEV coordinates.
    """
    risk = pred["ego_risk"]
    best = risk.argmin(dim=-1)
    return pred["ego_plan"][torch.arange(pred["ego_plan"].shape[0], device=risk.device), best]


@torch.no_grad()
def estimate_plan_collision(plan: torch.Tensor, agent_traj: torch.Tensor, agent_mask: torch.Tensor, cfg: ClosedLoopConfig):
    """Simple collision proxy between selected ego plan and predicted agents.

    Args:
        plan: [B, T, 2]
        agent_traj: [B, A, K, T, 2]
        agent_mask: [B, A]
    """
    best_agent_mode = agent_traj[:, :, 0]
    dist = ((plan[:, None] - best_agent_mode) * cfg.norm_to_meter).pow(2).sum(dim=-1).sqrt()
    collision = (dist < cfg.collision_threshold_m) & (agent_mask[..., None] > 0.5)
    return collision.any(dim=(1, 2)).float()


@torch.no_grad()
def rollout_one_step_world_model(model, batch: dict, cfg: ClosedLoopConfig | None = None):
    """One-step closed-loop hook.

    This is not a full vehicle simulator yet. It is the integration point where
    the planner consumes predicted occupancy/agents and chooses an ego rollout.
    """
    cfg = cfg or ClosedLoopConfig()
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
    )
    plan = select_lowest_risk_plan(pred)
    collision = estimate_plan_collision(plan, pred["traj"], batch["agent_mask"], cfg)
    return {"pred": pred, "selected_plan": plan, "collision_proxy": collision}
