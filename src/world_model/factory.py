from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from src.world_model.config import WorldModelConfig
from src.world_model.jepa import LeWorldModel


@dataclass
class WorldModelArtifacts:
    model: nn.Module
    optimizer: Optimizer


def build_world_model(cfg: WorldModelConfig, *, obs_shape: tuple[int, int, int], device: str) -> WorldModelArtifacts:
    model = LeWorldModel(
        obs_shape=obs_shape,
        num_actions=cfg.data.expected_num_actions,
        context_frames=cfg.data.chunk_length,
        cfg=cfg.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    return WorldModelArtifacts(model=model, optimizer=optimizer)
