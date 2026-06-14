from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from src.world_model.config import WorldModelOptimizerConfig, WorldModelTrainLoopConfig


def resolve_amp_dtype(cfg: WorldModelTrainLoopConfig) -> torch.dtype:
    if cfg.amp_dtype == "bfloat16":
        return torch.bfloat16
    if cfg.amp_dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype={cfg.amp_dtype!r}")


def amp_enabled(cfg: WorldModelTrainLoopConfig, device: torch.device) -> bool:
    return bool(cfg.amp and device.type == "cuda")


def autocast_context(cfg: WorldModelTrainLoopConfig, device: torch.device):
    if not amp_enabled(cfg, device):
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=resolve_amp_dtype(cfg),
    )


def build_grad_scaler(cfg: WorldModelTrainLoopConfig, device: torch.device) -> torch.amp.GradScaler:
    scaler_enabled = amp_enabled(cfg, device) and resolve_amp_dtype(cfg) == torch.float16
    return torch.amp.GradScaler(device.type, enabled=scaler_enabled)


def maybe_compile_model(model: nn.Module, cfg: WorldModelTrainLoopConfig) -> nn.Module:
    if not cfg.compile_model:
        return model
    return torch.compile(model, mode=cfg.compile_mode)


def run_world_model_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    training_cfg: WorldModelTrainLoopConfig,
    optimizer_cfg: WorldModelOptimizerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with autocast_context(training_cfg, device):
        loss, metrics = model.loss(batch, sigreg_weight=training_cfg.sigreg_weight)

    if optimizer is None:
        return loss, metrics
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=optimizer_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=optimizer_cfg.max_grad_norm)
        optimizer.step()
    return loss, metrics
