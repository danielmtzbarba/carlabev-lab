from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import nn

from src.utils.common_logging import add_file_handler, event_message, get_logger
from src.world_model.config import (
    WorldModelConfig,
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from src.world_model.data import build_world_model_data
from src.world_model.factory import build_world_model
from src.world_model.runtime import build_grad_scaler, run_world_model_step

LOGGER = get_logger("world_model.evaluate")


@dataclass
class WorldModelCheckpointEvalConfig:
    checkpoint_path: str
    dataset_paths: list[str] | None = None
    device: str | None = None
    batch_size: int | None = None
    include_train_split: bool = False
    output_path: str | None = None


@dataclass(frozen=True)
class WorldModelEvalMetrics:
    loss: float
    pred_loss: float
    reg_loss: float
    avg_fetch_ms: float
    avg_transfer_ms: float
    avg_step_ms: float
    batches: int


@dataclass(frozen=True)
class WorldModelCheckpointEvalResult:
    checkpoint_path: str
    output_path: str
    dataset_paths: tuple[str, ...]
    source_names: tuple[str, ...]
    device: str
    run_dir: str | None
    chunk_length: int
    batch_size: int
    train_metrics: WorldModelEvalMetrics | None
    val_metrics: WorldModelEvalMetrics


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _batch_position(index: int, total: int) -> str:
    width = max(2, len(str(max(total, 1))))
    return f"{index:0{width}d}/{total:0{width}d}"


def _config_from_payload(payload: dict[str, Any]) -> WorldModelConfig:
    return WorldModelConfig(
        run_name=payload["run_name"],
        data=WorldModelDataConfig(**payload["data"]),
        model=WorldModelModelConfig(**payload["model"]),
        optimizer=WorldModelOptimizerConfig(**payload["optimizer"]),
        training=WorldModelTrainLoopConfig(**payload["training"]),
    )


def _evaluate_loader(
    model: nn.Module,
    loader,
    *,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: WorldModelConfig,
    phase: str,
) -> WorldModelEvalMetrics:
    model.train(False)
    losses: list[float] = []
    pred_losses: list[float] = []
    reg_losses: list[float] = []
    fetch_times: list[float] = []
    transfer_times: list[float] = []
    step_times: list[float] = []

    iterator = iter(loader)
    for batch_index in range(len(loader)):
        fetch_start = time.perf_counter()
        batch = next(iterator)
        fetch_time = time.perf_counter() - fetch_start

        transfer_start = time.perf_counter()
        batch = _to_device(batch, device)
        _sync_device(device)
        transfer_time = time.perf_counter() - transfer_start

        step_start = time.perf_counter()
        _loss, metrics = run_world_model_step(
            model,
            batch,
            optimizer=None,
            scaler=scaler,
            device=device,
            training_cfg=cfg.training,
            optimizer_cfg=cfg.optimizer,
        )
        _sync_device(device)
        step_time = time.perf_counter() - step_start

        losses.append(metrics["loss"])
        pred_losses.append(metrics["pred_loss"])
        reg_losses.append(metrics["reg_loss"])
        fetch_times.append(fetch_time)
        transfer_times.append(transfer_time)
        step_times.append(step_time)

        if (
            cfg.training.timing_log_interval > 0
            and ((batch_index + 1) % cfg.training.timing_log_interval == 0 or (batch_index + 1) == len(loader))
        ):
            recent_slice = slice(max(0, len(step_times) - cfg.training.timing_log_interval), len(step_times))
            LOGGER.info(
                event_message(
                    "EVAL_WM",
                    phase,
                    batch=_batch_position(batch_index + 1, len(loader)),
                    loss=metrics["loss"],
                    pred_loss=metrics["pred_loss"],
                    reg_loss=metrics["reg_loss"],
                    fetch_ms=mean(fetch_times[recent_slice]) * 1000.0,
                    transfer_ms=mean(transfer_times[recent_slice]) * 1000.0,
                    step_ms=mean(step_times[recent_slice]) * 1000.0,
                )
            )

    if not losses:
        return WorldModelEvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return WorldModelEvalMetrics(
        loss=float(mean(losses)),
        pred_loss=float(mean(pred_losses)),
        reg_loss=float(mean(reg_losses)),
        avg_fetch_ms=float(mean(fetch_times) * 1000.0),
        avg_transfer_ms=float(mean(transfer_times) * 1000.0),
        avg_step_ms=float(mean(step_times) * 1000.0),
        batches=len(loader),
    )


def _default_output_path(checkpoint_path: Path) -> Path:
    run_dir = checkpoint_path.parent.parent
    return run_dir / "artifacts" / "eval_checkpoint.json"


def evaluate_world_model_checkpoint(cfg: WorldModelCheckpointEvalConfig) -> WorldModelCheckpointEvalResult:
    checkpoint_path = Path(cfg.checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device or "cpu")
    if "world_model_config" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain world_model_config: {checkpoint_path}")

    run_cfg = _config_from_payload(checkpoint["world_model_config"])
    if cfg.dataset_paths:
        run_cfg.data.dataset_paths = list(cfg.dataset_paths)
    if cfg.batch_size is not None:
        run_cfg.data.batch_size = cfg.batch_size
    if cfg.device is not None:
        run_cfg.training.device = cfg.device

    output_path = Path(cfg.output_path).expanduser().resolve() if cfg.output_path is not None else _default_output_path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path.parent.parent / "eval_world_model.log")

    LOGGER.info(
        event_message(
            "EVAL_WM",
            "INIT",
            checkpoint_path=checkpoint_path,
            datasets=run_cfg.data.dataset_paths,
            device=run_cfg.training.device,
            batch_size=run_cfg.data.batch_size,
            chunk_length=run_cfg.data.chunk_length,
        )
    )

    data_artifacts = build_world_model_data(run_cfg.data, device=run_cfg.training.device)
    artifacts = build_world_model(run_cfg, obs_shape=data_artifacts.obs_shape, device=run_cfg.training.device)
    model = artifacts.model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch_device = torch.device(run_cfg.training.device)
    scaler = build_grad_scaler(run_cfg.training, torch_device)

    train_metrics = None
    if cfg.include_train_split:
        LOGGER.info(event_message("EVAL_WM", "TRAIN_START", batches=len(data_artifacts.train_loader)))
        train_metrics = _evaluate_loader(
            model,
            data_artifacts.train_loader,
            scaler=scaler,
            device=torch_device,
            cfg=run_cfg,
            phase="TRAIN_BATCH",
        )
    LOGGER.info(event_message("EVAL_WM", "VAL_START", batches=len(data_artifacts.val_loader)))
    val_metrics = _evaluate_loader(
        model,
        data_artifacts.val_loader,
        scaler=scaler,
        device=torch_device,
        cfg=run_cfg,
        phase="VAL_BATCH",
    )

    result = WorldModelCheckpointEvalResult(
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        dataset_paths=tuple(run_cfg.data.dataset_paths),
        source_names=tuple(source.source_name for source in data_artifacts.indexed.sources),
        device=run_cfg.training.device,
        run_dir=str(checkpoint_path.parent.parent) if checkpoint_path.parent.name == "checkpoints" else None,
        chunk_length=run_cfg.data.chunk_length,
        batch_size=run_cfg.data.batch_size,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    LOGGER.info(
        event_message(
            "EVAL_WM",
            "DONE",
            val_loss=val_metrics.loss,
            val_pred_loss=val_metrics.pred_loss,
            val_reg_loss=val_metrics.reg_loss,
            output_path=output_path,
        )
    )
    return result
