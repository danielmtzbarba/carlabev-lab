from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch import nn

from src.utils.common_logging import add_file_handler, build_progress, get_logger, kv_message
from src.world_model.config import WorldModelConfig
from src.world_model.contracts import WorldModelSequenceConfig
from src.world_model.data import WorldModelDataArtifacts, build_world_model_data
from src.world_model.factory import WorldModelArtifacts, build_world_model
from src.world_model.runtime import build_grad_scaler, maybe_compile_model, run_world_model_step
from src.world_model.run_paths import WorldModelRunPaths
from src.world_model.validate import validate_datasets

LOGGER = get_logger("world_model.train")


@dataclass
class TrainWorldModelResult:
    run_dir: str
    checkpoint_path: str
    best_checkpoint_path: str
    train_steps: int
    epochs: int
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float


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


def _epoch_loop(
    model: nn.Module,
    loader,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: WorldModelConfig,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: list[float] = []
    pred_losses: list[float] = []
    reg_losses: list[float] = []
    fetch_times: list[float] = []
    transfer_times: list[float] = []
    step_times: list[float] = []

    if progress is not None and task_id is not None:
        progress.reset(task_id, total=max(len(loader), 1), completed=0, loss="-")
        progress.update(task_id, visible=True)

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
            optimizer=optimizer,
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
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1, loss=f"{metrics['loss']:.4f}")
        if (
            cfg.training.timing_log_interval > 0
            and ((batch_index + 1) % cfg.training.timing_log_interval == 0 or (batch_index + 1) == len(loader))
        ):
            recent_slice = slice(max(0, len(step_times) - cfg.training.timing_log_interval), len(step_times))
            LOGGER.info(
                kv_message(
                    "Batch timing",
                    phase="train" if train_mode else "val",
                    batch=f"{batch_index + 1}/{len(loader)}",
                    fetch_ms=mean(fetch_times[recent_slice]) * 1000.0,
                    transfer_ms=mean(transfer_times[recent_slice]) * 1000.0,
                    step_ms=mean(step_times[recent_slice]) * 1000.0,
                )
            )

    if not losses:
        if progress is not None and task_id is not None:
            progress.update(task_id, visible=False)
        return {
            "loss": 0.0,
            "pred_loss": 0.0,
            "reg_loss": 0.0,
            "avg_fetch_ms": 0.0,
            "avg_transfer_ms": 0.0,
            "avg_step_ms": 0.0,
        }
    epoch_metrics = {
        "loss": float(mean(losses)),
        "pred_loss": float(mean(pred_losses)),
        "reg_loss": float(mean(reg_losses)),
        "avg_fetch_ms": float(mean(fetch_times) * 1000.0),
        "avg_transfer_ms": float(mean(transfer_times) * 1000.0),
        "avg_step_ms": float(mean(step_times) * 1000.0),
    }
    if progress is not None and task_id is not None:
        progress.update(task_id, loss=f"{epoch_metrics['loss']:.4f}", visible=False)
    return epoch_metrics


def _save_checkpoint(
    checkpoint_path,
    *,
    artifacts: WorldModelArtifacts,
    epoch: int,
    best_val_loss: float,
    cfg: WorldModelConfig,
    data_artifacts: WorldModelDataArtifacts,
) -> None:
    model_to_save = getattr(artifacts.model, "_orig_mod", artifacts.model)
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": artifacts.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "world_model_config": asdict(cfg),
            "indexed_sources": [source.source_name for source in data_artifacts.indexed.sources],
        },
        checkpoint_path,
    )


def train_world_model(
    cfg: WorldModelConfig,
    *,
    show_progress: bool = True,
) -> TrainWorldModelResult:
    if not cfg.data.dataset_paths:
        raise ValueError("Provide at least one dataset path.")

    run_paths = WorldModelRunPaths(cfg.run_name)
    run_paths.ensure_dirs()
    add_file_handler(run_paths.run_dir / "train_world_model.log")
    LOGGER.info(
        kv_message(
            "Init training",
            run_name=cfg.run_name,
            device=cfg.training.device,
            epochs=cfg.training.epochs,
            amp=cfg.training.amp,
            amp_dtype=cfg.training.amp_dtype,
            compile_model=cfg.training.compile_model,
            timing_log_interval=cfg.training.timing_log_interval,
        )
    )
    LOGGER.info(kv_message("Validate datasets", paths=cfg.data.dataset_paths))
    validation_report = validate_datasets(
        cfg.data.dataset_paths,
        cfg=WorldModelSequenceConfig(
            chunk_length=cfg.data.chunk_length,
            stride=cfg.data.stride,
            val_ratio=cfg.data.val_ratio,
            expected_num_actions=cfg.data.expected_num_actions,
        ),
        chunk_lengths=(1, cfg.data.chunk_length),
        cache_sequence_indices=cfg.data.cache_sequence_indices,
        sequence_cache_dir=cfg.data.sequence_cache_dir,
    )
    run_paths.config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    run_paths.latest_pointer_path.write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "run_id": run_paths.run_id,
                "run_dir": str(run_paths.run_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_paths.artifacts_dir / "validation_report.json").write_text(
        validation_report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    LOGGER.info(
        kv_message(
            "Build datasets",
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
            prefetch_factor=cfg.data.prefetch_factor,
        )
    )
    data_artifacts = build_world_model_data(cfg.data, device=cfg.training.device)
    LOGGER.info(kv_message("Build model"))
    artifacts = build_world_model(
        cfg,
        obs_shape=data_artifacts.obs_shape,
        device=cfg.training.device,
    )
    torch_device = torch.device(cfg.training.device)
    artifacts.model = maybe_compile_model(artifacts.model, cfg.training)
    grad_scaler = build_grad_scaler(cfg.training, torch_device)

    best_val_loss = float("inf")
    final_train_loss = 0.0
    final_val_loss = 0.0
    history: list[dict[str, float | int]] = []
    train_steps = 0
    LOGGER.info(
        kv_message(
            "Start optimization",
            train_batches=len(data_artifacts.train_loader),
            val_batches=len(data_artifacts.val_loader),
        )
    )

    with build_progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
        refresh_per_second=4,
    ) as progress:
        epoch_task_id = progress.add_task(
            "World-model epochs",
            total=cfg.training.epochs,
            loss="-",
        )
        train_task_id = progress.add_task(
            "Train batches",
            total=max(len(data_artifacts.train_loader), 1),
            loss="-",
            visible=False,
        )
        val_task_id = progress.add_task(
            "Val batches",
            total=max(len(data_artifacts.val_loader), 1),
            loss="-",
            visible=False,
        )

        for epoch in range(1, cfg.training.epochs + 1):
            LOGGER.info(kv_message("Start epoch", epoch=epoch, total_epochs=cfg.training.epochs))
            progress.update(
                train_task_id,
                description=f"Train epoch {epoch}/{cfg.training.epochs}",
            )
            train_metrics = _epoch_loop(
                artifacts.model,
                data_artifacts.train_loader,
                optimizer=artifacts.optimizer,
                scaler=grad_scaler,
                device=torch_device,
                cfg=cfg,
                progress=progress,
                task_id=train_task_id,
            )
            if len(data_artifacts.val_dataset) > 0:
                progress.update(
                    val_task_id,
                    description=f"Val epoch {epoch}/{cfg.training.epochs}",
                )
                val_metrics = _epoch_loop(
                    artifacts.model,
                    data_artifacts.val_loader,
                    optimizer=None,
                    scaler=grad_scaler,
                    device=torch_device,
                    cfg=cfg,
                    progress=progress,
                    task_id=val_task_id,
                )
            else:
                val_metrics = dict(train_metrics)

            final_train_loss = train_metrics["loss"]
            final_val_loss = val_metrics["loss"]
            train_steps += len(data_artifacts.train_loader)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_pred_loss": train_metrics["pred_loss"],
                    "train_reg_loss": train_metrics["reg_loss"],
                    "train_avg_fetch_ms": train_metrics["avg_fetch_ms"],
                    "train_avg_transfer_ms": train_metrics["avg_transfer_ms"],
                    "train_avg_step_ms": train_metrics["avg_step_ms"],
                    "val_loss": val_metrics["loss"],
                    "val_pred_loss": val_metrics["pred_loss"],
                    "val_reg_loss": val_metrics["reg_loss"],
                    "val_avg_fetch_ms": val_metrics["avg_fetch_ms"],
                    "val_avg_transfer_ms": val_metrics["avg_transfer_ms"],
                    "val_avg_step_ms": val_metrics["avg_step_ms"],
                }
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                _save_checkpoint(
                    run_paths.checkpoints_dir / "world_model_best.pt",
                    artifacts=artifacts,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    cfg=cfg,
                    data_artifacts=data_artifacts,
                )
            if epoch % cfg.training.save_every == 0:
                _save_checkpoint(
                    run_paths.checkpoints_dir / f"world_model_epoch_{epoch:03d}.pt",
                    artifacts=artifacts,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    cfg=cfg,
                    data_artifacts=data_artifacts,
                )
            progress.update(
                epoch_task_id,
                advance=1,
                loss=f"train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f}",
            )
            LOGGER.info(
                kv_message(
                    "Epoch complete",
                    epoch=f"{epoch}/{cfg.training.epochs}",
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    train_fetch_ms=train_metrics["avg_fetch_ms"],
                    train_transfer_ms=train_metrics["avg_transfer_ms"],
                    train_step_ms=train_metrics["avg_step_ms"],
                    val_fetch_ms=val_metrics["avg_fetch_ms"],
                    val_transfer_ms=val_metrics["avg_transfer_ms"],
                    val_step_ms=val_metrics["avg_step_ms"],
                )
            )

    _save_checkpoint(
        run_paths.checkpoints_dir / "world_model_final.pt",
        artifacts=artifacts,
        epoch=cfg.training.epochs,
        best_val_loss=best_val_loss,
        cfg=cfg,
        data_artifacts=data_artifacts,
    )
    (run_paths.artifacts_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        kv_message(
            "Finish training",
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )
    )

    return TrainWorldModelResult(
        run_dir=str(run_paths.run_dir),
        checkpoint_path=str(run_paths.checkpoints_dir / "world_model_final.pt"),
        best_checkpoint_path=str(run_paths.checkpoints_dir / "world_model_best.pt"),
        train_steps=train_steps,
        epochs=cfg.training.epochs,
        best_val_loss=float(best_val_loss),
        final_train_loss=float(final_train_loss),
        final_val_loss=float(final_val_loss),
    )
