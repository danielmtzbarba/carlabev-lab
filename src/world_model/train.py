from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch import nn

from src.utils.common_logging import add_file_handler, get_logger
from src.world_model.config import WorldModelConfig
from src.world_model.contracts import WorldModelSequenceConfig
from src.world_model.data import WorldModelDataArtifacts, build_world_model_data
from src.world_model.factory import WorldModelArtifacts, build_world_model
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
    device: torch.device,
    sigreg_weight: float,
    max_grad_norm: float,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: list[float] = []
    pred_losses: list[float] = []
    reg_losses: list[float] = []

    if progress is not None and task_id is not None:
        progress.reset(task_id, total=max(len(loader), 1), completed=0, loss="-")
        progress.update(task_id, visible=True)

    for batch in loader:
        batch = _to_device(batch, device)
        if optimizer is not None:
            optimizer.zero_grad()
        loss, metrics = model.loss(batch, sigreg_weight=sigreg_weight)
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        losses.append(metrics["loss"])
        pred_losses.append(metrics["pred_loss"])
        reg_losses.append(metrics["reg_loss"])
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1, loss=f"{metrics['loss']:.4f}")

    if not losses:
        if progress is not None and task_id is not None:
            progress.update(task_id, visible=False)
        return {"loss": 0.0, "pred_loss": 0.0, "reg_loss": 0.0}
    epoch_metrics = {
        "loss": float(mean(losses)),
        "pred_loss": float(mean(pred_losses)),
        "reg_loss": float(mean(reg_losses)),
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
    torch.save(
        {
            "model_state_dict": artifacts.model.state_dict(),
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
        "Initializing world-model training run_name=%s device=%s epochs=%d",
        cfg.run_name,
        cfg.training.device,
        cfg.training.epochs,
    )
    LOGGER.info("Validating dataset paths: %s", ", ".join(cfg.data.dataset_paths))
    validation_report = validate_datasets(
        cfg.data.dataset_paths,
        cfg=WorldModelSequenceConfig(
            chunk_length=cfg.data.chunk_length,
            stride=cfg.data.stride,
            val_ratio=cfg.data.val_ratio,
            expected_num_actions=cfg.data.expected_num_actions,
        ),
        chunk_lengths=(1, cfg.data.chunk_length),
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

    LOGGER.info("Building world-model datasets and dataloaders")
    data_artifacts = build_world_model_data(cfg.data)
    LOGGER.info("Building world-model model and optimizer")
    artifacts = build_world_model(
        cfg,
        obs_shape=data_artifacts.obs_shape,
        device=cfg.training.device,
    )
    torch_device = torch.device(cfg.training.device)

    best_val_loss = float("inf")
    final_train_loss = 0.0
    final_val_loss = 0.0
    history: list[dict[str, float | int]] = []
    train_steps = 0
    LOGGER.info(
        "Starting world-model optimization train_batches=%d val_batches=%d",
        len(data_artifacts.train_loader),
        len(data_artifacts.val_loader),
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
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
            LOGGER.info("Epoch %d/%d started", epoch, cfg.training.epochs)
            progress.update(
                train_task_id,
                description=f"Train epoch {epoch}/{cfg.training.epochs}",
            )
            train_metrics = _epoch_loop(
                artifacts.model,
                data_artifacts.train_loader,
                optimizer=artifacts.optimizer,
                device=torch_device,
                sigreg_weight=cfg.training.sigreg_weight,
                max_grad_norm=cfg.optimizer.max_grad_norm,
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
                    device=torch_device,
                    sigreg_weight=cfg.training.sigreg_weight,
                    max_grad_norm=cfg.optimizer.max_grad_norm,
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
                    "val_loss": val_metrics["loss"],
                    "val_pred_loss": val_metrics["pred_loss"],
                    "val_reg_loss": val_metrics["reg_loss"],
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
                "Epoch %d/%d finished train_loss=%.6f val_loss=%.6f",
                epoch,
                cfg.training.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
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
        "Finished world-model training best_val_loss=%.6f final_train_loss=%.6f final_val_loss=%.6f",
        best_val_loss,
        final_train_loss,
        final_val_loss,
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
