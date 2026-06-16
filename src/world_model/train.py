from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import torch
from torch import nn

from src.utils.common_logging import add_file_handler, event_message, get_logger
from src.world_model.config import WorldModelConfig
from src.world_model.contracts import WorldModelSequenceConfig
from src.world_model.data import WorldModelDataArtifacts, build_world_model_data
from src.world_model.evaluate import WorldModelCheckpointEvalConfig, evaluate_world_model_checkpoint
from src.world_model.factory import WorldModelArtifacts, build_world_model
from src.world_model.results_db import record_world_model_study_run
from src.world_model.runtime import build_grad_scaler, maybe_compile_model, run_world_model_step
from src.world_model.run_paths import WorldModelRunPaths
from src.world_model.staging import has_complete_prepared_shard_cache, is_tmp_dataset_path, prepare_dataset_shard_cache
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


def _batch_position(index: int, total: int) -> str:
    width = max(2, len(str(max(total, 1))))
    return f"{index:0{width}d}/{total:0{width}d}"


def _ensure_tmp_prepared_dataset_caches(dataset_paths: list[str]) -> None:
    for dataset_path in dataset_paths:
        if not is_tmp_dataset_path(dataset_path):
            continue
        if has_complete_prepared_shard_cache(dataset_path):
            LOGGER.info(event_message("TRAIN", "PREPARED_CACHE_READY", dataset_path=dataset_path))
            continue
        LOGGER.info(event_message("TRAIN", "PREPARED_CACHE_BUILD", dataset_path=dataset_path))
        prepare_dataset_shard_cache(dataset_path)


def _epoch_loop(
    model: nn.Module,
    loader,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: WorldModelConfig,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: list[float] = []
    pred_losses: list[float] = []
    reg_losses: list[float] = []
    fetch_times: list[float] = []
    transfer_times: list[float] = []
    step_times: list[float] = []

    iterator = iter(loader)
    phase = "TRAIN_BATCH" if train_mode else "VAL_BATCH"
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
        if (
            cfg.training.timing_log_interval > 0
            and ((batch_index + 1) % cfg.training.timing_log_interval == 0 or (batch_index + 1) == len(loader))
        ):
            recent_slice = slice(max(0, len(step_times) - cfg.training.timing_log_interval), len(step_times))
            LOGGER.info(
                event_message(
                    "TRAIN",
                    phase,
                    epoch=f"{epoch}/{total_epochs}",
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
        event_message(
            "TRAIN",
            "INIT",
            run_name=cfg.run_name,
            device=cfg.training.device,
            epochs=cfg.training.epochs,
            amp=cfg.training.amp,
            amp_dtype=cfg.training.amp_dtype,
            compile_model=cfg.training.compile_model,
            timing_log_interval=cfg.training.timing_log_interval,
        )
    )
    _ensure_tmp_prepared_dataset_caches(cfg.data.dataset_paths)
    LOGGER.info(event_message("TRAIN", "VALIDATE_DATASETS", paths=cfg.data.dataset_paths))
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
        event_message(
            "TRAIN",
            "BUILD_DATASETS",
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
            prefetch_factor=cfg.data.prefetch_factor,
        )
    )
    data_artifacts = build_world_model_data(cfg.data, device=cfg.training.device)
    LOGGER.info(event_message("TRAIN", "BUILD_MODEL"))
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
        event_message(
            "TRAIN",
            "OPTIMIZE_START",
            train_batches=len(data_artifacts.train_loader),
            val_batches=len(data_artifacts.val_loader),
        )
    )

    if show_progress:
        LOGGER.debug(event_message("TRAIN", "PROGRESS_DISABLED", reason="log_first_training_output"))

    for epoch in range(1, cfg.training.epochs + 1):
        LOGGER.info(event_message("TRAIN", "EPOCH_START", epoch=f"{epoch}/{cfg.training.epochs}"))
        train_metrics = _epoch_loop(
            artifacts.model,
            data_artifacts.train_loader,
            optimizer=artifacts.optimizer,
            scaler=grad_scaler,
            device=torch_device,
            cfg=cfg,
            epoch=epoch,
            total_epochs=cfg.training.epochs,
        )
        if len(data_artifacts.val_dataset) > 0:
            val_metrics = _epoch_loop(
                artifacts.model,
                data_artifacts.val_loader,
                optimizer=None,
                scaler=grad_scaler,
                device=torch_device,
                cfg=cfg,
                epoch=epoch,
                total_epochs=cfg.training.epochs,
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
        LOGGER.info(
            event_message(
                "TRAIN",
                "EPOCH_DONE",
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
        event_message(
            "TRAIN",
            "DONE",
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )
    )

    result = TrainWorldModelResult(
        run_dir=str(run_paths.run_dir),
        checkpoint_path=str(run_paths.checkpoints_dir / "world_model_final.pt"),
        best_checkpoint_path=str(run_paths.checkpoints_dir / "world_model_best.pt"),
        train_steps=train_steps,
        epochs=cfg.training.epochs,
        best_val_loss=float(best_val_loss),
        final_train_loss=float(final_train_loss),
        final_val_loss=float(final_val_loss),
    )
    if cfg.results_db_path and cfg.study_id is not None and cfg.exp_id is not None:
        LOGGER.info(event_message("TRAIN", "STUDY_EVAL_START", checkpoint=result.best_checkpoint_path))
        eval_result = evaluate_world_model_checkpoint(
            WorldModelCheckpointEvalConfig(
                checkpoint_path=result.best_checkpoint_path,
                dataset_paths=list(cfg.data.dataset_paths),
                device=cfg.training.device,
                batch_size=cfg.data.batch_size,
                include_train_split=True,
                output_path=str(run_paths.artifacts_dir / "eval_checkpoint.json"),
            )
        )
        db_path = record_world_model_study_run(
            db_path=cfg.results_db_path,
            study_id=cfg.study_id,
            exp_id=cfg.exp_id,
            seed=cfg.seed,
            experiment_name=cfg.experiment_name,
            run_name=cfg.run_name,
            train_result=result,
            eval_result=eval_result,
        )
        LOGGER.info(event_message("TRAIN", "STUDY_DB_WRITE", db_path=str(db_path)))
    return result
