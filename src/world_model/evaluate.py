from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
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
from src.world_model.runtime import autocast_context, build_grad_scaler

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
    cosine_similarity: float
    latent_rmse: float
    explained_variance: float
    top1_retrieval: float
    top5_retrieval: float
    avg_fetch_ms: float
    avg_transfer_ms: float
    avg_step_ms: float
    batches: int
    action_metrics: dict[str, dict[str, float]]
    route_metrics: dict[str, dict[str, float]]


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


@dataclass
class _MetricAccumulator:
    count: int = 0
    pred_loss_sum: float = 0.0
    cosine_sum: float = 0.0
    squared_error_sum: float = 0.0
    target_sq_sum: float = 0.0
    top1_hits: int = 0
    top5_hits: int = 0

    def update(
        self,
        *,
        pred_loss_sum: float,
        cosine_sum: float,
        squared_error_sum: float,
        target_sq_sum: float,
        top1_hits: int,
        top5_hits: int,
        count: int,
    ) -> None:
        self.count += count
        self.pred_loss_sum += pred_loss_sum
        self.cosine_sum += cosine_sum
        self.squared_error_sum += squared_error_sum
        self.target_sq_sum += target_sq_sum
        self.top1_hits += top1_hits
        self.top5_hits += top5_hits

    def to_summary(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "count": 0.0,
                "pred_loss": 0.0,
                "cosine_similarity": 0.0,
                "latent_rmse": 0.0,
                "explained_variance": 0.0,
                "top1_retrieval": 0.0,
                "top5_retrieval": 0.0,
            }
        explained_variance = 1.0 - (self.squared_error_sum / max(self.target_sq_sum, 1e-12))
        return {
            "count": float(self.count),
            "pred_loss": float(self.pred_loss_sum / self.count),
            "cosine_similarity": float(self.cosine_sum / self.count),
            "latent_rmse": float((self.squared_error_sum / self.count) ** 0.5),
            "explained_variance": float(explained_variance),
            "top1_retrieval": float(self.top1_hits / self.count),
            "top5_retrieval": float(self.top5_hits / self.count),
        }


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


def _dominant_route_labels(batch: dict[str, Any]) -> torch.Tensor:
    fractions = torch.stack(
        [
            batch["straight_fraction"],
            batch["left_turn_fraction"],
            batch["right_turn_fraction"],
        ],
        dim=-1,
    )
    return torch.argmax(fractions, dim=-1)


def _retrieval_hits(pred: torch.Tensor, target: torch.Tensor, *, k: int) -> torch.Tensor:
    similarities = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).transpose(0, 1)
    topk_indices = similarities.topk(k=min(k, similarities.size(1)), dim=1).indices
    truth = torch.arange(similarities.size(0), device=similarities.device).unsqueeze(1)
    return (topk_indices == truth).any(dim=1)


def _update_group_metrics(
    accumulator: dict[str, _MetricAccumulator],
    labels: torch.Tensor,
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    squared_error: torch.Tensor,
    cosine_similarity: torch.Tensor,
    top1_hits: torch.Tensor,
    top5_hits: torch.Tensor,
    label_names: dict[int, str],
) -> None:
    for label_value, label_name in label_names.items():
        mask = labels == label_value
        if not torch.any(mask):
            continue
        pred_group = pred[mask]
        target_group = target[mask]
        sq_group = squared_error[mask]
        cosine_group = cosine_similarity[mask]
        top1_group = top1_hits[mask]
        top5_group = top5_hits[mask]
        accumulator.setdefault(label_name, _MetricAccumulator()).update(
            pred_loss_sum=float(F.mse_loss(pred_group, target_group, reduction="sum").detach().cpu()),
            cosine_sum=float(cosine_group.sum().detach().cpu()),
            squared_error_sum=float(sq_group.sum().detach().cpu()),
            target_sq_sum=float(target_group.pow(2).sum().detach().cpu()),
            top1_hits=int(top1_group.sum().detach().cpu()),
            top5_hits=int(top5_group.sum().detach().cpu()),
            count=int(mask.sum().detach().cpu()),
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
    cosine_scores: list[float] = []
    latent_rmses: list[float] = []
    explained_variances: list[float] = []
    top1_scores: list[float] = []
    top5_scores: list[float] = []
    fetch_times: list[float] = []
    transfer_times: list[float] = []
    step_times: list[float] = []
    action_metrics: dict[str, _MetricAccumulator] = {}
    route_metrics: dict[str, _MetricAccumulator] = {}
    route_names = {0: "straight", 1: "left", 2: "right"}

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
        with torch.no_grad():
            with autocast_context(cfg.training, device):
                outputs = model.forward(batch["obs"], batch["action"], batch["next_obs"])
                pred_loss = F.mse_loss(outputs["pred_latents"], outputs["target_latents"])
                reg_loss = model.regularizer(outputs["regularizer_latents"])
                total_loss = pred_loss + cfg.training.sigreg_weight * reg_loss
        _sync_device(device)
        step_time = time.perf_counter() - step_start

        pred_flat = outputs["pred_latents"].reshape(-1, outputs["pred_latents"].size(-1)).float()
        target_flat = outputs["target_latents"].reshape(-1, outputs["target_latents"].size(-1)).float()
        squared_error = (pred_flat - target_flat).pow(2).sum(dim=-1)
        cosine_similarity = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        top1_hits = _retrieval_hits(pred_flat, target_flat, k=1)
        top5_hits = _retrieval_hits(pred_flat, target_flat, k=5)
        batch_pred_loss = float(pred_loss.detach().cpu())
        batch_reg_loss = float(reg_loss.detach().cpu())
        batch_total_loss = float(total_loss.detach().cpu())
        batch_rmse = float(torch.sqrt(squared_error.mean()).detach().cpu())
        batch_target_sq = float(target_flat.pow(2).sum().detach().cpu())
        batch_sq_sum = float(squared_error.sum().detach().cpu())
        batch_explained_variance = 1.0 - (batch_sq_sum / max(batch_target_sq, 1e-12))
        batch_cosine = float(cosine_similarity.mean().detach().cpu())
        batch_top1 = float(top1_hits.float().mean().detach().cpu())
        batch_top5 = float(top5_hits.float().mean().detach().cpu())

        action_labels = batch["action"].reshape(-1).long()
        route_labels = _dominant_route_labels(batch).reshape(-1).long()
        unique_actions = torch.unique(action_labels).tolist()
        action_names = {int(action_id): f"action_{int(action_id)}" for action_id in unique_actions}
        _update_group_metrics(
            action_metrics,
            action_labels,
            pred=pred_flat,
            target=target_flat,
            squared_error=squared_error,
            cosine_similarity=cosine_similarity,
            top1_hits=top1_hits,
            top5_hits=top5_hits,
            label_names=action_names,
        )
        _update_group_metrics(
            route_metrics,
            route_labels,
            pred=pred_flat,
            target=target_flat,
            squared_error=squared_error,
            cosine_similarity=cosine_similarity,
            top1_hits=top1_hits,
            top5_hits=top5_hits,
            label_names=route_names,
        )

        losses.append(batch_total_loss)
        pred_losses.append(batch_pred_loss)
        reg_losses.append(batch_reg_loss)
        cosine_scores.append(batch_cosine)
        latent_rmses.append(batch_rmse)
        explained_variances.append(batch_explained_variance)
        top1_scores.append(batch_top1)
        top5_scores.append(batch_top5)
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
                    loss=batch_total_loss,
                    pred_loss=batch_pred_loss,
                    reg_loss=batch_reg_loss,
                    cosine=batch_cosine,
                    latent_rmse=batch_rmse,
                    top1=batch_top1,
                    top5=batch_top5,
                    fetch_ms=mean(fetch_times[recent_slice]) * 1000.0,
                    transfer_ms=mean(transfer_times[recent_slice]) * 1000.0,
                    step_ms=mean(step_times[recent_slice]) * 1000.0,
                )
            )

    if not losses:
        return WorldModelEvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, {}, {})
    return WorldModelEvalMetrics(
        loss=float(mean(losses)),
        pred_loss=float(mean(pred_losses)),
        reg_loss=float(mean(reg_losses)),
        cosine_similarity=float(mean(cosine_scores)),
        latent_rmse=float(mean(latent_rmses)),
        explained_variance=float(mean(explained_variances)),
        top1_retrieval=float(mean(top1_scores)),
        top5_retrieval=float(mean(top5_scores)),
        avg_fetch_ms=float(mean(fetch_times) * 1000.0),
        avg_transfer_ms=float(mean(transfer_times) * 1000.0),
        avg_step_ms=float(mean(step_times) * 1000.0),
        batches=len(loader),
        action_metrics={name: metric.to_summary() for name, metric in sorted(action_metrics.items())},
        route_metrics={name: metric.to_summary() for name, metric in sorted(route_metrics.items())},
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
            val_cosine=val_metrics.cosine_similarity,
            val_rmse=val_metrics.latent_rmse,
            val_top1=val_metrics.top1_retrieval,
            val_top5=val_metrics.top5_retrieval,
            output_path=output_path,
        )
    )
    return result
