from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from src.world_model.config import (
    WorldModelConfig,
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from src.world_model.data import build_world_model_data
from src.world_model.factory import build_world_model
from src.world_model.run_paths import WorldModelRunPaths


@dataclass
class WorldModelBenchmarkConfig:
    run_name: str = "lewm-benchmark"
    data: WorldModelDataConfig = field(default_factory=WorldModelDataConfig)
    model: WorldModelModelConfig = field(default_factory=WorldModelModelConfig)
    optimizer: WorldModelOptimizerConfig = field(default_factory=WorldModelOptimizerConfig)
    training: WorldModelTrainLoopConfig = field(default_factory=WorldModelTrainLoopConfig)
    batch_sizes: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    chunk_lengths: list[int] = field(default_factory=lambda: [8, 16, 32])
    warmup_batches: int = 2
    measure_batches: int = 10


@dataclass(frozen=True)
class WorldModelBenchmarkResult:
    batch_size: int
    chunk_length: int
    status: str
    warmup_batches: int
    measured_batches: int
    measured_samples: int
    elapsed_seconds: float | None
    batches_per_second: float | None
    samples_per_second: float | None
    tokens_per_second: float | None
    peak_memory_mb: float | None
    last_loss: float | None
    error_message: str | None = None


@dataclass(frozen=True)
class WorldModelBenchmarkSummary:
    run_dir: str
    config_path: str
    json_path: str
    csv_path: str
    device: str
    obs_shape: tuple[int, int, int] | None
    results: tuple[WorldModelBenchmarkResult, ...]


def _is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _cleanup_device(device: torch.device) -> None:
    gc.collect()
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_device(device: torch.device) -> None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return None


def _iter_batches(loader, count: int):
    produced = 0
    iterator = iter(loader)
    while produced < count:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        yield batch
        produced += 1


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message and "memory" in message


def _run_single_benchmark(
    cfg: WorldModelBenchmarkConfig,
    *,
    batch_size: int,
    chunk_length: int,
) -> tuple[WorldModelBenchmarkResult, tuple[int, int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if cfg.measure_batches <= 0:
        raise ValueError("measure_batches must be positive")
    if cfg.warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")

    run_cfg = WorldModelConfig(
        run_name=cfg.run_name,
        data=replace(cfg.data, batch_size=batch_size, chunk_length=chunk_length),
        model=cfg.model,
        optimizer=cfg.optimizer,
        training=cfg.training,
    )
    data_artifacts = build_world_model_data(run_cfg.data)
    artifacts = build_world_model(
        run_cfg,
        obs_shape=data_artifacts.obs_shape,
        device=run_cfg.training.device,
    )
    device = torch.device(run_cfg.training.device)
    model = artifacts.model
    optimizer = artifacts.optimizer
    last_loss: float | None = None

    try:
        model.train(True)
        for batch in _iter_batches(data_artifacts.train_loader, cfg.warmup_batches):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics = model.loss(batch, sigreg_weight=cfg.training.sigreg_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        _cleanup_device(device)
        _reset_peak_memory(device)
        _sync_device(device)

        measured_samples = 0
        start = time.perf_counter()
        for batch in _iter_batches(data_artifacts.train_loader, cfg.measure_batches):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics = model.loss(batch, sigreg_weight=cfg.training.sigreg_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.max_grad_norm)
            optimizer.step()
            measured_samples += int(batch["obs"].shape[0])
            last_loss = float(loss.detach().cpu().item())
        _sync_device(device)
        elapsed_seconds = time.perf_counter() - start
        peak_memory_mb = _peak_memory_mb(device)
        batches_per_second = cfg.measure_batches / elapsed_seconds if elapsed_seconds > 0 else None
        samples_per_second = measured_samples / elapsed_seconds if elapsed_seconds > 0 else None
        tokens_per_second = (
            (measured_samples * chunk_length) / elapsed_seconds if elapsed_seconds > 0 else None
        )
        result = WorldModelBenchmarkResult(
            batch_size=batch_size,
            chunk_length=chunk_length,
            status="ok",
            warmup_batches=cfg.warmup_batches,
            measured_batches=cfg.measure_batches,
            measured_samples=measured_samples,
            elapsed_seconds=elapsed_seconds,
            batches_per_second=batches_per_second,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            peak_memory_mb=peak_memory_mb,
            last_loss=last_loss,
        )
        return result, data_artifacts.obs_shape
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        _cleanup_device(device)
        result = WorldModelBenchmarkResult(
            batch_size=batch_size,
            chunk_length=chunk_length,
            status="oom",
            warmup_batches=cfg.warmup_batches,
            measured_batches=0,
            measured_samples=0,
            elapsed_seconds=None,
            batches_per_second=None,
            samples_per_second=None,
            tokens_per_second=None,
            peak_memory_mb=_peak_memory_mb(device),
            last_loss=last_loss,
            error_message=str(exc),
        )
        return result, data_artifacts.obs_shape
    finally:
        del model
        del optimizer
        del artifacts
        del data_artifacts
        _cleanup_device(device)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, results: list[WorldModelBenchmarkResult]) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(WorldModelBenchmarkResult(
        batch_size=0,
        chunk_length=0,
        status="ok",
        warmup_batches=0,
        measured_batches=0,
        measured_samples=0,
        elapsed_seconds=None,
        batches_per_second=None,
        samples_per_second=None,
        tokens_per_second=None,
        peak_memory_mb=None,
        last_loss=None,
        error_message=None,
    )).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def benchmark_world_model(cfg: WorldModelBenchmarkConfig) -> WorldModelBenchmarkSummary:
    if not cfg.data.dataset_paths:
        raise ValueError("Provide at least one dataset path.")
    if not cfg.batch_sizes:
        raise ValueError("Provide at least one batch size.")
    if not cfg.chunk_lengths:
        raise ValueError("Provide at least one chunk length.")

    run_paths = WorldModelRunPaths(cfg.run_name)
    run_paths.ensure_dirs()
    config_path = run_paths.run_dir / "benchmark_config.json"
    json_path = run_paths.artifacts_dir / "benchmark_results.json"
    csv_path = run_paths.artifacts_dir / "benchmark_results.csv"
    _write_json(config_path, asdict(cfg))

    results: list[WorldModelBenchmarkResult] = []
    obs_shape: tuple[int, int, int] | None = None
    for chunk_length in cfg.chunk_lengths:
        for batch_size in cfg.batch_sizes:
            result, current_obs_shape = _run_single_benchmark(
                cfg,
                batch_size=batch_size,
                chunk_length=chunk_length,
            )
            if obs_shape is None:
                obs_shape = current_obs_shape
            results.append(result)

    payload = {
        "run_name": cfg.run_name,
        "run_dir": str(run_paths.run_dir),
        "device": cfg.training.device,
        "obs_shape": list(obs_shape) if obs_shape is not None else None,
        "results": [asdict(result) for result in results],
    }
    _write_json(json_path, payload)
    _write_csv(csv_path, results)
    return WorldModelBenchmarkSummary(
        run_dir=str(run_paths.run_dir),
        config_path=str(config_path),
        json_path=str(json_path),
        csv_path=str(csv_path),
        device=cfg.training.device,
        obs_shape=obs_shape,
        results=tuple(results),
    )
