from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from rich.progress import BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.utils.common_logging import add_file_handler, build_progress, get_logger, kv_message
from src.world_model.config import (
    WorldModelConfig,
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from src.world_model.data import (
    IndexedDataset,
    WorldModelDataArtifacts,
    build_dataloader,
    build_index,
    build_world_model_data_from_indexed,
)
from src.world_model.factory import build_world_model
from src.world_model.runtime import build_grad_scaler, maybe_compile_model, run_world_model_step
from src.world_model.run_paths import WorldModelRunPaths

LOGGER = get_logger("world_model.benchmark")


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


@dataclass(frozen=True)
class _ChunkDataCacheEntry:
    train_dataset: Any
    val_dataset: Any
    obs_shape: tuple[int, int, int]


def _build_results_payload(
    *,
    cfg: WorldModelBenchmarkConfig,
    run_paths: WorldModelRunPaths,
    obs_shape: tuple[int, int, int] | None,
    results: list[WorldModelBenchmarkResult],
) -> dict[str, Any]:
    return {
        "run_name": cfg.run_name,
        "run_dir": str(run_paths.run_dir),
        "device": cfg.training.device,
        "obs_shape": list(obs_shape) if obs_shape is not None else None,
        "results": [asdict(result) for result in results],
    }


def _persist_results(
    *,
    cfg: WorldModelBenchmarkConfig,
    run_paths: WorldModelRunPaths,
    json_path: Path,
    csv_path: Path,
    obs_shape: tuple[int, int, int] | None,
    results: list[WorldModelBenchmarkResult],
) -> None:
    payload = _build_results_payload(
        cfg=cfg,
        run_paths=run_paths,
        obs_shape=obs_shape,
        results=results,
    )
    _write_json(json_path, payload)
    _write_csv(csv_path, results)


def _build_artifacts_from_cached_chunk(
    indexed: IndexedDataset,
    cfg: WorldModelDataConfig,
    *,
    chunk_entry: _ChunkDataCacheEntry | None = None,
    device: str | None = None,
) -> WorldModelDataArtifacts:
    if chunk_entry is None:
        return build_world_model_data_from_indexed(indexed, cfg, device=device)

    LOGGER.info(kv_message("Reuse chunk cache", batch_size=cfg.batch_size, chunk_length=cfg.chunk_length))
    train_loader = build_dataloader(
        chunk_entry.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )
    val_loader = build_dataloader(
        chunk_entry.val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )
    LOGGER.info(
        kv_message(
            "Built cached loaders",
            train_windows=len(chunk_entry.train_dataset),
            val_windows=len(chunk_entry.val_dataset),
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            obs_shape=chunk_entry.obs_shape,
        )
    )
    return WorldModelDataArtifacts(
        train_dataset=chunk_entry.train_dataset,
        val_dataset=chunk_entry.val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        indexed=indexed,
        obs_shape=chunk_entry.obs_shape,
        num_actions=cfg.expected_num_actions,
    )


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


def _is_worker_crash(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "dataloader worker" in message or "exited unexpectedly" in message


def _run_single_benchmark(
    cfg: WorldModelBenchmarkConfig,
    *,
    batch_size: int,
    chunk_length: int,
    indexed: IndexedDataset,
    chunk_entry: _ChunkDataCacheEntry | None = None,
    progress=None,
    task_id: int | None = None,
    batch_task_id: int | None = None,
) -> tuple[WorldModelBenchmarkResult, tuple[int, int, int], _ChunkDataCacheEntry | None]:
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
    LOGGER.info(
        kv_message(
            "Start benchmark",
            chunk_length=chunk_length,
            batch_size=batch_size,
            device=cfg.training.device,
        )
    )
    if progress is not None and task_id is not None:
        progress.update(
            task_id,
            candidate=f"chunk={chunk_length} batch={batch_size}",
            stage="preparing loaders",
        )
        if batch_task_id is not None:
            progress.update(
                batch_task_id,
                description="Benchmark batches",
                total=1,
                completed=0,
                visible=False,
                candidate=f"chunk={chunk_length} batch={batch_size}",
                stage="idle",
                phase="-",
            )
    LOGGER.info(kv_message("Prepare datasets"))
    data_artifacts = _build_artifacts_from_cached_chunk(
        indexed,
        run_cfg.data,
        chunk_entry=chunk_entry,
        device=run_cfg.training.device,
    )
    if progress is not None and task_id is not None:
        progress.update(task_id, stage="building model")
    LOGGER.info(
        kv_message(
            "Build model",
            obs_shape=data_artifacts.obs_shape,
            train_batches=len(data_artifacts.train_loader),
        )
    )
    artifacts = build_world_model(
        run_cfg,
        obs_shape=data_artifacts.obs_shape,
        device=run_cfg.training.device,
    )
    device = torch.device(run_cfg.training.device)
    artifacts.model = maybe_compile_model(artifacts.model, run_cfg.training)
    grad_scaler = build_grad_scaler(run_cfg.training, device)
    model = artifacts.model
    optimizer = artifacts.optimizer
    last_loss: float | None = None

    try:
        model.train(True)
        if cfg.warmup_batches > 0:
            if progress is not None and task_id is not None:
                progress.update(task_id, stage=f"warmup {cfg.warmup_batches} batches")
                if batch_task_id is not None:
                    progress.reset(
                        batch_task_id,
                        total=max(cfg.warmup_batches, 1),
                        completed=0,
                        visible=True,
                        candidate=f"chunk={chunk_length} batch={batch_size}",
                        stage=f"warmup {cfg.warmup_batches} batches",
                        phase="warmup",
                    )
            LOGGER.info(kv_message("Warmup", batches=cfg.warmup_batches))
        for batch in _iter_batches(data_artifacts.train_loader, cfg.warmup_batches):
            batch = _to_device(batch, device)
            loss, _metrics = run_world_model_step(
                model,
                batch,
                optimizer=optimizer,
                scaler=grad_scaler,
                device=device,
                training_cfg=run_cfg.training,
                optimizer_cfg=run_cfg.optimizer,
            )
            last_loss = float(loss.detach().cpu().item())
            if progress is not None and batch_task_id is not None:
                progress.update(batch_task_id, advance=1)

        _cleanup_device(device)
        _reset_peak_memory(device)
        _sync_device(device)

        measured_samples = 0
        if progress is not None and task_id is not None:
            progress.update(task_id, stage=f"measuring {cfg.measure_batches} batches")
            if batch_task_id is not None:
                progress.reset(
                    batch_task_id,
                    total=max(cfg.measure_batches, 1),
                    completed=0,
                    visible=True,
                    candidate=f"chunk={chunk_length} batch={batch_size}",
                    stage=f"measuring {cfg.measure_batches} batches",
                    phase="measure",
                )
        LOGGER.info(kv_message("Measure", batches=cfg.measure_batches))
        start = time.perf_counter()
        for batch in _iter_batches(data_artifacts.train_loader, cfg.measure_batches):
            batch = _to_device(batch, device)
            loss, _metrics = run_world_model_step(
                model,
                batch,
                optimizer=optimizer,
                scaler=grad_scaler,
                device=device,
                training_cfg=run_cfg.training,
                optimizer_cfg=run_cfg.optimizer,
            )
            measured_samples += int(batch["obs"].shape[0])
            last_loss = float(loss.detach().cpu().item())
            if progress is not None and batch_task_id is not None:
                progress.update(batch_task_id, advance=1)
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
        LOGGER.info(
            kv_message(
                "Finish benchmark",
                chunk_length=chunk_length,
                batch_size=batch_size,
                status="ok",
                samples_per_second=samples_per_second or 0.0,
                tokens_per_second=tokens_per_second or 0.0,
                peak_memory_mb="-" if peak_memory_mb is None else f"{peak_memory_mb:.1f}",
            )
        )
        if progress is not None and task_id is not None:
            progress.update(task_id, stage="done")
            if batch_task_id is not None:
                progress.update(
                    batch_task_id,
                    visible=False,
                    candidate=f"chunk={chunk_length} batch={batch_size}",
                    stage="done",
                    phase="done",
                )
        built_chunk_entry = None
        if chunk_entry is None:
            built_chunk_entry = _ChunkDataCacheEntry(
                train_dataset=data_artifacts.train_dataset,
                val_dataset=data_artifacts.val_dataset,
                obs_shape=data_artifacts.obs_shape,
            )
        return result, data_artifacts.obs_shape, built_chunk_entry
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
        LOGGER.warning(
            "Benchmark OOM chunk_length=%d batch_size=%d error=%s",
            chunk_length,
            batch_size,
            exc,
        )
        if progress is not None and task_id is not None:
            progress.update(task_id, stage="oom")
            if batch_task_id is not None:
                progress.update(
                    batch_task_id,
                    visible=False,
                    candidate=f"chunk={chunk_length} batch={batch_size}",
                    stage="oom",
                    phase="oom",
                )
        built_chunk_entry = None
        if chunk_entry is None:
            built_chunk_entry = _ChunkDataCacheEntry(
                train_dataset=data_artifacts.train_dataset,
                val_dataset=data_artifacts.val_dataset,
                obs_shape=data_artifacts.obs_shape,
            )
        return result, data_artifacts.obs_shape, built_chunk_entry
    except RuntimeError as exc:
        if not _is_worker_crash(exc):
            raise
        _cleanup_device(device)
        result = WorldModelBenchmarkResult(
            batch_size=batch_size,
            chunk_length=chunk_length,
            status="worker_crash",
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
        LOGGER.warning(
            "Benchmark worker crash chunk_length=%d batch_size=%d error=%s",
            chunk_length,
            batch_size,
            exc,
        )
        if progress is not None and task_id is not None:
            progress.update(task_id, stage="worker crash")
            if batch_task_id is not None:
                progress.update(
                    batch_task_id,
                    visible=False,
                    candidate=f"chunk={chunk_length} batch={batch_size}",
                    stage="worker crash",
                    phase="worker_crash",
                )
        built_chunk_entry = None
        if chunk_entry is None:
            built_chunk_entry = _ChunkDataCacheEntry(
                train_dataset=data_artifacts.train_dataset,
                val_dataset=data_artifacts.val_dataset,
                obs_shape=data_artifacts.obs_shape,
            )
        return result, data_artifacts.obs_shape, built_chunk_entry
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


def benchmark_world_model(
    cfg: WorldModelBenchmarkConfig,
    *,
    show_progress: bool = True,
) -> WorldModelBenchmarkSummary:
    if not cfg.data.dataset_paths:
        raise ValueError("Provide at least one dataset path.")
    if not cfg.batch_sizes:
        raise ValueError("Provide at least one batch size.")
    if not cfg.chunk_lengths:
        raise ValueError("Provide at least one chunk length.")

    run_paths = WorldModelRunPaths(cfg.run_name)
    run_paths.ensure_dirs()
    add_file_handler(run_paths.run_dir / "benchmark.log")
    config_path = run_paths.run_dir / "benchmark_config.json"
    json_path = run_paths.artifacts_dir / "benchmark_results.json"
    csv_path = run_paths.artifacts_dir / "benchmark_results.csv"
    LOGGER.info(
        "Initializing world-model benchmark run_name=%s device=%s candidates=%d amp=%s amp_dtype=%s compile_model=%s",
        cfg.run_name,
        cfg.training.device,
        len(cfg.chunk_lengths) * len(cfg.batch_sizes),
        cfg.training.amp,
        cfg.training.amp_dtype,
        cfg.training.compile_model,
    )
    LOGGER.info("Dataset paths: %s", ", ".join(cfg.data.dataset_paths))
    _write_json(config_path, asdict(cfg))

    results: list[WorldModelBenchmarkResult] = []
    obs_shape: tuple[int, int, int] | None = None
    total_candidates = len(cfg.chunk_lengths) * len(cfg.batch_sizes)
    LOGGER.info("Building shared dataset index once for benchmark sweep")
    _persist_results(
        cfg=cfg,
        run_paths=run_paths,
        json_path=json_path,
        csv_path=csv_path,
        obs_shape=obs_shape,
        results=results,
    )
    with build_progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.fields[candidate]}"),
        TextColumn("stage={task.fields[stage]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
        refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task(
            "Benchmark sweep",
            total=total_candidates,
            candidate="-",
            stage="initializing",
        )
        index_task_id = progress.add_task(
            "Dataset index",
            total=1,
            candidate="dataset index",
            stage="initializing",
        )
        batch_task_id = progress.add_task(
            "Benchmark batches",
            total=1,
            candidate="-",
            stage="idle",
            phase="-",
            visible=False,
        )
        indexed = build_index(
            cfg.data.dataset_paths,
            progress=progress,
            task_id=index_task_id,
        )
        chunk_cache: dict[int, _ChunkDataCacheEntry] = {}
        candidate_index = 0
        for chunk_length in cfg.chunk_lengths:
            for batch_size in cfg.batch_sizes:
                candidate_index += 1
                LOGGER.info(
                    "Candidate %d/%d chunk_length=%d batch_size=%d",
                    candidate_index,
                    total_candidates,
                    chunk_length,
                    batch_size,
                )
                chunk_entry = chunk_cache.get(chunk_length)
                result, current_obs_shape, built_chunk_entry = _run_single_benchmark(
                    cfg,
                    batch_size=batch_size,
                    chunk_length=chunk_length,
                    indexed=indexed,
                    chunk_entry=chunk_entry,
                    progress=progress,
                    task_id=task_id,
                    batch_task_id=batch_task_id,
                )
                if obs_shape is None:
                    obs_shape = current_obs_shape
                if chunk_entry is None and built_chunk_entry is not None:
                    chunk_cache[chunk_length] = built_chunk_entry
                results.append(result)
                _persist_results(
                    cfg=cfg,
                    run_paths=run_paths,
                    json_path=json_path,
                    csv_path=csv_path,
                    obs_shape=obs_shape,
                    results=results,
                )
                progress.advance(task_id)

    ok_results = [result for result in results if result.status == "ok"]
    LOGGER.info(
        "Finished world-model benchmark successful=%d total=%d results_json=%s",
        len(ok_results),
        len(results),
        json_path,
    )
    return WorldModelBenchmarkSummary(
        run_dir=str(run_paths.run_dir),
        config_path=str(config_path),
        json_path=str(json_path),
        csv_path=str(csv_path),
        device=cfg.training.device,
        obs_shape=obs_shape,
        results=tuple(results),
    )
