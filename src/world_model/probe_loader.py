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
from rich.table import Table

from src.utils.common_logging import add_file_handler, build_progress, event_message, get_console, get_logger
from src.world_model.config import WorldModelDataConfig
from src.world_model.contracts import WorldModelSequenceConfig
from src.world_model.data import (
    IndexedDataset,
    build_dataloader,
    build_index,
    build_sequence_datasets_from_indexed,
)
from src.world_model.run_paths import WorldModelRunPaths

LOGGER = get_logger("world_model.probe_loader")


@dataclass
class WorldModelLoaderProbeConfig:
    run_name: str = "lewm-loader-probe"
    data: WorldModelDataConfig = field(default_factory=WorldModelDataConfig)
    batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])
    chunk_lengths: list[int] = field(default_factory=lambda: [4, 8])
    num_workers_options: list[int] = field(default_factory=lambda: [0, 2, 4])
    pin_memory_options: list[bool] = field(default_factory=lambda: [False, True])
    persistent_workers_options: list[bool] = field(default_factory=lambda: [False, True])
    prefetch_factors: list[int] = field(default_factory=lambda: [2])
    warmup_batches: int = 1
    measure_batches: int = 3
    move_to_device: bool = False
    device: str = "cuda"


@dataclass(frozen=True)
class WorldModelLoaderProbeResult:
    batch_size: int
    chunk_length: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool | None
    prefetch_factor: int | None
    status: str
    warmup_batches: int
    measured_batches: int
    measured_samples: int
    elapsed_seconds: float | None
    batches_per_second: float | None
    samples_per_second: float | None
    first_batch_seconds: float | None
    host_to_device: bool
    peak_memory_mb: float | None
    chunk_build_ms: float | None
    loader_build_ms: float | None
    avg_warmup_fetch_ms: float | None
    avg_warmup_transfer_ms: float | None
    avg_warmup_batch_total_ms: float | None
    avg_measure_fetch_ms: float | None
    avg_measure_transfer_ms: float | None
    avg_measure_batch_total_ms: float | None
    error_message: str | None = None


@dataclass(frozen=True)
class WorldModelLoaderProbeSummary:
    run_dir: str
    config_path: str
    json_path: str
    csv_path: str
    device: str
    obs_shape: tuple[int, int, int] | None
    results: tuple[WorldModelLoaderProbeResult, ...]


@dataclass(frozen=True)
class _ChunkDatasetCacheEntry:
    train_dataset: Any
    val_dataset: Any
    obs_shape: tuple[int, int, int]


def _is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _reset_peak_memory(device: torch.device) -> None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return None


def _sync_device(device: torch.device) -> None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _cleanup_device(device: torch.device) -> None:
    gc.collect()
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


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


def _batch_position(index: int, total: int) -> str:
    width = max(2, len(str(max(total, 1))))
    return f"{index:0{width}d}/{total:0{width}d}"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _separator(label: str) -> None:
    LOGGER.info("=" * 104)
    LOGGER.info(event_message("PROBE", label))
    LOGGER.info("=" * 104)


def _log_table(table: Table) -> None:
    console = get_console()
    with console.capture() as capture:
        console.print(table)
    for line in capture.get().rstrip().splitlines():
        LOGGER.info(line)


def _log_candidate_timing_table(candidate_label: str, result: WorldModelLoaderProbeResult) -> None:
    table = Table(title=f"Loader Probe Timing: {candidate_label}", header_style="bold magenta")
    table.add_column("Stage", style="bold")
    table.add_column("Avg ms", justify="right")
    table.add_column("Details")
    rows = [
        ("Chunk Build", result.chunk_build_ms, "sequence dataset + cache load/build"),
        ("Loader Build", result.loader_build_ms, "DataLoader construction"),
        ("Warmup Fetch", result.avg_warmup_fetch_ms, f"avg over {result.warmup_batches} batch(es)"),
        ("Warmup Transfer", result.avg_warmup_transfer_ms, "host to device + sync"),
        ("Warmup Total", result.avg_warmup_batch_total_ms, "fetch + transfer"),
        ("Measure Fetch", result.avg_measure_fetch_ms, f"avg over {result.measured_batches} batch(es)"),
        ("Measure Transfer", result.avg_measure_transfer_ms, "host to device + sync"),
        ("Measure Total", result.avg_measure_batch_total_ms, "fetch + transfer"),
    ]
    for stage, avg_ms, details in rows:
        table.add_row(stage, "-" if avg_ms is None else f"{avg_ms:.3f}", details)
    _log_table(table)


def _log_final_summary_table(results: list[WorldModelLoaderProbeResult]) -> None:
    table = Table(title="Loader Probe Summary", header_style="bold magenta")
    table.add_column("Candidate", style="bold")
    table.add_column("Status")
    table.add_column("Measure Fetch", justify="right")
    table.add_column("Measure Transfer", justify="right")
    table.add_column("Measure Total", justify="right")
    table.add_column("Samples/s", justify="right")
    for result in results:
        candidate = (
            f"c{result.chunk_length}-b{result.batch_size}-w{result.num_workers}-p{int(result.pin_memory)}"
        )
        table.add_row(
            candidate,
            result.status,
            "-" if result.avg_measure_fetch_ms is None else f"{result.avg_measure_fetch_ms:.3f}",
            "-" if result.avg_measure_transfer_ms is None else f"{result.avg_measure_transfer_ms:.3f}",
            "-" if result.avg_measure_batch_total_ms is None else f"{result.avg_measure_batch_total_ms:.3f}",
            "-" if result.samples_per_second is None else f"{result.samples_per_second:.3f}",
        )
    _log_table(table)


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message and "memory" in message


def _is_worker_crash(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "dataloader worker" in message or "exited unexpectedly" in message


def _candidate_count(cfg: WorldModelLoaderProbeConfig) -> int:
    count = 0
    for num_workers in cfg.num_workers_options:
        persistent_options = [None] if num_workers == 0 else cfg.persistent_workers_options
        prefetch_options = [None] if num_workers == 0 else cfg.prefetch_factors
        count += (
            len(cfg.batch_sizes)
            * len(cfg.chunk_lengths)
            * len(cfg.pin_memory_options)
            * len(persistent_options)
            * len(prefetch_options)
        )
    return count


def _build_results_payload(
    *,
    cfg: WorldModelLoaderProbeConfig,
    run_paths: WorldModelRunPaths,
    obs_shape: tuple[int, int, int] | None,
    results: list[WorldModelLoaderProbeResult],
) -> dict[str, Any]:
    return {
        "run_name": cfg.run_name,
        "run_dir": str(run_paths.run_dir),
        "device": cfg.device,
        "obs_shape": list(obs_shape) if obs_shape is not None else None,
        "results": [asdict(result) for result in results],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, results: list[WorldModelLoaderProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in WorldModelLoaderProbeResult.__dataclass_fields__.values()]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _persist_results(
    *,
    cfg: WorldModelLoaderProbeConfig,
    run_paths: WorldModelRunPaths,
    json_path: Path,
    csv_path: Path,
    obs_shape: tuple[int, int, int] | None,
    results: list[WorldModelLoaderProbeResult],
) -> None:
    payload = _build_results_payload(
        cfg=cfg,
        run_paths=run_paths,
        obs_shape=obs_shape,
        results=results,
    )
    _write_json(json_path, payload)
    _write_csv(csv_path, results)


def _build_chunk_dataset_entry(
    indexed: IndexedDataset,
    data_cfg: WorldModelDataConfig,
) -> _ChunkDatasetCacheEntry:
    train_dataset, val_dataset = build_sequence_datasets_from_indexed(
        indexed,
        cfg=WorldModelSequenceConfig(
            chunk_length=data_cfg.chunk_length,
            stride=data_cfg.stride,
            val_ratio=data_cfg.val_ratio,
            expected_num_actions=data_cfg.expected_num_actions,
        ),
        include_metadata=data_cfg.include_metadata,
        cache_sequence_indices=data_cfg.cache_sequence_indices,
        sequence_cache_dir=data_cfg.sequence_cache_dir,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after sequence-window construction.")
    sample = train_dataset[0]
    obs_shape = tuple(sample["obs"].shape[1:])
    return _ChunkDatasetCacheEntry(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        obs_shape=obs_shape,
    )


def _run_single_probe(
    cfg: WorldModelLoaderProbeConfig,
    *,
    batch_size: int,
    chunk_length: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
    indexed: IndexedDataset,
    chunk_entry: _ChunkDatasetCacheEntry | None = None,
    progress=None,
    task_id: int | None = None,
    batch_task_id: int | None = None,
) -> tuple[WorldModelLoaderProbeResult, tuple[int, int, int], _ChunkDatasetCacheEntry | None]:
    candidate_label = (
        f"chunk={chunk_length} batch={batch_size} workers={num_workers} "
        f"pin={int(pin_memory)}"
    )
    if progress is not None and task_id is not None:
        progress.update(task_id, candidate=candidate_label, stage="preparing loader")
        if batch_task_id is not None:
            progress.update(
                batch_task_id,
                description="Loader batches",
                total=1,
                completed=0,
                visible=False,
                candidate=candidate_label,
                stage="idle",
                phase="-",
            )

    device = torch.device(cfg.device)
    chunk_build_ms: float | None = None
    loader_build_ms: float | None = None
    warmup_fetch_ms_values: list[float] = []
    warmup_transfer_ms_values: list[float] = []
    warmup_total_ms_values: list[float] = []
    measure_fetch_ms_values: list[float] = []
    measure_transfer_ms_values: list[float] = []
    measure_total_ms_values: list[float] = []
    try:
        _separator(f"START {candidate_label}")
        if chunk_entry is None:
            LOGGER.info(event_message("PROBE", "CHUNK_BUILD", chunk_length=chunk_length))
            chunk_cfg = replace(cfg.data, chunk_length=chunk_length)
            chunk_build_start = time.perf_counter()
            chunk_entry = _build_chunk_dataset_entry(indexed, chunk_cfg)
            chunk_build_ms = (time.perf_counter() - chunk_build_start) * 1000.0
            LOGGER.info(
                event_message(
                    "PROBE",
                    "CHUNK_READY",
                    chunk_length=chunk_length,
                    train_windows=len(chunk_entry.train_dataset),
                    val_windows=len(chunk_entry.val_dataset),
                    obs_shape=chunk_entry.obs_shape,
                    build_ms=chunk_build_ms,
                )
            )
        else:
            LOGGER.info(event_message("PROBE", "CHUNK_REUSE", chunk_length=chunk_length))

        loader_build_start = time.perf_counter()
        loader = build_dataloader(
            chunk_entry.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            device=cfg.device if cfg.move_to_device else None,
        )
        loader_build_ms = (time.perf_counter() - loader_build_start) * 1000.0
        LOGGER.info(
            event_message(
                "PROBE",
                "START",
                candidate=candidate_label,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                train_batches=len(loader),
                move_to_device=cfg.move_to_device,
                loader_build_ms=loader_build_ms,
            )
        )

        if progress is not None and task_id is not None:
            progress.update(task_id, stage=f"warmup {cfg.warmup_batches} batches")
            if batch_task_id is not None:
                progress.reset(
                    batch_task_id,
                    total=max(cfg.warmup_batches + cfg.measure_batches, 1),
                    completed=0,
                    visible=True,
                    candidate=candidate_label,
                    stage=f"warmup {cfg.warmup_batches} batches",
                    phase="warmup",
                )

        first_batch_seconds: float | None = None
        if cfg.warmup_batches > 0:
            iterator = iter(loader)
            for batch_index in range(cfg.warmup_batches):
                fetch_start = time.perf_counter()
                batch = next(iterator)
                fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
                transfer_start = time.perf_counter()
                if cfg.move_to_device:
                    batch = _to_device(batch, device)
                    _sync_device(device)
                transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
                batch_total_ms = fetch_ms + transfer_ms
                warmup_fetch_ms_values.append(fetch_ms)
                warmup_transfer_ms_values.append(transfer_ms)
                warmup_total_ms_values.append(batch_total_ms)
                if batch_index == 0:
                    first_batch_seconds = batch_total_ms / 1000.0
                LOGGER.info(
                    event_message(
                        "PROBE",
                        "WARMUP_BATCH",
                        candidate=candidate_label,
                        batch=_batch_position(batch_index + 1, cfg.warmup_batches),
                        batch_size=int(batch["obs"].shape[0]),
                        fetch_ms=fetch_ms,
                        transfer_ms=transfer_ms,
                        batch_total_ms=batch_total_ms,
                    )
                )
                if progress is not None and batch_task_id is not None:
                    progress.update(batch_task_id, advance=1)
        else:
            iterator = iter(loader)
            fetch_start = time.perf_counter()
            batch = next(iterator)
            fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
            transfer_start = time.perf_counter()
            if cfg.move_to_device:
                batch = _to_device(batch, device)
                _sync_device(device)
            transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
            batch_total_ms = fetch_ms + transfer_ms
            warmup_fetch_ms_values.append(fetch_ms)
            warmup_transfer_ms_values.append(transfer_ms)
            warmup_total_ms_values.append(batch_total_ms)
            first_batch_seconds = batch_total_ms / 1000.0
            LOGGER.info(
                event_message(
                    "PROBE",
                    "WARMUP_BATCH",
                    candidate=candidate_label,
                    batch=_batch_position(1, 1),
                    batch_size=int(batch["obs"].shape[0]),
                    fetch_ms=fetch_ms,
                    transfer_ms=transfer_ms,
                    batch_total_ms=batch_total_ms,
                )
            )
            del batch

        _cleanup_device(device)
        _reset_peak_memory(device)
        _sync_device(device)

        measured_samples = 0
        if progress is not None and task_id is not None:
            progress.update(task_id, stage=f"measuring {cfg.measure_batches} batches")
            if batch_task_id is not None:
                progress.update(
                    batch_task_id,
                    candidate=candidate_label,
                    stage=f"measuring {cfg.measure_batches} batches",
                    phase="measure",
                )
        start = time.perf_counter()
        iterator = iter(loader)
        for batch_index in range(cfg.measure_batches):
            fetch_start = time.perf_counter()
            batch = next(iterator)
            fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
            transfer_start = time.perf_counter()
            if cfg.move_to_device:
                batch = _to_device(batch, device)
                _sync_device(device)
            transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
            batch_total_ms = fetch_ms + transfer_ms
            measure_fetch_ms_values.append(fetch_ms)
            measure_transfer_ms_values.append(transfer_ms)
            measure_total_ms_values.append(batch_total_ms)
            measured_samples += int(batch["obs"].shape[0])
            LOGGER.info(
                event_message(
                    "PROBE",
                    "MEASURE_BATCH",
                    candidate=candidate_label,
                    batch=_batch_position(batch_index + 1, cfg.measure_batches),
                    batch_size=int(batch["obs"].shape[0]),
                    fetch_ms=fetch_ms,
                    transfer_ms=transfer_ms,
                    batch_total_ms=batch_total_ms,
                )
            )
            if progress is not None and batch_task_id is not None:
                progress.update(batch_task_id, advance=1)
        _sync_device(device)
        elapsed_seconds = time.perf_counter() - start
        peak_memory_mb = _peak_memory_mb(device) if cfg.move_to_device else None
        result = WorldModelLoaderProbeResult(
            batch_size=batch_size,
            chunk_length=chunk_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            status="ok",
            warmup_batches=cfg.warmup_batches,
            measured_batches=cfg.measure_batches,
            measured_samples=measured_samples,
            elapsed_seconds=elapsed_seconds,
            batches_per_second=cfg.measure_batches / elapsed_seconds if elapsed_seconds > 0 else None,
            samples_per_second=measured_samples / elapsed_seconds if elapsed_seconds > 0 else None,
            first_batch_seconds=first_batch_seconds,
            host_to_device=cfg.move_to_device,
            peak_memory_mb=peak_memory_mb,
            chunk_build_ms=chunk_build_ms,
            loader_build_ms=loader_build_ms,
            avg_warmup_fetch_ms=_mean_or_none(warmup_fetch_ms_values),
            avg_warmup_transfer_ms=_mean_or_none(warmup_transfer_ms_values),
            avg_warmup_batch_total_ms=_mean_or_none(warmup_total_ms_values),
            avg_measure_fetch_ms=_mean_or_none(measure_fetch_ms_values),
            avg_measure_transfer_ms=_mean_or_none(measure_transfer_ms_values),
            avg_measure_batch_total_ms=_mean_or_none(measure_total_ms_values),
        )
        LOGGER.info(
            event_message(
                "PROBE",
                "DONE",
                candidate=candidate_label,
                status="ok",
                samples_per_second=result.samples_per_second or 0.0,
                first_batch_seconds=result.first_batch_seconds or 0.0,
            )
        )
        _log_candidate_timing_table(candidate_label, result)
        _separator(f"END {candidate_label}")
        return result, chunk_entry.obs_shape, chunk_entry
    except RuntimeError as exc:
        status = "worker_crash" if _is_worker_crash(exc) else "oom" if _is_oom_error(exc) else "error"
        LOGGER.exception("Loader probe failed %s status=%s", candidate_label, status)
        result = WorldModelLoaderProbeResult(
                batch_size=batch_size,
                chunk_length=chunk_length,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                status=status,
                warmup_batches=cfg.warmup_batches,
                measured_batches=cfg.measure_batches,
                measured_samples=0,
                elapsed_seconds=None,
                batches_per_second=None,
                samples_per_second=None,
                first_batch_seconds=None,
                host_to_device=cfg.move_to_device,
                peak_memory_mb=None,
                chunk_build_ms=chunk_build_ms,
                loader_build_ms=loader_build_ms,
                avg_warmup_fetch_ms=_mean_or_none(warmup_fetch_ms_values),
                avg_warmup_transfer_ms=_mean_or_none(warmup_transfer_ms_values),
                avg_warmup_batch_total_ms=_mean_or_none(warmup_total_ms_values),
                avg_measure_fetch_ms=_mean_or_none(measure_fetch_ms_values),
                avg_measure_transfer_ms=_mean_or_none(measure_transfer_ms_values),
                avg_measure_batch_total_ms=_mean_or_none(measure_total_ms_values),
                error_message=str(exc),
        )
        _log_candidate_timing_table(candidate_label, result)
        _separator(f"END {candidate_label}")
        return (result, chunk_entry.obs_shape if chunk_entry is not None else None, chunk_entry)
    finally:
        if progress is not None and task_id is not None:
            progress.update(task_id, stage="done")
            if batch_task_id is not None:
                progress.update(batch_task_id, visible=False, candidate=candidate_label, stage="done", phase="done")
        _cleanup_device(device)


def probe_world_model_loader(
    cfg: WorldModelLoaderProbeConfig,
    *,
    show_progress: bool = False,
) -> WorldModelLoaderProbeSummary:
    if not cfg.data.dataset_paths:
        raise ValueError("Provide at least one dataset path.")
    if cfg.measure_batches <= 0:
        raise ValueError("measure_batches must be positive")
    if cfg.warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")

    run_paths = WorldModelRunPaths(cfg.run_name)
    run_paths.ensure_dirs()
    config_path = run_paths.run_dir / "loader_probe_config.json"
    json_path = run_paths.artifacts_dir / "loader_probe_results.json"
    csv_path = run_paths.artifacts_dir / "loader_probe_results.csv"
    add_file_handler(run_paths.run_dir / "loader_probe.log")
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    LOGGER.info(
        event_message(
            "PROBE",
            "INIT",
            run_name=cfg.run_name,
            device=cfg.device,
            candidates=_candidate_count(cfg),
            move_to_device=cfg.move_to_device,
        )
    )
    LOGGER.info(event_message("PROBE", "DATASET_PATHS", paths=cfg.data.dataset_paths))
    LOGGER.info(event_message("PROBE", "INDEX_BUILD"))
    index_build_start = time.perf_counter()
    indexed = build_index(cfg.data.dataset_paths)
    LOGGER.info(
        event_message(
            "PROBE",
            "INDEX_READY",
            transitions=len(indexed.transitions),
            episodes=len(indexed.episodes),
            shards=len(indexed.shards),
            build_ms=(time.perf_counter() - index_build_start) * 1000.0,
        )
    )

    results: list[WorldModelLoaderProbeResult] = []
    obs_shape: tuple[int, int, int] | None = None
    chunk_cache: dict[int, _ChunkDatasetCacheEntry] = {}

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
            "Loader probe sweep",
            total=max(_candidate_count(cfg), 1),
            candidate="-",
            stage="initializing",
        )
        batch_task_id = progress.add_task(
            "Loader batches",
            total=1,
            candidate="-",
            stage="idle",
            phase="-",
            visible=False,
        )
        for chunk_length in cfg.chunk_lengths:
            for batch_size in cfg.batch_sizes:
                for num_workers in cfg.num_workers_options:
                    persistent_options = [None] if num_workers == 0 else cfg.persistent_workers_options
                    prefetch_options = [None] if num_workers == 0 else cfg.prefetch_factors
                    for pin_memory in cfg.pin_memory_options:
                        for persistent_workers in persistent_options:
                            for prefetch_factor in prefetch_options:
                                result, current_obs_shape, built_chunk_entry = _run_single_probe(
                                    cfg,
                                    batch_size=batch_size,
                                    chunk_length=chunk_length,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    persistent_workers=persistent_workers,
                                    prefetch_factor=prefetch_factor,
                                    indexed=indexed,
                                    chunk_entry=chunk_cache.get(chunk_length),
                                    progress=progress,
                                    task_id=task_id,
                                    batch_task_id=batch_task_id,
                                )
                                if built_chunk_entry is not None and chunk_length not in chunk_cache:
                                    chunk_cache[chunk_length] = built_chunk_entry
                                if current_obs_shape is not None:
                                    obs_shape = current_obs_shape
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

    _separator("FINAL SUMMARY")
    _log_final_summary_table(results)
    _separator("END SUMMARY")

    return WorldModelLoaderProbeSummary(
        run_dir=str(run_paths.run_dir),
        config_path=str(config_path),
        json_path=str(json_path),
        csv_path=str(csv_path),
        device=cfg.device,
        obs_shape=obs_shape,
        results=tuple(results),
    )
