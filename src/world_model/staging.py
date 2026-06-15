from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.utils.common_logging import event_message, get_logger
from src.utils.storage_paths import resolve_artifact_path
from src.world_model.data import prepare_shard_cache

LOGGER = get_logger("world_model.staging")


@dataclass(frozen=True)
class StagedDatasetResult:
    source_dir: str
    staged_dir: str
    shard_count: int
    total_bytes: int
    prepared_shards: int


def _summary_file(dataset_dir: Path) -> Path:
    return dataset_dir / "summary.json"


def _load_summary(dataset_dir: Path) -> dict:
    summary_file = _summary_file(dataset_dir)
    if not summary_file.exists():
        raise FileNotFoundError(f"Expected summary.json in dataset directory: {dataset_dir}")
    return json.loads(summary_file.read_text(encoding="utf-8"))


def _rewrite_staged_summary_paths(staged_dir: Path) -> dict:
    summary = _load_summary(staged_dir)
    rewritten_shards: list[dict] = []
    for shard_meta in summary.get("shards", []):
        rewritten = dict(shard_meta)
        rewritten["path"] = Path(str(shard_meta["path"])).name
        rewritten_shards.append(rewritten)
    summary["shards"] = rewritten_shards
    summary["output_dir"] = str(staged_dir)
    _summary_file(staged_dir).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _coerce_dataset_dir(path: str | Path) -> Path:
    dataset_dir = resolve_artifact_path(path)
    if dataset_dir.is_file():
        dataset_dir = dataset_dir.parent
    return dataset_dir


def _default_tmp_root() -> Path:
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "manual")
    tmpdir = os.environ.get("TMPDIR") or "/tmp"
    return Path(tmpdir) / slurm_job_id / "carlabev-world-model"


def _dataset_size_bytes(dataset_dir: Path) -> int:
    total = 0
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def stage_dataset_to_tmp(
    path: str | Path,
    *,
    tmp_root: str | Path | None = None,
    dest_name: str | None = None,
    overwrite: bool = True,
    prepare_shards: bool = True,
) -> StagedDatasetResult:
    source_dir = _coerce_dataset_dir(path)
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source_dir}")

    summary = _load_summary(source_dir)
    shard_count = int(summary.get("shard_count", len(summary.get("shards", []))))
    if shard_count <= 0:
        raise ValueError(f"No shards listed in summary for {source_dir}")

    tmp_base = Path(tmp_root).expanduser() if tmp_root is not None else _default_tmp_root()
    destination_name = dest_name or source_dir.name
    staged_dir = tmp_base / destination_name

    LOGGER.info(event_message("STAGE", "COPY_START", source=source_dir, destination=staged_dir))
    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    if staged_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {staged_dir}")
        shutil.rmtree(staged_dir)
    shutil.copytree(source_dir, staged_dir)

    staged_summary = _rewrite_staged_summary_paths(staged_dir)
    staged_shards = staged_summary.get("shards", [])
    if len(staged_shards) != shard_count:
        raise ValueError(
            f"Staged shard-count mismatch for {staged_dir}: expected {shard_count} got {len(staged_shards)}"
        )

    prepared_shards = 0
    if prepare_shards:
        LOGGER.info(event_message("STAGE", "PREPARE_START", shards=shard_count))
        for shard_meta in staged_shards:
            prepare_shard_cache(staged_dir / Path(shard_meta["path"]).name)
            prepared_shards += 1
        LOGGER.info(event_message("STAGE", "PREPARE_DONE", prepared_shards=prepared_shards))

    total_bytes = _dataset_size_bytes(staged_dir)
    LOGGER.info(
        event_message(
            "STAGE",
            "DONE",
            shard_count=shard_count,
            prepared_shards=prepared_shards,
            total_bytes=total_bytes,
            staged_dir=staged_dir,
        )
    )
    return StagedDatasetResult(
        source_dir=str(source_dir),
        staged_dir=str(staged_dir),
        shard_count=shard_count,
        total_bytes=total_bytes,
        prepared_shards=prepared_shards,
    )
