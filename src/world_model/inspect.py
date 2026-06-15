from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.utils.storage_paths import resolve_artifact_path
from src.world_model.data import resolve_dataset_shard_path


def inspect_dataset_path(path: str, *, shard_index: int = 0) -> dict[str, Any]:
    target = resolve_artifact_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    if target.is_dir():
        summary_file = target / "summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(
                f"Expected summary.json in dataset directory: {target}"
            )

        with open(summary_file, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        shards = summary.get("shards", [])
        if not shards:
            raise ValueError(f"No shards listed in summary: {summary_file}")
        if shard_index < 0 or shard_index >= len(shards):
            raise IndexError(
                f"shard_index {shard_index} is out of range for {len(shards)} shards."
            )
        shard_path = resolve_dataset_shard_path(target, shards[shard_index]["path"])
    else:
        summary = None
        shard_path = target

    with np.load(shard_path, allow_pickle=False) as data:
        arrays = {
            key: {
                "shape": list(data[key].shape),
                "dtype": str(data[key].dtype),
            }
            for key in data.files
        }

    return {
        "dataset_dir": str(target if target.is_dir() else shard_path.parent),
        "summary": summary,
        "inspected_shard_path": str(shard_path),
        "arrays": arrays,
    }
