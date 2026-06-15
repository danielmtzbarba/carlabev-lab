from __future__ import annotations

import json

import pytest

from src.world_model.data import build_index
from src.world_model.staging import stage_dataset_to_tmp
from tests.world_model.test_train import _collect_train_dataset


@pytest.mark.integration
def test_stage_dataset_to_tmp_copies_dataset(monkeypatch, tiny_cfg, tmp_workdir):
    output_dir = _collect_train_dataset(monkeypatch, tiny_cfg, tmp_workdir, total_transitions=8)
    tmp_root = tmp_workdir / "tmp"

    result = stage_dataset_to_tmp(str(output_dir), tmp_root=tmp_root, dest_name="staged-seed-0")

    staged_dir = tmp_root / "staged-seed-0"
    assert result.source_dir == str(output_dir)
    assert result.staged_dir == str(staged_dir)
    assert result.shard_count == 2
    assert result.prepared_shards == 2
    assert result.total_bytes > 0
    assert (staged_dir / "summary.json").exists()
    assert (staged_dir / "shard_000000.npz").exists()
    assert (staged_dir / "shard_000001.npz").exists()
    assert (staged_dir / ".wm_cache" / "prepared_shards" / "shard_000000" / "obs.npy").exists()
    assert (staged_dir / ".wm_cache" / "prepared_shards" / "shard_000001" / "manifest.json").exists()
    summary = json.loads((staged_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["output_dir"] == str(staged_dir)
    assert summary["shards"][0]["path"] == "shard_000000.npz"
    assert summary["shards"][1]["path"] == "shard_000001.npz"

    indexed = build_index([str(staged_dir)])
    assert all(str(ref.shard_path).startswith(str(staged_dir)) for ref in indexed.transitions)
