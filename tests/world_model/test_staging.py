from __future__ import annotations

import pytest

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
    assert result.total_bytes > 0
    assert (staged_dir / "summary.json").exists()
    assert (staged_dir / "shard_000000.npz").exists()
    assert (staged_dir / "shard_000001.npz").exists()
