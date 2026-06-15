from __future__ import annotations

import json

import pytest

from src.world_model.config import WorldModelDataConfig
from src.world_model.probe_loader import WorldModelLoaderProbeConfig, probe_world_model_loader
from tests.world_model.test_train import _collect_train_dataset


@pytest.mark.integration
def test_probe_world_model_loader_smoke(monkeypatch, tiny_cfg, tmp_workdir):
    output_dir = _collect_train_dataset(monkeypatch, tiny_cfg, tmp_workdir, total_transitions=8)

    summary = probe_world_model_loader(
        WorldModelLoaderProbeConfig(
            run_name="wm-loader-probe-smoke",
            data=WorldModelDataConfig(
                dataset_paths=[str(output_dir)],
                batch_size=2,
                num_workers=0,
                chunk_length=2,
                stride=1,
                val_ratio=0.25,
                expected_num_actions=5,
                include_metadata=False,
            ),
            batch_sizes=[1, 2],
            chunk_lengths=[2],
            num_workers_options=[0],
            pin_memory_options=[False],
            persistent_workers_options=[False],
            prefetch_factors=[2],
            warmup_batches=1,
            measure_batches=2,
            move_to_device=False,
            device="cpu",
        ),
        show_progress=False,
    )

    run_dir = tmp_workdir / "runs" / "world_model" / "wm-loader-probe-smoke"
    assert summary.obs_shape == (3, 8, 8)
    assert len(summary.results) == 2
    assert all(result.status == "ok" for result in summary.results)
    assert (run_dir / "loader_probe_config.json").exists()
    assert (run_dir / "artifacts" / "loader_probe_results.json").exists()
    assert (run_dir / "artifacts" / "loader_probe_results.csv").exists()

    payload = json.loads(
        (run_dir / "artifacts" / "loader_probe_results.json").read_text(encoding="utf-8")
    )
    assert payload["device"] == "cpu"
    assert len(payload["results"]) == 2
