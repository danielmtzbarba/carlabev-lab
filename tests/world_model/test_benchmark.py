from __future__ import annotations

import json

import pytest

from src.world_model.benchmark import WorldModelBenchmarkConfig, benchmark_world_model
from src.world_model.config import (
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from tests.world_model.test_train import _collect_train_dataset


@pytest.mark.integration
def test_benchmark_world_model_smoke(monkeypatch, tiny_cfg, tmp_workdir):
    output_dir = _collect_train_dataset(monkeypatch, tiny_cfg, tmp_workdir, total_transitions=8)

    summary = benchmark_world_model(
        WorldModelBenchmarkConfig(
            run_name="wm-bench-smoke",
            data=WorldModelDataConfig(
                dataset_paths=[str(output_dir)],
                batch_size=2,
                num_workers=0,
                chunk_length=2,
                stride=1,
                val_ratio=0.25,
                expected_num_actions=5,
            ),
            model=WorldModelModelConfig(
                encoder_backend="lewm_compatible_vit",
                patch_size=4,
                encoder_dim=16,
                encoder_depth=1,
                predictor_depth=1,
                num_heads=4,
                mlp_ratio=2.0,
                action_embed_dim=8,
                latent_dim=16,
                dropout=0.0,
            ),
            optimizer=WorldModelOptimizerConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                max_grad_norm=1.0,
            ),
            training=WorldModelTrainLoopConfig(
                epochs=1,
                device="cpu",
                save_every=1,
                sigreg_weight=0.01,
            ),
            batch_sizes=[1, 2],
            chunk_lengths=[2],
            warmup_batches=0,
            measure_batches=1,
        )
    )

    run_dir = tmp_workdir / "runs" / "world_model" / "wm-bench-smoke"
    assert summary.obs_shape == (3, 8, 8)
    assert len(summary.results) == 2
    assert all(result.status == "ok" for result in summary.results)
    assert (run_dir / "benchmark_config.json").exists()
    assert (run_dir / "artifacts" / "benchmark_results.json").exists()
    assert (run_dir / "artifacts" / "benchmark_results.csv").exists()

    payload = json.loads((run_dir / "artifacts" / "benchmark_results.json").read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert len(payload["results"]) == 2
