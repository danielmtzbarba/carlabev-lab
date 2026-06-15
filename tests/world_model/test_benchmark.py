from __future__ import annotations

import json

import pytest

import src.world_model.benchmark as benchmark_mod
from src.world_model.benchmark import (
    WorldModelBenchmarkConfig,
    WorldModelBenchmarkResult,
    benchmark_world_model,
)
from src.world_model.config import (
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from tests.world_model.test_train import _collect_train_dataset


@pytest.mark.unit
def test_worker_crash_detection_matches_dataloader_failure():
    exc = RuntimeError("DataLoader worker (pid 123) exited unexpectedly")

    assert benchmark_mod._is_worker_crash(exc) is True


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
        ),
        show_progress=False,
    )

    run_dir = tmp_workdir / "runs" / "world_model" / "wm-bench-smoke"
    assert summary.obs_shape == (3, 8, 8)
    assert len(summary.results) == 2
    assert all(result.status == "ok" for result in summary.results)
    assert (run_dir / "benchmark_config.json").exists()
    assert (run_dir / "artifacts" / "benchmark_results.json").exists()
    assert (run_dir / "artifacts" / "benchmark_results.csv").exists()
    assert (run_dir / "artifacts" / "benchmark_state.json").exists()

    payload = json.loads((run_dir / "artifacts" / "benchmark_results.json").read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert len(payload["results"]) == 2
    assert "avg_fetch_ms" in payload["results"][0]
    assert "avg_transfer_ms" in payload["results"][0]
    assert "avg_step_ms" in payload["results"][0]
    state_payload = json.loads((run_dir / "artifacts" / "benchmark_state.json").read_text(encoding="utf-8"))
    assert state_payload["status"] == "finished"
    assert state_payload["current_candidate"] is None
    assert len(state_payload["completed_results"]) == 2
    assert summary.state_path.endswith("benchmark_state.json")


@pytest.mark.unit
def test_benchmark_reuses_chunk_cache(monkeypatch, tmp_workdir):
    monkeypatch.setenv("CARLABEV_RUNS_ROOT", str(tmp_workdir / "runs"))

    calls: list[tuple[int, int, bool]] = []

    def fake_build_index(_dataset_paths, *, progress=None, task_id=None):
        del progress, task_id
        return object()

    def fake_run_single_benchmark(
        cfg,
        *,
        batch_size,
        chunk_length,
        indexed,
        chunk_entry=None,
        state_callback=None,
        progress=None,
        task_id=None,
        batch_task_id=None,
    ):
        del cfg, indexed, state_callback, progress, task_id, batch_task_id
        calls.append((chunk_length, batch_size, chunk_entry is None))
        built_chunk_entry = None
        if chunk_entry is None:
            built_chunk_entry = benchmark_mod._ChunkDataCacheEntry(
                train_dataset=[0, 1],
                val_dataset=[2],
                obs_shape=(3, 8, 8),
            )
        return (
            WorldModelBenchmarkResult(
                batch_size=batch_size,
                chunk_length=chunk_length,
                status="ok",
                warmup_batches=0,
                measured_batches=1,
                measured_samples=batch_size,
                elapsed_seconds=1.0,
                batches_per_second=1.0,
                samples_per_second=float(batch_size),
                tokens_per_second=float(batch_size * chunk_length),
                avg_fetch_ms=1.0,
                avg_transfer_ms=2.0,
                avg_step_ms=3.0,
                peak_memory_mb=None,
                last_loss=0.1,
            ),
            (3, 8, 8),
            built_chunk_entry,
        )

    monkeypatch.setattr(benchmark_mod, "build_index", fake_build_index)
    monkeypatch.setattr(benchmark_mod, "_run_single_benchmark", fake_run_single_benchmark)

    summary = benchmark_world_model(
        WorldModelBenchmarkConfig(
            run_name="wm-bench-cache",
            data=WorldModelDataConfig(dataset_paths=["datasets/world_model/demo"]),
            batch_sizes=[1, 2],
            chunk_lengths=[2, 4],
            warmup_batches=0,
            measure_batches=1,
        ),
        show_progress=False,
    )

    assert len(summary.results) == 4
    assert calls == [
        (2, 1, True),
        (2, 2, False),
        (4, 1, True),
        (4, 2, False),
    ]
    payload = json.loads(
        (
            tmp_workdir
            / "runs"
            / "world_model"
            / "wm-bench-cache"
            / "artifacts"
            / "benchmark_results.json"
        ).read_text(encoding="utf-8")
    )
    assert len(payload["results"]) == 4


@pytest.mark.unit
def test_benchmark_persists_in_progress_state(monkeypatch, tmp_workdir):
    monkeypatch.setenv("CARLABEV_RUNS_ROOT", str(tmp_workdir / "runs"))

    def fake_build_index(_dataset_paths, *, progress=None, task_id=None):
        del progress, task_id
        return object()

    captured_state: dict[str, object] = {}

    def fake_run_single_benchmark(
        cfg,
        *,
        batch_size,
        chunk_length,
        indexed,
        chunk_entry=None,
        state_callback=None,
        progress=None,
        task_id=None,
        batch_task_id=None,
    ):
        del cfg, indexed, chunk_entry, progress, task_id, batch_task_id
        assert state_callback is not None
        state_callback(
            {
                "batch_size": batch_size,
                "chunk_length": chunk_length,
                "stage": "measuring",
                "warmup_batches": 0,
                "measure_batches": 1,
                "warmup_completed": 0,
                "measured_completed": 1,
                "last_loss": 0.5,
                "avg_fetch_ms": 1.0,
                "avg_transfer_ms": 2.0,
                "avg_step_ms": 3.0,
                "error_message": None,
            }
        )
        state_path = (
            tmp_workdir
            / "runs"
            / "world_model"
            / "wm-bench-state"
            / "artifacts"
            / "benchmark_state.json"
        )
        captured_state.update(json.loads(state_path.read_text(encoding="utf-8")))
        return (
            WorldModelBenchmarkResult(
                batch_size=batch_size,
                chunk_length=chunk_length,
                status="ok",
                warmup_batches=0,
                measured_batches=1,
                measured_samples=batch_size,
                elapsed_seconds=1.0,
                batches_per_second=1.0,
                samples_per_second=float(batch_size),
                tokens_per_second=float(batch_size * chunk_length),
                avg_fetch_ms=1.0,
                avg_transfer_ms=2.0,
                avg_step_ms=3.0,
                peak_memory_mb=None,
                last_loss=0.1,
            ),
            (3, 8, 8),
            None,
        )

    monkeypatch.setattr(benchmark_mod, "build_index", fake_build_index)
    monkeypatch.setattr(benchmark_mod, "_run_single_benchmark", fake_run_single_benchmark)

    benchmark_world_model(
        WorldModelBenchmarkConfig(
            run_name="wm-bench-state",
            data=WorldModelDataConfig(dataset_paths=["datasets/world_model/demo"]),
            batch_sizes=[1],
            chunk_lengths=[2],
            warmup_batches=0,
            measure_batches=1,
        ),
        show_progress=False,
    )

    assert captured_state["status"] == "running"
    assert captured_state["current_candidate"]["stage"] == "measuring"
    assert captured_state["current_candidate"]["candidate_index"] == 1
