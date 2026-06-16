from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import FakeVectorEnv

import src.carlabev_lab.world_model.train as train_cli_mod
from src.world_model.collector import collect_dataset
from src.world_model.config import (
    WorldModelConfig,
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from src.world_model.evaluate import WorldModelCheckpointEvalResult, WorldModelEvalMetrics
from src.world_model.train import train_world_model


def _collect_train_dataset(monkeypatch, tiny_cfg, tmp_path, *, total_transitions: int = 8):
    import src.world_model.collector as collector_mod

    env = FakeVectorEnv(num_envs=1, obs_shape=(3, 8, 8), terminate_every=4)
    monkeypatch.setattr(collector_mod, "make_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        collector_mod,
        "build_train_protocol_sampler",
        lambda _cfg: SimpleNamespace(
            protocol=SimpleNamespace(protocol_id="random_nav_train"),
            initial_options=lambda num_envs: {
                "reset_mask": np.ones((num_envs,), dtype=bool),
                "protocol_id": "random_nav_train",
            },
            next_options=lambda reset_mask, mean_return=None, curriculum_state=None: {
                "reset_mask": reset_mask,
                "protocol_id": "random_nav_train",
            },
            initial_reset_seeds=lambda num_envs: [1000 + i for i in range(num_envs)],
            next_reset_seeds=lambda reset_mask: [2000 + i for i in range(len(reset_mask))],
        ),
    )
    monkeypatch.setattr(
        collector_mod,
        "_current_env_metadata",
        lambda _envs, env_index: {
            "route_signature": f"route-{env_index}",
            "scene_signature": f"scene-{env_index}",
            "straight_fraction": 0.6,
            "left_turn_fraction": 0.2,
            "right_turn_fraction": 0.2,
        },
    )

    cfg = tiny_cfg
    cfg.num_envs = 1
    output_dir = tmp_path / "dataset"
    collect_dataset(
        cfg,
        total_transitions=total_transitions,
        steps_per_shard=4,
        output_dir=str(output_dir),
        policy="random",
        show_progress=False,
    )
    return output_dir


@pytest.mark.integration
def test_train_world_model_smoke(monkeypatch, tiny_cfg, tmp_workdir):
    output_dir = _collect_train_dataset(monkeypatch, tiny_cfg, tmp_workdir, total_transitions=8)

    result = train_world_model(
        WorldModelConfig(
            run_name="wm-smoke",
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
        ),
        show_progress=False,
    )

    run_dir = tmp_workdir / "runs" / "world_model" / "wm-smoke"
    assert result.epochs == 1
    assert (run_dir / "checkpoints" / "world_model_final.pt").exists()
    assert (run_dir / "checkpoints" / "world_model_best.pt").exists()
    assert (run_dir / "artifacts" / "validation_report.json").exists()
    assert (run_dir / "artifacts" / "history.json").exists()

    history = json.loads((run_dir / "artifacts" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["epoch"] == 1
    assert "train_avg_fetch_ms" in history[0]
    assert "train_avg_transfer_ms" in history[0]
    assert "train_avg_step_ms" in history[0]
    assert "val_avg_fetch_ms" in history[0]
    assert "val_avg_transfer_ms" in history[0]
    assert "val_avg_step_ms" in history[0]


@pytest.mark.unit
def test_world_model_train_cli_supports_exp_subcommand(monkeypatch):
    captured: dict[str, object] = {}

    def fake_train_world_model(cfg, *, show_progress=True):
        captured["cfg"] = cfg
        captured["show_progress"] = show_progress
        return SimpleNamespace(
            run_dir="runs/world_model/demo",
            best_checkpoint_path="runs/world_model/demo/checkpoints/world_model_best.pt",
            final_train_loss=0.1,
            final_val_loss=0.2,
        )

    monkeypatch.setattr(train_cli_mod, "train_world_model", fake_train_world_model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "src.carlabev_lab.world_model.train",
            "exp",
            "--study-id",
            "PPO_NAVIGATION_DIFFICULTY",
            "--exp-id",
            "1",
            "--seed",
            "2",
        ],
    )

    train_cli_mod.main()

    cfg = captured["cfg"]
    assert cfg.data.batch_size == 32
    assert cfg.data.chunk_length == 4
    assert cfg.training.device == "cuda"


@pytest.mark.unit
def test_ensure_tmp_prepared_dataset_caches_builds_missing_cache(monkeypatch):
    import src.world_model.train as train_mod

    calls: list[str] = []
    monkeypatch.setattr(train_mod, "is_tmp_dataset_path", lambda path: path == "/tmp/demo")
    monkeypatch.setattr(train_mod, "has_complete_prepared_shard_cache", lambda _path: False)
    monkeypatch.setattr(train_mod, "prepare_dataset_shard_cache", lambda path: calls.append(path))

    train_mod._ensure_tmp_prepared_dataset_caches(["/tmp/demo", "/data/horse/ws/demo"])

    assert calls == ["/tmp/demo"]


@pytest.mark.unit
def test_ensure_tmp_prepared_dataset_caches_skips_ready_cache(monkeypatch):
    import src.world_model.train as train_mod

    calls: list[str] = []
    monkeypatch.setattr(train_mod, "is_tmp_dataset_path", lambda _path: True)
    monkeypatch.setattr(train_mod, "has_complete_prepared_shard_cache", lambda _path: True)
    monkeypatch.setattr(train_mod, "prepare_dataset_shard_cache", lambda path: calls.append(path))

    train_mod._ensure_tmp_prepared_dataset_caches(["/tmp/demo"])

    assert calls == []


@pytest.mark.integration
def test_train_world_model_records_study_results_in_sqlite(monkeypatch, tiny_cfg, tmp_workdir):
    output_dir = _collect_train_dataset(monkeypatch, tiny_cfg, tmp_workdir, total_transitions=8)

    def fake_eval_checkpoint(_cfg):
        metrics = WorldModelEvalMetrics(
            loss=0.4,
            pred_loss=0.3,
            reg_loss=1.0,
            cosine_similarity=0.6,
            latent_rmse=8.0,
            explained_variance=0.45,
            top1_retrieval=0.12,
            top5_retrieval=0.5,
            avg_fetch_ms=10.0,
            avg_transfer_ms=2.0,
            avg_step_ms=20.0,
            batches=3,
            action_metrics={"action_0": {"count": 8.0, "pred_loss": 1.0, "cosine_similarity": 0.7, "latent_rmse": 2.0, "explained_variance": 0.6, "top1_retrieval": 0.2, "top5_retrieval": 0.7}},
            route_metrics={"straight": {"count": 8.0, "pred_loss": 1.0, "cosine_similarity": 0.7, "latent_rmse": 2.0, "explained_variance": 0.6, "top1_retrieval": 0.2, "top5_retrieval": 0.7}},
        )
        return WorldModelCheckpointEvalResult(
            checkpoint_path="runs/world_model/wm-study/checkpoints/world_model_best.pt",
            output_path="runs/world_model/wm-study/artifacts/eval_checkpoint.json",
            dataset_paths=(str(output_dir),),
            source_names=("demo:source",),
            device="cpu",
            run_dir="runs/world_model/wm-study",
            chunk_length=2,
            batch_size=2,
            train_metrics=metrics,
            val_metrics=metrics,
        )

    monkeypatch.setattr("src.world_model.train.evaluate_world_model_checkpoint", fake_eval_checkpoint)

    result = train_world_model(
        WorldModelConfig(
            run_name="wm-study",
            study_id="WM_DATA_PHASE1",
            exp_id=5,
            seed=2,
            experiment_name="WM_DATA_PPO_MEDIUM",
            results_db_path="results/world_model/wm_data_phase1_runs.db",
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
        ),
        show_progress=False,
    )

    db_path = tmp_workdir / "results" / "world_model" / "wm_data_phase1_runs.db"
    assert result.run_dir.endswith("runs/world_model/wm-study")
    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT study_id, exp_id, experiment_name, val_cosine_similarity, val_top5_retrieval FROM world_model_study_runs WHERE run_name = ?",
        ("wm-study",),
    ).fetchone()
    conn.close()
    assert row == ("WM_DATA_PHASE1", 5, "WM_DATA_PPO_MEDIUM", 0.6, 0.5)
