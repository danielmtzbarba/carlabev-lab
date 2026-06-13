from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import FakeVectorEnv

from src.world_model.collector import collect_dataset
from src.world_model.config import (
    WorldModelConfig,
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
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
        )
    )

    run_dir = tmp_workdir / "runs" / "world_model" / "wm-smoke"
    assert result.epochs == 1
    assert (run_dir / "checkpoints" / "world_model_final.pt").exists()
    assert (run_dir / "checkpoints" / "world_model_best.pt").exists()
    assert (run_dir / "artifacts" / "validation_report.json").exists()
    assert (run_dir / "artifacts" / "history.json").exists()

    history = json.loads((run_dir / "artifacts" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["epoch"] == 1
