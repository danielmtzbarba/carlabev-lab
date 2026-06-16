from __future__ import annotations

import pytest

from src.world_model.experiment_loader import (
    WorldModelTrainExperimentArgs,
    build_world_model_train_config_from_experiment,
)


@pytest.mark.unit
def test_build_world_model_train_config_from_experiment_uses_difficulty_preset():
    cfg = build_world_model_train_config_from_experiment(
        WorldModelTrainExperimentArgs(
            study_id="PPO_NAVIGATION_DIFFICULTY",
            exp_id=1,
            seed=2,
        )
    )

    assert cfg.run_name == "wm-ppo_navigation_difficulty-exp1-seed2-lewm-ppo-difficulty-hpc"
    assert cfg.data.dataset_paths == [
        "datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_1/train/seed_2"
    ]
    assert cfg.data.batch_size == 32
    assert cfg.data.chunk_length == 4
    assert cfg.data.num_workers == 0
    assert cfg.data.pin_memory is True
    assert cfg.training.device == "cuda"
    assert cfg.training.amp is True
    assert cfg.training.amp_dtype == "bfloat16"
    assert cfg.training.epochs == 5


@pytest.mark.unit
def test_build_world_model_train_config_from_experiment_allows_overrides():
    cfg = build_world_model_train_config_from_experiment(
        WorldModelTrainExperimentArgs(
            study_id="PPO_NAVIGATION_DIFFICULTY",
            exp_id=3,
            seed=5,
            dataset_name="custom-dataset",
            dataset_path="/tmp/staged-seed-5",
            run_name="custom-run",
            batch_size=64,
            chunk_length=8,
            epochs=3,
            device="cpu",
            amp=False,
        )
    )

    assert cfg.run_name == "custom-run"
    assert cfg.data.dataset_paths == ["/tmp/staged-seed-5"]
    assert cfg.data.batch_size == 64
    assert cfg.data.chunk_length == 8
    assert cfg.training.epochs == 3
    assert cfg.training.device == "cpu"
    assert cfg.training.amp is False
