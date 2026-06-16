from __future__ import annotations

import pytest

from src.world_model.experiment_loader import (
    WorldModelTrainExperimentArgs,
    build_world_model_train_config_from_experiment,
    get_world_model_experiment_spec,
    get_world_model_study_config,
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
    assert cfg.results_db_path is None


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


@pytest.mark.unit
def test_world_model_phase1_study_has_expected_experiments():
    study = get_world_model_study_config("WM_DATA_PHASE1")

    assert sorted(study.experiments) == [1, 2, 3, 4, 5, 6, 7]
    assert get_world_model_experiment_spec("WM_DATA_PHASE1", 1).name == "WM_DATA_RANDOM"
    assert get_world_model_experiment_spec("WM_DATA_PHASE1", 7).name == "WM_DATA_PPO_MEDIUM_RANDOM"


@pytest.mark.unit
def test_build_world_model_train_config_from_world_model_study_supports_mixed_sources():
    cfg = build_world_model_train_config_from_experiment(
        WorldModelTrainExperimentArgs(
            study_id="WM_DATA_PHASE1",
            exp_id=4,
            seed=2,
        )
    )

    assert cfg.run_name == "wm-wm_data_phase1-exp4-seed2-wm-data-ppo-easy-random"
    assert cfg.data.dataset_paths == [
        "datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_2/train/seed_2",
        "datasets/world_model/lewm-random-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_3/train/seed_2",
    ]
    assert cfg.data.batch_size == 32
    assert cfg.data.chunk_length == 4
    assert cfg.data.pin_memory is True
    assert cfg.training.device == "cuda"
    assert cfg.training.epochs == 5
    assert cfg.experiment_name == "WM_DATA_PPO_EASY_RANDOM"
    assert cfg.results_db_path == "results/world_model/wm_data_phase1_runs.db"


@pytest.mark.unit
def test_build_world_model_train_config_from_world_model_study_accepts_dataset_paths_override():
    cfg = build_world_model_train_config_from_experiment(
        WorldModelTrainExperimentArgs(
            study_id="WM_DATA_PHASE1",
            exp_id=7,
            seed=5,
            dataset_paths=["/tmp/medium-ppo", "/tmp/medium-random"],
        )
    )

    assert cfg.data.dataset_paths == ["/tmp/medium-ppo", "/tmp/medium-random"]


@pytest.mark.unit
def test_build_world_model_train_config_from_world_model_study_rejects_single_path_for_mixed_experiment():
    with pytest.raises(ValueError, match="dataset_path"):
        build_world_model_train_config_from_experiment(
            WorldModelTrainExperimentArgs(
                study_id="WM_DATA_PHASE1",
                exp_id=6,
                seed=2,
                dataset_path="/tmp/only-one-root",
            )
        )
