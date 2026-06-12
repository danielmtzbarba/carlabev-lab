from types import SimpleNamespace

import optuna
import pytest

import src.tuning.engine as engine_mod
from src.config.studies.registry import get_study_config


def _cli_args(**overrides):
    payload = {
        "study_id": "PPO_NAVIGATION",
        "exp_id": 26,
        "stage": "policy_dynamics",
        "n_trials": None,
        "total_timesteps": None,
        "eval_episodes": None,
        "eval_final_episodes": None,
        "num_seeds": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


@pytest.mark.unit
def test_normalize_stage_name_accepts_explicit_and_legacy_aliases():
    assert engine_mod.normalize_stage_name("policy_dynamics") == "policy_dynamics"
    assert engine_mod.normalize_stage_name("1") == "policy_dynamics"
    assert engine_mod.normalize_stage_name("2a") == "rollout_geometry"


@pytest.mark.unit
def test_resolved_stage_config_uses_study_defaults_and_cli_overrides():
    tuning = get_study_config("PPO_NAVIGATION").tuning
    assert tuning is not None

    tuning_cfg, stage_cfg = engine_mod.resolved_stage_config(
        tuning,
        "policy_dynamics",
        _cli_args(n_trials=12, total_timesteps=345, num_seeds=2, eval_episodes=7),
    )

    assert stage_cfg.n_trials == 12
    assert stage_cfg.total_timesteps == 345
    assert tuning_cfg.num_seeds == 2
    assert tuning_cfg.eval_episodes == 7
    assert tuning_cfg.eval_final_episodes == tuning.eval_final_episodes


@pytest.mark.unit
def test_run_stage_objective_mutates_policy_dynamics(monkeypatch, tiny_cfg, dummy_trial):
    captured = []
    tuning = get_study_config("PPO_NAVIGATION").tuning
    assert tuning is not None
    tuning_cfg, stage_cfg = engine_mod.resolved_stage_config(tuning, "policy_dynamics", _cli_args())

    monkeypatch.setattr(engine_mod, "get_study_db_path", lambda _study_id: "results/test.db")
    monkeypatch.setattr(
        engine_mod,
        "run_experiment",
        lambda args, trial, seed_idx: captured.append((args, trial, seed_idx)) or 42.0,
    )

    score = engine_mod.run_stage_objective(
        dummy_trial,
        base_args=tiny_cfg,
        study_id="PPO_NAVIGATION",
        tuning_config=tuning_cfg,
        stage_cfg=stage_cfg,
        inherited_overrides={},
    )

    assert score == pytest.approx(42.0)
    args = captured[0][0]
    assert args.ppo.learning_rate == pytest.approx(2e-4)
    assert args.ppo.gae_lambda == pytest.approx(0.95)
    assert args.ppo.gamma == pytest.approx(0.99)
    assert args.logging.db_path == "results/test.db"
    assert args.save_model is False
    assert args.capture_video is False
    assert dummy_trial.user_attrs["tuning_stage"] == "policy_dynamics"


@pytest.mark.unit
def test_run_stage_objective_applies_inherited_and_sampled_overrides(monkeypatch, tiny_cfg, dummy_trial):
    captured = []
    tuning = get_study_config("PPO_NAVIGATION").tuning
    assert tuning is not None
    tuning_cfg, stage_cfg = engine_mod.resolved_stage_config(tuning, "rollout_geometry", _cli_args(stage="rollout_geometry"))

    monkeypatch.setattr(engine_mod, "get_study_db_path", lambda _study_id: "results/test.db")
    monkeypatch.setattr(
        engine_mod,
        "run_experiment",
        lambda args, trial, seed_idx: captured.append((args, trial, seed_idx)) or 33.0,
    )

    score = engine_mod.run_stage_objective(
        dummy_trial,
        base_args=tiny_cfg,
        study_id="PPO_NAVIGATION",
        tuning_config=tuning_cfg,
        stage_cfg=stage_cfg,
        inherited_overrides={
            "ppo.learning_rate": 1e-4,
            "ppo.gae_lambda": 0.97,
            "ppo.gamma": 0.995,
        },
    )

    assert score == pytest.approx(33.0)
    args = captured[0][0]
    assert args.ppo.learning_rate == pytest.approx(1e-4)
    assert args.ppo.gae_lambda == pytest.approx(0.97)
    assert args.ppo.gamma == pytest.approx(0.995)
    assert args.ppo.num_steps == 128
    assert args.ppo.update_epochs == 4
    assert args.ppo.num_minibatches == 2
    assert dummy_trial.user_attrs["tuning_stage"] == "rollout_geometry"


@pytest.mark.unit
def test_inherited_stage_overrides_uses_best_completed_trials():
    tuning = get_study_config("PPO_NAVIGATION").tuning
    assert tuning is not None
    stage_cfg = tuning.stages["loss_regularization"]

    trial_policy = optuna.trial.create_trial(
        params={"learning_rate": 1e-4, "gae_lambda": 0.97, "gamma": 0.995},
        distributions={
            "learning_rate": optuna.distributions.FloatDistribution(5e-5, 5e-4, log=True),
            "gae_lambda": optuna.distributions.FloatDistribution(0.9, 0.99),
            "gamma": optuna.distributions.FloatDistribution(0.98, 0.9995),
        },
        value=10.0,
        user_attrs={"tuning_stage": "policy_dynamics"},
    )
    trial_rollout = optuna.trial.create_trial(
        params={"num_steps": 256, "update_epochs": 6, "num_minibatches": 4},
        distributions={
            "num_steps": optuna.distributions.CategoricalDistribution([128, 256, 512]),
            "update_epochs": optuna.distributions.IntDistribution(3, 8),
            "num_minibatches": optuna.distributions.CategoricalDistribution([2, 4, 8]),
        },
        value=9.0,
        user_attrs={"tuning_stage": "rollout_geometry"},
    )

    study = optuna.create_study(direction="maximize")
    study.add_trial(trial_policy)
    study.add_trial(trial_rollout)

    overrides = engine_mod.inherited_stage_overrides(study, tuning, stage_cfg)

    assert overrides["ppo.learning_rate"] == pytest.approx(1e-4)
    assert overrides["ppo.num_steps"] == 256
