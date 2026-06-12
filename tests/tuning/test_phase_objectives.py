from types import SimpleNamespace

import pytest

import src.tuning.phase1 as phase1_mod
import src.tuning.phase2a as phase2a_mod
import src.tuning.phase2b as phase2b_mod


def _opt_args():
    return SimpleNamespace(
        study_id="PPO_NAVIGATION",
        num_seeds=2,
        timesteps_phase_1=100,
        timesteps_phase_2a=200,
        timesteps_phase_2b=300,
        eval_episodes=5,
        eval_final_episodes=7,
    )


@pytest.mark.unit
def test_phase1_mutates_only_continuous_ppo_params(monkeypatch, tiny_cfg, dummy_trial):
    captured = []
    monkeypatch.setattr(phase1_mod, "get_study_db_path", lambda _study_id: "results/test.db")
    monkeypatch.setattr(phase1_mod, "run_experiment", lambda args, trial, seed_idx: captured.append((args, trial, seed_idx)) or 42.0)

    score = phase1_mod.phase_1_objective(dummy_trial, tiny_cfg, _opt_args())

    assert score == pytest.approx(42.0)
    assert len(captured) == 2
    first_args, trial, seed_idx = captured[0]
    assert trial is dummy_trial
    assert seed_idx == 0
    assert first_args.ppo.learning_rate == pytest.approx(2e-4)
    assert first_args.ppo.gae_lambda == pytest.approx(0.95)
    assert first_args.ppo.gamma == pytest.approx(0.99)
    assert first_args.logging.db_path == "results/test.db"
    assert first_args.capture_video is False
    assert first_args.save_model is False
    assert dummy_trial.user_attrs["phase"] == 1


@pytest.mark.unit
def test_phase2a_carries_phase1_params_and_mutates_rollout_shape(monkeypatch, tiny_cfg, dummy_trial):
    captured = []
    monkeypatch.setattr(phase2a_mod, "get_study_db_path", lambda _study_id: "results/test.db")
    monkeypatch.setattr(phase2a_mod, "run_experiment", lambda args, trial, seed_idx: captured.append((args, trial, seed_idx)) or 33.0)

    score = phase2a_mod.phase_2a_objective(
        dummy_trial,
        tiny_cfg,
        _opt_args(),
        {"learning_rate": 1e-4, "gae_lambda": 0.97, "gamma": 0.995},
    )

    assert score == pytest.approx(33.0)
    args = captured[0][0]
    assert args.ppo.learning_rate == pytest.approx(1e-4)
    assert args.ppo.gae_lambda == pytest.approx(0.97)
    assert args.ppo.gamma == pytest.approx(0.995)
    assert args.ppo.num_steps == 128
    assert args.ppo.update_epochs == 4
    assert args.ppo.num_minibatches == 2
    assert args.logging.db_path == "results/test.db"
    assert dummy_trial.user_attrs["phase"] == "2a"


@pytest.mark.unit
def test_phase2b_carries_prior_params_and_mutates_coefficients(monkeypatch, tiny_cfg, dummy_trial):
    captured = []
    monkeypatch.setattr(phase2b_mod, "run_experiment", lambda args, trial, seed_idx: captured.append((args, trial, seed_idx)) or 21.0)

    score = phase2b_mod.phase_2b_objective(
        dummy_trial,
        tiny_cfg,
        _opt_args(),
        {
            "learning_rate": 1e-4,
            "gae_lambda": 0.97,
            "gamma": 0.995,
            "num_steps": 256,
            "update_epochs": 6,
            "num_minibatches": 4,
        },
    )

    assert score == pytest.approx(21.0)
    args = captured[0][0]
    assert args.ppo.learning_rate == pytest.approx(1e-4)
    assert args.ppo.num_steps == 256
    assert args.ppo.update_epochs == 6
    assert args.ppo.num_minibatches == 4
    assert args.ppo.clip_coef_start == pytest.approx(0.2)
    assert args.ppo.ent_coef_start == pytest.approx(0.01)
    assert args.ppo.vf_coef_start == pytest.approx(0.6)
    assert args.ppo.max_grad_norm == pytest.approx(0.5)
    assert args.ppo.ent_decay_factor == pytest.approx(0.3)
    assert dummy_trial.user_attrs["phase"] == "2b"
