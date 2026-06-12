from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.eval.eval_ppo import _aggregate_protocol_results, _evenly_spaced_indices, evaluate_ppo


@pytest.mark.unit
def test_evenly_spaced_indices_handles_small_counts():
    assert _evenly_spaced_indices(5, 0) == []
    assert _evenly_spaced_indices(5, 1) == [0]
    assert _evenly_spaced_indices(3, 10) == [0, 1, 2]


@pytest.mark.unit
def test_aggregate_protocol_results_weights_by_episode_count():
    protocol_results = {
        "a": {"episodes": 1, "mean_return": 10.0, "std_return": 1.0, "mean_length": 2.0, "success_rate": 1.0, "collision_rate": 0.0, "unfinished_rate": 0.0, "mean_abs_accel_long": 0.1, "mean_abs_accel_lat": 0.1, "mean_abs_jerk_long": 0.1, "mean_abs_jerk_lat": 0.1, "mean_abs_yaw_rate": 0.1, "mean_abs_yaw_acc": 0.1, "comfort_violation_rate": 0.0, "harsh_brake_rate": 0.0},
        "b": {"episodes": 3, "mean_return": 4.0, "std_return": 2.0, "mean_length": 6.0, "success_rate": 0.0, "collision_rate": 0.5, "unfinished_rate": 0.5, "mean_abs_accel_long": 0.3, "mean_abs_accel_lat": 0.3, "mean_abs_jerk_long": 0.3, "mean_abs_jerk_lat": 0.3, "mean_abs_yaw_rate": 0.3, "mean_abs_yaw_acc": 0.3, "comfort_violation_rate": 0.5, "harsh_brake_rate": 0.25},
    }

    aggregate = _aggregate_protocol_results(protocol_results)

    assert aggregate["mean_return"] == pytest.approx(5.5)
    assert aggregate["success_rate"] == pytest.approx(0.25)
    assert aggregate["evaluated_protocol_ids"] == ["a", "b"]


@pytest.mark.integration
def test_evaluate_ppo_runs_with_fake_dependencies(monkeypatch, tiny_cfg, discrete_env, tmp_workdir):
    import src.eval.eval_ppo as eval_mod

    class DummyAgent:
        is_continuous = False

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return None

        def get_action_and_value(self, obs):
            batch = obs.shape[0]
            return (
                torch.zeros((batch,), dtype=torch.int64),
                torch.zeros((batch,), dtype=torch.float32),
                torch.zeros((batch,), dtype=torch.float32),
                torch.zeros((batch, 1), dtype=torch.float32),
            )

    monkeypatch.setattr(eval_mod, "make_env", lambda *_args, **_kwargs: discrete_env)
    monkeypatch.setattr(eval_mod, "build_agent", lambda *_args, **_kwargs: SimpleNamespace(agent=DummyAgent()))
    monkeypatch.setattr(
        eval_mod,
        "build_eval_protocol_samplers",
        lambda *_args, **_kwargs: {"random_nav_eval": SimpleNamespace(initial_options=lambda n: {"reset_mask": np.ones((n,), dtype=bool)}, next_options=lambda reset_mask: {"reset_mask": reset_mask})},
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})

    cfg = deepcopy(tiny_cfg)
    cfg.run_dir = str(tmp_workdir / "runs" / "eval-smoke")

    payload = evaluate_ppo(
        cfg=cfg,
        model_path="unused.pt",
        num_episodes=2,
        num_envs=discrete_env.num_envs,
        device="cpu",
        save_payload=False,
    )

    assert payload["aggregate"]["success_rate"] == pytest.approx(1.0)
    assert payload["protocols"]["random_nav_eval"]["episodes"] == 2
