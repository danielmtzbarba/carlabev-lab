import pytest

import src.trainers.ppo as ppo_mod
from src.config.studies.registry import get_train_protocol


@pytest.mark.integration
def test_train_ppo_smoke(monkeypatch, tiny_cfg, discrete_env, dummy_logger, fixed_eval_payload, tmp_workdir):
    class Sampler:
        def __init__(self, protocol):
            self.protocol = protocol

        def initial_options(self, num_envs):
            return {"reset_mask": [True] * num_envs}

        def next_options(self, reset_mask, mean_return=None, curriculum_state=None):
            del mean_return, curriculum_state
            return {"reset_mask": reset_mask}

    protocol = get_train_protocol(tiny_cfg.study_id, "traffic_route_curriculum_train")
    monkeypatch.setattr(ppo_mod, "build_train_protocol_sampler", lambda _cfg: Sampler(protocol))
    monkeypatch.setattr(ppo_mod, "evaluate_ppo", lambda *args, **kwargs: fixed_eval_payload)

    score = ppo_mod.train_ppo(tiny_cfg, discrete_env, dummy_logger, device="cpu", trial=None)

    assert score is None
    assert tiny_cfg.ppo.batch_size == 8
    assert tiny_cfg.ppo.minibatch_size == 8
    assert tiny_cfg.ppo.num_iterations == 1
    assert dummy_logger.episode_logs
    assert len(dummy_logger.evaluation_logs) == 2
    assert (tmp_workdir / "runs" / tiny_cfg.study_id / f"exp_{tiny_cfg.exp_id}" / "trial_manual" / f"seed_{tiny_cfg.seed}" / "checkpoints").exists()
