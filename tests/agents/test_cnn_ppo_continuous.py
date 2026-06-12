import pytest
import torch

from src.agents import build_agent


@pytest.mark.unit
def test_continuous_agent_output_shapes_and_bounds(tiny_cfg, continuous_env):
    artifacts = build_agent(tiny_cfg, continuous_env, device="cpu")
    obs = torch.zeros((continuous_env.num_envs,) + continuous_env.single_observation_space.shape)

    raw_action, action, logprob, entropy, value = artifacts.agent.get_action_and_value(obs)

    assert raw_action.shape == (continuous_env.num_envs, 3)
    assert action.shape == (continuous_env.num_envs, 3)
    assert logprob.shape == (continuous_env.num_envs,)
    assert entropy.shape == (continuous_env.num_envs,)
    assert value.shape == (continuous_env.num_envs, 1)
    assert torch.all(action[:, 0] >= 0.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= -1.0)
    assert torch.all(action[:, 1] <= 1.0)
    assert torch.all(action[:, 2] >= 0.0)
    assert torch.all(action[:, 2] <= 1.0)
    assert torch.isfinite(logprob).all()
