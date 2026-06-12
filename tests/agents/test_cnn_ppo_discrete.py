import pytest
import torch

from src.agents import build_agent


@pytest.mark.unit
def test_discrete_agent_output_shapes(tiny_cfg, discrete_env):
    artifacts = build_agent(tiny_cfg, discrete_env, device="cpu")
    obs = torch.zeros((discrete_env.num_envs,) + discrete_env.single_observation_space.shape)

    action, logprob, entropy, value = artifacts.agent.get_action_and_value(obs)

    assert action.shape == (discrete_env.num_envs,)
    assert logprob.shape == (discrete_env.num_envs,)
    assert entropy.shape == (discrete_env.num_envs,)
    assert value.shape == (discrete_env.num_envs, 1)


@pytest.mark.unit
def test_discrete_agent_deterministic_action_matches_argmax(tiny_cfg, discrete_env):
    artifacts = build_agent(tiny_cfg, discrete_env, device="cpu")
    obs = torch.zeros((discrete_env.num_envs,) + discrete_env.single_observation_space.shape)

    action, _, _, _ = artifacts.agent.get_action_and_value(obs, deterministic=True)
    logits = artifacts.agent.actor(artifacts.agent.backbone(obs))

    assert torch.equal(action, torch.argmax(logits, dim=-1))
