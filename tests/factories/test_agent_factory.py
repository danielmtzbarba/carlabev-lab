import pytest

from src.agents import PPOArtifacts, build_agent
from src.agents.cnn_ppo import ContinuousConvolutionalPPO, DiscreteConvolutionalPPO


@pytest.mark.unit
def test_build_agent_returns_discrete_ppo_artifacts(tiny_cfg, discrete_env):
    artifacts = build_agent(tiny_cfg, discrete_env, device="cpu")

    assert isinstance(artifacts, PPOArtifacts)
    assert isinstance(artifacts.agent, DiscreteConvolutionalPPO)


@pytest.mark.unit
def test_build_agent_returns_continuous_ppo_artifacts(tiny_cfg, continuous_env):
    artifacts = build_agent(tiny_cfg, continuous_env, device="cpu")

    assert isinstance(artifacts, PPOArtifacts)
    assert isinstance(artifacts.agent, ContinuousConvolutionalPPO)


@pytest.mark.unit
def test_build_agent_rejects_unsupported_algorithm(tiny_cfg, discrete_env):
    tiny_cfg.algorithm = "vector-ppo"

    with pytest.raises(ValueError, match="Unsupported agent algorithm"):
        build_agent(tiny_cfg, discrete_env, device="cpu")
