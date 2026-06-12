from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

from .cnn_ppo import ContinuousConvolutionalPPO, DiscreteConvolutionalPPO


@dataclass
class PPOArtifacts:
    agent: nn.Module
    optimizer: Optimizer


def _build_cnn_ppo_agent(args, envs, device) -> PPOArtifacts:
    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    channels = getattr(args.ppo, "channels", [32, 64, 64])
    fc_size = getattr(args.ppo, "fc_size", 512)

    if is_continuous:
        agent = ContinuousConvolutionalPPO(
            envs, channels=channels, fc_size=fc_size
        ).to(device)
    else:
        agent = DiscreteConvolutionalPPO(
            envs, channels=channels, fc_size=fc_size
        ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.ppo.learning_rate, eps=1e-5)
    return PPOArtifacts(agent=agent, optimizer=optimizer)


def build_agent(args, envs, device):
    algorithm_builders = {
        "cnn-ppo": _build_cnn_ppo_agent,
    }

    try:
        builder = algorithm_builders[args.algorithm]
    except KeyError as exc:
        available = ", ".join(sorted(algorithm_builders))
        raise ValueError(
            f"Unsupported agent algorithm '{args.algorithm}'. "
            f"Available algorithms: {available}"
        ) from exc

    return builder(args, envs, device)
