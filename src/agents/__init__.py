from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.optim import Optimizer

from .cnn_ppo import ContinuousConvolutionalPPO, DiscreteConvolutionalPPO
from .dqn import QNetwork
from .ppo_vector import VectorPPO
from .sac import Actor, SoftQNetwork


@dataclass
class PPOArtifacts:
    agent: nn.Module
    optimizer: Optimizer


@dataclass
class DQNArtifacts:
    q_network: nn.Module
    optimizer: Optimizer
    target_network: nn.Module
    replay_buffer: ReplayBuffer


@dataclass
class SACArtifacts:
    actor: nn.Module
    qf1: nn.Module
    qf2: nn.Module
    qf1_target: nn.Module
    qf2_target: nn.Module
    q_optimizer: Optimizer
    actor_optimizer: Optimizer
    replay_buffer: ReplayBuffer


def _build_dqn_agent(args, envs, device) -> DQNArtifacts:
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    return DQNArtifacts(
        q_network=q_network,
        optimizer=optimizer,
        target_network=target_network,
        replay_buffer=replay_buffer,
    )


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


def _build_vector_ppo_agent(args, envs, device) -> PPOArtifacts:
    agent = VectorPPO(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    return PPOArtifacts(agent=agent, optimizer=optimizer)


def _build_sac_agent(args, envs, device) -> SACArtifacts:
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()), lr=args.policy_lr, eps=1e-4
    )

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    return SACArtifacts(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        replay_buffer=replay_buffer,
    )


def build_agent(args, envs, device):
    algorithm_builders = {
        "cnn-ppo": _build_cnn_ppo_agent,
        "vector-ppo": _build_vector_ppo_agent,
        "dqn": _build_dqn_agent,
        "sac": _build_sac_agent,
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
