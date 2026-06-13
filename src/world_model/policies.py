from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np
import torch

from src.agents import build_agent


class PolicyAdapter(Protocol):
    def act(self, obs_batch: np.ndarray) -> np.ndarray: ...


@dataclass
class RandomPolicyAdapter:
    action_space: gym.Space
    num_envs: int

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        del obs_batch
        return np.asarray(
            [self.action_space.sample() for _ in range(self.num_envs)]
        )


@dataclass
class PPOPolicyAdapter:
    agent: torch.nn.Module
    device: str

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.agent.get_action_and_value(obs_t)
            if self.agent.is_continuous:
                _, action, _, _, _ = out
            else:
                action, _, _, _ = out
        return action.cpu().numpy()


def build_policy(
    *,
    policy_name: str,
    cfg,
    envs,
    device: str,
    checkpoint_path: str | None = None,
) -> PolicyAdapter:
    if policy_name == "random":
        return RandomPolicyAdapter(
            action_space=envs.single_action_space,
            num_envs=getattr(cfg, "num_envs", 1),
        )

    if policy_name == "ppo":
        if checkpoint_path is None:
            raise ValueError("`checkpoint_path` is required when `policy='ppo'`.")
        artifacts = build_agent(cfg, envs, device)
        agent = artifacts.agent
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.eval()
        return PPOPolicyAdapter(agent=agent, device=device)

    raise ValueError(f"Unsupported policy {policy_name!r}.")
