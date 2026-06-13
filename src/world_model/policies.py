from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from src.agents import build_agent
from src.utils.run_paths import RunPaths


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


def _load_run_config_payload(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config for checkpoint validation: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_ppo_checkpoint(cfg, checkpoint_path: str | None = None) -> tuple[str, str]:
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint}")
        run_dir = checkpoint.parent.parent
        return str(checkpoint), str(run_dir)

    latest_pointer = RunPaths(
        study_id=cfg.study_id,
        exp_id=cfg.exp_id,
        trial_number=None,
        seed=cfg.seed,
    ).latest_pointer_path
    if not latest_pointer.exists():
        raise FileNotFoundError(
            f"Could not resolve latest PPO run for study={cfg.study_id} exp_id={cfg.exp_id}. "
            f"Missing {latest_pointer}."
        )
    with open(latest_pointer, "r", encoding="utf-8") as handle:
        latest = json.load(handle)
    run_dir = Path(latest["run_dir"])
    checkpoint = run_dir / "checkpoints" / "ppo_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resolved PPO checkpoint does not exist: {checkpoint}")
    return str(checkpoint), str(run_dir)


def validate_ppo_checkpoint_compatibility(cfg, envs, checkpoint_path: str) -> dict:
    checkpoint = Path(checkpoint_path)
    run_dir = checkpoint.parent.parent
    payload = _load_run_config_payload(run_dir / "config.yaml")
    saved_args = payload.get("args", {})
    saved_env = saved_args.get("env", {})

    mismatches: list[str] = []

    def check(label: str, current, saved) -> None:
        if current != saved:
            mismatches.append(f"{label}: current={current!r} checkpoint={saved!r}")

    check("study_id", cfg.study_id, saved_args.get("study_id"))
    check("exp_id", cfg.exp_id, saved_args.get("exp_id"))
    check("algorithm", cfg.algorithm, saved_args.get("algorithm"))
    check("env.obs_mode", cfg.env.obs_mode, saved_env.get("obs_mode"))
    check("env.semantic_mask_ch", cfg.env.semantic_mask_ch, saved_env.get("semantic_mask_ch"))
    check("env.temporal_fusion_mode", cfg.env.temporal_fusion_mode, saved_env.get("temporal_fusion_mode"))
    check("env.frame_stack", cfg.env.frame_stack, saved_env.get("frame_stack"))
    check("env.action_mode", cfg.env.action_mode, saved_env.get("action_mode"))
    check("env.action_profile_id", cfg.env.action_profile_id, saved_env.get("action_profile_id"))
    check("env.fov_masked", cfg.env.fov_masked, saved_env.get("fov_masked"))
    check("env.ego_anchor_x_frac", cfg.env.ego_anchor_x_frac, saved_env.get("ego_anchor_x_frac"))
    check("env.ego_anchor_y_frac", cfg.env.ego_anchor_y_frac, saved_env.get("ego_anchor_y_frac"))
    check("env.obs_size", tuple(cfg.env.obs_size), tuple(saved_env.get("obs_size", ()) or ()))

    current_obs_shape = tuple(envs.single_observation_space.shape)
    saved_obs_mode = saved_env.get("obs_mode")
    saved_masked = saved_obs_mode == "bev_semantic"
    saved_frame_stack = int(saved_env.get("frame_stack", 1))
    if saved_env.get("obs_mode") == "vector":
        expected_saved_shape = (7,)
    else:
        if saved_masked:
            channels_by_mode = {
                "binary": 1,
                "2-class": 2,
                "4-class": 4,
                "5-class": 5,
                "6-class": 6,
                "7-class": 7,
            }
            base_channels = channels_by_mode[str(saved_env.get("semantic_mask_ch"))]
        else:
            base_channels = 1
        temporal_mode = saved_env.get("temporal_fusion_mode", "stack")
        if temporal_mode == "stack":
            expected_saved_shape = (
                base_channels * saved_frame_stack,
                int(saved_env["obs_size"][0]),
                int(saved_env["obs_size"][1]),
            )
        else:
            expected_saved_shape = (
                base_channels + saved_frame_stack - 1,
                int(saved_env["obs_size"][0]),
                int(saved_env["obs_size"][1]),
            )
    if current_obs_shape != tuple(expected_saved_shape):
        mismatches.append(
            f"observation_shape: current={current_obs_shape!r} checkpoint={tuple(expected_saved_shape)!r}"
        )

    current_action_space = envs.single_action_space
    saved_action_mode = saved_env.get("action_mode")
    if saved_action_mode == "discrete":
        if not isinstance(current_action_space, gym.spaces.Discrete):
            mismatches.append(
                f"action_space_type: current={type(current_action_space).__name__!r} checkpoint='Discrete'"
            )
    elif saved_action_mode == "continuous":
        if not isinstance(current_action_space, gym.spaces.Box):
            mismatches.append(
                f"action_space_type: current={type(current_action_space).__name__!r} checkpoint='Box'"
            )

    if mismatches:
        joined = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(
            "PPO checkpoint is incompatible with the requested dataset configuration:\n"
            f"{joined}\n"
            f"checkpoint={checkpoint}"
        )

    return {"checkpoint_path": str(checkpoint), "source_run_dir": str(run_dir)}


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
