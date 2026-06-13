from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from CarlaBEV.envs import make_env

from src.carlabev_lab.simulator.signatures import extract_scene_route_metadata
from src.config.base_config import to_carlabev_run_config
from src.config.reset_protocol import build_train_protocol_sampler
from src.world_model.dataset_schema import (
    DatasetCollectionSummary,
    DatasetShardSummary,
    summary_path,
)
from src.world_model.policies import (
    build_policy,
    resolve_ppo_checkpoint,
    validate_ppo_checkpoint_compatibility,
)


def _normalize_action(action: Any) -> np.ndarray:
    arr = np.asarray(action)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


@dataclass
class _ShardBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    next_obs: list[np.ndarray] = field(default_factory=list)
    env_index: list[int] = field(default_factory=list)
    episode_id: list[int] = field(default_factory=list)
    step_in_episode: list[int] = field(default_factory=list)
    protocol_id: list[str] = field(default_factory=list)
    reset_seed: list[int] = field(default_factory=list)
    route_signature: list[str] = field(default_factory=list)
    scene_signature: list[str] = field(default_factory=list)
    straight_fraction: list[float] = field(default_factory=list)
    left_turn_fraction: list[float] = field(default_factory=list)
    right_turn_fraction: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        obs: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        next_obs: np.ndarray,
        env_index: int,
        episode_id: int,
        step_in_episode: int,
        protocol_id: str,
        reset_seed: int | None,
        route_signature: str,
        scene_signature: str,
        straight_fraction: float,
        left_turn_fraction: float,
        right_turn_fraction: float,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(_normalize_action(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self.env_index.append(int(env_index))
        self.episode_id.append(int(episode_id))
        self.step_in_episode.append(int(step_in_episode))
        self.protocol_id.append(protocol_id)
        self.reset_seed.append(-1 if reset_seed is None else int(reset_seed))
        self.route_signature.append(route_signature)
        self.scene_signature.append(scene_signature)
        self.straight_fraction.append(float(straight_fraction))
        self.left_turn_fraction.append(float(left_turn_fraction))
        self.right_turn_fraction.append(float(right_turn_fraction))

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.next_obs.clear()
        self.env_index.clear()
        self.episode_id.clear()
        self.step_in_episode.clear()
        self.protocol_id.clear()
        self.reset_seed.clear()
        self.route_signature.clear()
        self.scene_signature.clear()
        self.straight_fraction.clear()
        self.left_turn_fraction.clear()
        self.right_turn_fraction.clear()

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "obs": np.stack(self.obs, axis=0),
            "actions": np.stack(self.actions, axis=0),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=bool),
            "terminated": np.asarray(self.terminated, dtype=bool),
            "truncated": np.asarray(self.truncated, dtype=bool),
            "next_obs": np.stack(self.next_obs, axis=0),
            "env_index": np.asarray(self.env_index, dtype=np.int32),
            "episode_id": np.asarray(self.episode_id, dtype=np.int32),
            "step_in_episode": np.asarray(self.step_in_episode, dtype=np.int32),
            "protocol_id": np.asarray(self.protocol_id, dtype="<U64"),
            "reset_seed": np.asarray(self.reset_seed, dtype=np.int64),
            "route_signature": np.asarray(self.route_signature, dtype="<U32"),
            "scene_signature": np.asarray(self.scene_signature, dtype="<U32"),
            "straight_fraction": np.asarray(self.straight_fraction, dtype=np.float32),
            "left_turn_fraction": np.asarray(self.left_turn_fraction, dtype=np.float32),
            "right_turn_fraction": np.asarray(self.right_turn_fraction, dtype=np.float32),
        }


def default_output_dir(cfg, *, split: str, dataset_name: str) -> Path:
    return (
        Path("datasets")
        / "world_model"
        / dataset_name
        / cfg.study_id
        / f"exp_{cfg.exp_id}"
        / split
        / f"seed_{cfg.seed}"
    )


def _write_shard(buffer: _ShardBuffer, output_dir: Path, shard_index: int) -> DatasetShardSummary:
    shard_path = output_dir / f"shard_{shard_index:06d}.npz"
    np.savez_compressed(shard_path, **buffer.to_arrays())
    return DatasetShardSummary(
        shard_index=shard_index,
        transitions=len(buffer),
        path=str(shard_path),
    )


def _current_env_metadata(envs, env_index: int) -> dict[str, Any]:
    if not hasattr(envs, "envs"):
        return {}
    try:
        env = envs.envs[env_index]
    except (AttributeError, IndexError):
        return {}
    base_env = getattr(env, "unwrapped", env)
    if not hasattr(base_env, "map") or not hasattr(base_env.map, "route"):
        return {}
    try:
        return extract_scene_route_metadata(base_env)
    except Exception:
        return {}


def collect_dataset(
    cfg,
    *,
    total_transitions: int,
    steps_per_shard: int,
    split: str = "train",
    dataset_name: str = "default",
    output_dir: str | None = None,
    policy: str = "random",
    checkpoint_path: str | None = None,
    device: str = "cpu",
    show_progress: bool = True,
) -> DatasetCollectionSummary:
    if total_transitions <= 0:
        raise ValueError("`total_transitions` must be positive.")
    if steps_per_shard <= 0:
        raise ValueError("`steps_per_shard` must be positive.")

    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(
        cfg, split=split, dataset_name=dataset_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = make_env(to_carlabev_run_config(cfg), eval=False)
    sampler = build_train_protocol_sampler(cfg)
    checkpoint_info = {"checkpoint_path": None, "source_run_dir": None}
    if policy == "ppo":
        resolved_checkpoint_path, _resolved_run_dir = resolve_ppo_checkpoint(
            cfg, checkpoint_path=checkpoint_path
        )
        checkpoint_info = validate_ppo_checkpoint_compatibility(
            cfg, envs, resolved_checkpoint_path
        )
        checkpoint_path = checkpoint_info["checkpoint_path"]
    policy_adapter = build_policy(
        policy_name=policy,
        cfg=cfg,
        envs=envs,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    collected = 0
    shard_index = 0
    shard_summaries: list[DatasetShardSummary] = []
    buffer = _ShardBuffer()
    num_envs = getattr(cfg, "num_envs", 1)
    episode_ids = [0 for _ in range(num_envs)]
    episode_steps = [0 for _ in range(num_envs)]

    options = sampler.initial_options(num_envs)
    reset_seeds = sampler.initial_reset_seeds(num_envs)
    obs, _ = envs.reset(seed=reset_seeds, options=options)
    episode_metadata = [_current_env_metadata(envs, index) for index in range(num_envs)]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} transitions"),
        TextColumn("shards={task.fields[shards]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
    ) as progress:
        task_id = progress.add_task(
            "Collecting world-model data",
            total=total_transitions,
            shards=0,
        )
        try:
            while collected < total_transitions:
                actions = policy_adapter.act(obs)
                next_obs, reward, terminated, truncated, _info = envs.step(actions)

                reward = np.asarray(reward, dtype=np.float32)
                terminated = np.asarray(terminated, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool)
                done = np.logical_or(terminated, truncated)

                remaining = total_transitions - collected
                rows_to_take = min(num_envs, remaining)
                protocol_id = str(options.get("protocol_id", sampler.protocol.protocol_id))

                for env_index in range(rows_to_take):
                    buffer.append(
                        obs=obs[env_index],
                        action=actions[env_index],
                        reward=float(reward[env_index]),
                        done=bool(done[env_index]),
                        terminated=bool(terminated[env_index]),
                        truncated=bool(truncated[env_index]),
                        next_obs=next_obs[env_index],
                        env_index=env_index,
                        episode_id=episode_ids[env_index],
                        step_in_episode=episode_steps[env_index],
                        protocol_id=protocol_id,
                        reset_seed=reset_seeds[env_index] if reset_seeds is not None else None,
                        route_signature=str(episode_metadata[env_index].get("route_signature", "")),
                        scene_signature=str(episode_metadata[env_index].get("scene_signature", "")),
                        straight_fraction=float(episode_metadata[env_index].get("straight_fraction", np.nan)),
                        left_turn_fraction=float(episode_metadata[env_index].get("left_turn_fraction", np.nan)),
                        right_turn_fraction=float(episode_metadata[env_index].get("right_turn_fraction", np.nan)),
                    )
                    collected += 1
                    progress.update(task_id, completed=collected)
                    if len(buffer) >= steps_per_shard:
                        shard_summaries.append(_write_shard(buffer, out_dir, shard_index))
                        shard_index += 1
                        buffer.clear()
                        progress.update(task_id, shards=shard_index)
                    episode_steps[env_index] += 1
                    if done[env_index]:
                        episode_ids[env_index] += 1
                        episode_steps[env_index] = 0

                if len(buffer) > 0 and collected >= total_transitions:
                    shard_summaries.append(_write_shard(buffer, out_dir, shard_index))
                    shard_index += 1
                    buffer.clear()
                    progress.update(task_id, shards=shard_index)

                obs = next_obs
                if np.any(done):
                    options = sampler.next_options(reset_mask=done.copy())
                    reset_seeds = sampler.next_reset_seeds(done.copy())
                    obs, _ = envs.reset(seed=reset_seeds, options=options)
                    episode_metadata = [_current_env_metadata(envs, index) for index in range(num_envs)]
        finally:
            envs.close()

    summary = DatasetCollectionSummary(
        output_dir=str(out_dir),
        total_transitions=collected,
        shard_count=len(shard_summaries),
        shards=shard_summaries,
        policy=policy,
        split=split,
        study_id=cfg.study_id,
        exp_id=cfg.exp_id,
        seed=cfg.seed,
        checkpoint_path=checkpoint_info["checkpoint_path"],
        source_run_dir=checkpoint_info["source_run_dir"],
    )
    with open(summary_path(out_dir), "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, sort_keys=True)
    return summary
