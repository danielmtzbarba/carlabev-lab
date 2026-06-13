from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _load_dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    summary_file = dataset_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Expected summary.json in dataset directory: {dataset_dir}"
        )
    with open(summary_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_dataset_dir(path: str) -> tuple[Path, dict[str, Any]]:
    dataset_dir = Path(path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Path does not exist: {dataset_dir}")
    if dataset_dir.is_file():
        dataset_dir = dataset_dir.parent
    summary = _load_dataset_summary(dataset_dir)
    return dataset_dir, summary


def summarize_dataset(path: str) -> dict[str, Any]:
    dataset_dir, summary = _coerce_dataset_dir(path)
    shard_entries = summary.get("shards", [])
    if not shard_entries:
        raise ValueError(f"No shards listed in summary for {dataset_dir}")

    action_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()

    total_transitions = 0
    total_done = 0
    total_terminated = 0
    total_rewards = 0.0
    total_straight = 0.0
    total_left = 0.0
    total_right = 0.0
    max_step_by_episode: dict[tuple[int, int], int] = {}
    route_by_episode: dict[tuple[int, int], str] = {}
    scene_by_episode: dict[tuple[int, int], str] = {}

    for shard in shard_entries:
        shard_path = Path(shard["path"])
        with np.load(shard_path, allow_pickle=False) as data:
            rows = int(data["obs"].shape[0])
            total_transitions += rows
            total_done += int(np.count_nonzero(data["dones"]))
            total_terminated += int(np.count_nonzero(data["terminated"]))
            total_rewards += float(np.sum(data["rewards"]))
            total_straight += float(np.nansum(data["straight_fraction"]))
            total_left += float(np.nansum(data["left_turn_fraction"]))
            total_right += float(np.nansum(data["right_turn_fraction"]))

            actions = data["actions"]
            if actions.ndim == 2 and actions.shape[1] == 1:
                flat_actions = actions[:, 0]
            else:
                flat_actions = actions
            for action in flat_actions:
                action_counts[str(np.asarray(action).tolist())] += 1

            for route in data["route_signature"]:
                route_counts[str(route)] += 1
            for scene in data["scene_signature"]:
                scene_counts[str(scene)] += 1

            env_indices = data["env_index"]
            episode_ids = data["episode_id"]
            steps = data["step_in_episode"]
            routes = data["route_signature"]
            scenes = data["scene_signature"]
            for env_index, episode_id, step, route, scene in zip(
                env_indices,
                episode_ids,
                steps,
                routes,
                scenes,
                strict=False,
            ):
                key = (int(env_index), int(episode_id))
                max_step_by_episode[key] = max(max_step_by_episode.get(key, -1), int(step))
                route_by_episode.setdefault(key, str(route))
                scene_by_episode.setdefault(key, str(scene))

    total_episodes = len(max_step_by_episode)
    episode_lengths = [step + 1 for step in max_step_by_episode.values()]
    unique_routes = len(route_counts)
    unique_scenes = len(scene_counts)
    unique_episode_routes = len(set(route_by_episode.values()))
    unique_episode_scenes = len(set(scene_by_episode.values()))

    return {
        "dataset_dir": str(dataset_dir),
        "study_id": summary.get("study_id"),
        "exp_id": summary.get("exp_id"),
        "policy": summary.get("policy"),
        "split": summary.get("split"),
        "seed": summary.get("seed"),
        "checkpoint_path": summary.get("checkpoint_path"),
        "source_run_dir": summary.get("source_run_dir"),
        "shard_count": len(shard_entries),
        "total_transitions": total_transitions,
        "total_episodes": total_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "max_episode_length": int(max(episode_lengths)) if episode_lengths else 0,
        "mean_reward": float(total_rewards / total_transitions) if total_transitions else 0.0,
        "done_rate": float(total_done / total_transitions) if total_transitions else 0.0,
        "terminated_rate": float(total_terminated / total_transitions) if total_transitions else 0.0,
        "unique_routes_transition": unique_routes,
        "unique_scenes_transition": unique_scenes,
        "unique_routes_episode": unique_episode_routes,
        "unique_scenes_episode": unique_episode_scenes,
        "route_uniqueness_rate_episode": (
            float(unique_episode_routes / total_episodes) if total_episodes else 0.0
        ),
        "scene_uniqueness_rate_episode": (
            float(unique_episode_scenes / total_episodes) if total_episodes else 0.0
        ),
        "mean_transitions_per_route": (
            float(total_transitions / unique_routes) if unique_routes else 0.0
        ),
        "max_transitions_per_route": int(max(route_counts.values())) if route_counts else 0,
        "mean_transitions_per_scene": (
            float(total_transitions / unique_scenes) if unique_scenes else 0.0
        ),
        "max_transitions_per_scene": int(max(scene_counts.values())) if scene_counts else 0,
        "mean_straight_fraction": float(total_straight / total_transitions) if total_transitions else 0.0,
        "mean_left_turn_fraction": float(total_left / total_transitions) if total_transitions else 0.0,
        "mean_right_turn_fraction": float(total_right / total_transitions) if total_transitions else 0.0,
        "action_histogram": dict(sorted(action_counts.items())),
    }
