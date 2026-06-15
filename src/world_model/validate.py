from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from src.world_model.contracts import DatasetRootConfig, DatasetValidationReportModel, WorldModelSequenceConfig
from src.world_model.data import build_index, load_or_build_sequence_window_cache


def validate_datasets(
    dataset_roots: list[str | Path | DatasetRootConfig],
    *,
    cfg: WorldModelSequenceConfig | None = None,
    chunk_lengths: tuple[int, ...] = (1, 8, 16),
    cache_sequence_indices: bool = True,
    sequence_cache_dir: str | None = None,
) -> DatasetValidationReportModel:
    cfg = cfg or WorldModelSequenceConfig()
    indexed = build_index(dataset_roots)

    transition_count = len(indexed.transitions)
    episode_lengths = [len(refs) for refs in indexed.episodes.values()]
    action_histogram = Counter(str(ref.action_value) for ref in indexed.transitions)
    routes_transition = [ref.route_signature for ref in indexed.transitions if ref.route_signature]
    scenes_transition = [ref.scene_signature for ref in indexed.transitions if ref.scene_signature]
    routes_episode = [
        refs[0].route_signature for refs in indexed.episodes.values() if refs and refs[0].route_signature
    ]
    scenes_episode = [
        refs[0].scene_signature for refs in indexed.episodes.values() if refs and refs[0].scene_signature
    ]
    dones = [int(ref.done) for ref in indexed.transitions]
    terminated = [int(ref.terminated) for ref in indexed.transitions]
    truncated = [int(ref.truncated) for ref in indexed.transitions]
    per_source_counts = Counter(ref.source_name for ref in indexed.transitions)
    valid_windows = {
        int(chunk_length): int(
            load_or_build_sequence_window_cache(
                indexed,
                chunk_length=int(chunk_length),
                stride=cfg.stride,
                enabled=cache_sequence_indices,
                cache_dir=sequence_cache_dir,
            ).window_transition_indices.shape[0]
        )
        for chunk_length in chunk_lengths
    }

    warnings: list[str] = []
    if transition_count == 0:
        warnings.append("No transitions found.")
    if action_histogram:
        max_action_share = max(action_histogram.values()) / transition_count
        if max_action_share >= 0.95:
            warnings.append("Action distribution appears heavily collapsed.")
    if routes_episode:
        route_uniqueness_rate_episode = len(set(routes_episode)) / len(routes_episode)
        if route_uniqueness_rate_episode <= 0.10:
            warnings.append("Route diversity appears low at the episode level.")
    else:
        route_uniqueness_rate_episode = 0.0
        warnings.append("No route signatures were found.")
    if scenes_episode:
        scene_uniqueness_rate_episode = len(set(scenes_episode)) / len(scenes_episode)
    else:
        scene_uniqueness_rate_episode = 0.0
        warnings.append("No scene signatures were found.")

    route_uniqueness_rate_transition = (
        len(set(routes_transition)) / len(routes_transition) if routes_transition else 0.0
    )
    scene_uniqueness_rate_transition = (
        len(set(scenes_transition)) / len(scenes_transition) if scenes_transition else 0.0
    )

    if cfg.expected_num_actions is not None and action_histogram:
        seen_actions = {int(key) for key in action_histogram}
        expected_actions = set(range(cfg.expected_num_actions))
        if not seen_actions.issubset(expected_actions):
            warnings.append(
                "Observed action ids exceed the expected discrete action range."
            )

    lengths = np.asarray(episode_lengths, dtype=np.int32) if episode_lengths else np.asarray([], dtype=np.int32)
    p50_length = float(np.percentile(lengths, 50)) if len(lengths) else 0.0
    p95_length = float(np.percentile(lengths, 95)) if len(lengths) else 0.0

    return DatasetValidationReportModel(
        dataset_roots=[str(source.root_path) for source in indexed.sources],
        source_names=[source.source_name for source in indexed.sources],
        total_datasets=len(indexed.sources),
        total_shards=len(indexed.shards),
        total_transitions=transition_count,
        total_episodes=len(indexed.episodes),
        valid_windows=valid_windows,
        action_histogram=dict(sorted(action_histogram.items(), key=lambda item: int(item[0]))),
        mean_episode_length=float(mean(episode_lengths)) if episode_lengths else 0.0,
        p50_episode_length=p50_length,
        p95_episode_length=p95_length,
        max_episode_length=max(episode_lengths) if episode_lengths else 0,
        unique_routes_episode=len(set(routes_episode)),
        unique_scenes_episode=len(set(scenes_episode)),
        unique_routes_transition=len(set(routes_transition)),
        unique_scenes_transition=len(set(scenes_transition)),
        mean_done_rate=float(mean(dones)) if dones else 0.0,
        mean_terminated_rate=float(mean(terminated)) if terminated else 0.0,
        mean_truncated_rate=float(mean(truncated)) if truncated else 0.0,
        route_uniqueness_rate_episode=route_uniqueness_rate_episode,
        scene_uniqueness_rate_episode=scene_uniqueness_rate_episode,
        route_uniqueness_rate_transition=route_uniqueness_rate_transition,
        scene_uniqueness_rate_transition=scene_uniqueness_rate_transition,
        dataset_transition_counts=dict(per_source_counts),
        warnings=warnings,
    )
