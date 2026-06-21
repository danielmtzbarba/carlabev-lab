from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.carlabev_lab.simulator.signatures import (
    extract_scene_route_metadata,
    route_signature,
    scene_signature,
)

from .common import (
    CARLABEV_REPO,
    CoverageData,
    DEFAULT_MAP_ASSET_SIZE,
    SCENE_PROFILES,
    SceneSample,
    town_map_asset_path,
    write_dataclass_csv,
    write_points_csv,
)

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options  # noqa: E402
from CarlaBEV.envs.carlabev import CarlaBEV  # noqa: E402


def make_env(frame_size: int) -> CarlaBEV:
    cfg = EnvConfig(
        seed=0,
        map_name="Town01",
        render_mode="rgb_array",
        obs_mode="bev_rgb",
        action_mode="discrete",
        size=frame_size,
        ego_anchor_x_frac=0.5,
        ego_anchor_y_frac=0.5,
        route_direction_metrics_enabled=True,
    )
    return CarlaBEV(cfg)


def extract_sample(
    *,
    scene_profile_id: str,
    seed: int,
    sample_index: int,
    applied_seed: int,
    env: CarlaBEV,
    info: dict[str, Any],
) -> SceneSample:
    hero = env.map.hero
    scenario = info.get("scenario", {})
    spawn = info.get("spawn_validation", {})
    metadata = extract_scene_route_metadata(env, info)
    return SceneSample(
        scene_profile_id=scene_profile_id,
        seed=seed,
        sample_index=sample_index,
        applied_seed=applied_seed,
        scene_signature=metadata["scene_signature"],
        route_signature=metadata["route_signature"],
        hero_x=round(float(hero.x), 6),
        hero_y=round(float(hero.y), 6),
        hero_yaw=round(float(hero.yaw), 6),
        hero_speed=round(float(hero.v), 6),
        route_length=round(float(env.len_ego_route), 6),
        num_vehicles=int(env.num_vehicles),
        route_profile=str(scenario.get("route_profile", "unknown")),
        route_turn_count=int(scenario.get("route_turn_count", 0)),
        route_intersection_like=bool(scenario.get("route_intersection_like", False)),
        straight_fraction=float(metadata["straight_fraction"]),
        left_turn_fraction=float(metadata["left_turn_fraction"]),
        right_turn_fraction=float(metadata["right_turn_fraction"]),
        spawn_valid=bool(spawn.get("valid", False)),
        spawn_reason=str(spawn.get("reason", "unknown")),
        reset_attempts=spawn.get("attempts"),
    )


def extract_coverage(env: CarlaBEV) -> CoverageData:
    route_x, route_y = env.map.route
    route_points = [(float(route_x[index]), float(route_y[index])) for index in range(min(len(route_x), len(route_y)))]
    hero = env.map.hero
    return CoverageData(
        spawn_points=[(float(hero.x), float(hero.y))],
        route_points=route_points,
    )


def representative_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted({int(round(v)) for v in np.linspace(0, total - 1, num=count)})


def save_frame(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(path)


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return None
    return mean(finite)


def build_pair_summary(samples: list[SceneSample]) -> dict[str, Any]:
    scene_signatures = [sample.scene_signature for sample in samples]
    route_signatures = [sample.route_signature for sample in samples]
    unique_signatures = len(set(scene_signatures))
    unique_route_signatures = len(set(route_signatures))
    unique_hero_poses = len({(round(sample.hero_x), round(sample.hero_y)) for sample in samples})
    counts = defaultdict(int)
    for signature in scene_signatures:
        counts[signature] += 1
    route_counts = defaultdict(int)
    for signature in route_signatures:
        route_counts[signature] += 1
    route_profile_counts = {
        profile: sum(1 for sample in samples if sample.route_profile == profile)
        for profile in sorted({sample.route_profile for sample in samples})
    }

    return {
        "num_samples": len(samples),
        "num_unique_scenes": unique_signatures,
        "num_unique_routes": unique_route_signatures,
        "num_unique_hero_poses": unique_hero_poses,
        "scene_repeat_ratio": 1.0 - (unique_signatures / len(samples) if samples else 0.0),
        "route_repeat_ratio": 1.0 - (unique_route_signatures / len(samples) if samples else 0.0),
        "most_common_scene_count": max(counts.values()) if counts else 0,
        "most_common_route_count": max(route_counts.values()) if route_counts else 0,
        "spawn_valid_rate": mean([1.0 if sample.spawn_valid else 0.0 for sample in samples]) if samples else None,
        "mean_route_length": finite_mean([sample.route_length for sample in samples]),
        "mean_num_vehicles": finite_mean([float(sample.num_vehicles) for sample in samples]),
        "mean_turn_count": finite_mean([float(sample.route_turn_count) for sample in samples]),
        "intersection_like_rate": mean([1.0 if sample.route_intersection_like else 0.0 for sample in samples]) if samples else None,
        "mean_straight_fraction": finite_mean([sample.straight_fraction for sample in samples]),
        "mean_left_turn_fraction": finite_mean([sample.left_turn_fraction for sample in samples]),
        "mean_right_turn_fraction": finite_mean([sample.right_turn_fraction for sample in samples]),
        "route_profile_counts": route_profile_counts,
        "spawn_reason_counts": {
            reason: sum(1 for sample in samples if sample.spawn_reason == reason)
            for reason in sorted({sample.spawn_reason for sample in samples})
        },
    }


def generate_dataset(
    *,
    seeds: list[int],
    scene_profile_ids: list[str],
    samples_per_seed: int,
    seed_mode: str,
    route_profile: str | None = None,
    route_profile_mix: dict[str, float] | None = None,
    min_turns: int | None = None,
    max_turns: int | None = None,
    intersection_required: bool | None = None,
    max_route_attempts: int | None = None,
    ego_route_graph: str = "full_vehicle",
    save_frames_per_pair: int,
    frame_size: int,
    output_dir: Path,
    map_asset_size: int = DEFAULT_MAP_ASSET_SIZE,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[SceneSample] = []
    summary: dict[str, Any] = {
        "seed_mode": seed_mode,
        "samples_per_seed": samples_per_seed,
        "seeds": seeds,
        "scene_profile_ids": scene_profile_ids,
        "route_profile": route_profile,
        "route_profile_mix": route_profile_mix,
        "min_turns": min_turns,
        "max_turns": max_turns,
        "intersection_required": intersection_required,
        "max_route_attempts": max_route_attempts,
        "ego_route_graph": ego_route_graph,
        "carlabev_repo": str(CARLABEV_REPO),
        "map_asset": str(town_map_asset_path("Town01", map_asset_size)),
        "pairs": {},
    }

    total_pairs = len(scene_profile_ids) * len(seeds)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        pair_task = progress.add_task("Seed/profile pairs", total=total_pairs)

        for scene_profile_id in scene_profile_ids:
            profile_spec = SCENE_PROFILES.get(scene_profile_id)
            if profile_spec is None:
                available = ", ".join(sorted(SCENE_PROFILES))
                raise ValueError(
                    f"Unknown scene_profile_id={scene_profile_id!r}. Available profiles: {available}"
                )
            for seed in seeds:
                env = make_env(frame_size)
                pair_samples: list[SceneSample] = []
                pair_spawn_points: list[tuple[float, float]] = []
                pair_route_points: list[tuple[float, float]] = []
                pair_key = f"{scene_profile_id}__seed_{seed}"
                pair_dir = output_dir / scene_profile_id / f"seed_{seed}"
                frame_indices = set(representative_indices(samples_per_seed, save_frames_per_pair))
                scene_task = progress.add_task(f"{scene_profile_id} seed={seed}", total=samples_per_seed)

                try:
                    for sample_index in range(samples_per_seed):
                        applied_seed = seed if seed_mode == "fixed" else seed + sample_index
                        reset_kwargs = dict(profile_spec.reset_kwargs)
                        reset_kwargs.update(
                            route_profile=route_profile,
                            route_profile_mix=route_profile_mix,
                            min_turns=min_turns,
                            max_turns=max_turns,
                            intersection_required=intersection_required,
                            max_route_attempts=max_route_attempts,
                            ego_route_graph=ego_route_graph,
                        )
                        options = build_random_navigation_options(
                            RandomNavigationReset(**reset_kwargs)
                        )
                        frame, info = env.reset(seed=applied_seed, options=options)
                        sample = extract_sample(
                            scene_profile_id=scene_profile_id,
                            seed=seed,
                            sample_index=sample_index,
                            applied_seed=applied_seed,
                            env=env,
                            info=info,
                        )
                        pair_samples.append(sample)
                        coverage = extract_coverage(env)
                        pair_spawn_points.extend(coverage.spawn_points)
                        pair_route_points.extend(coverage.route_points)

                        if sample_index in frame_indices:
                            save_frame(frame, pair_dir / "frames" / f"sample_{sample_index:04d}_seed_{applied_seed}.png")
                        progress.advance(scene_task)
                finally:
                    env.close()
                    progress.remove_task(scene_task)

                all_samples.extend(pair_samples)
                write_dataclass_csv(pair_samples, pair_dir / "samples.csv")
                write_points_csv(pair_spawn_points, pair_dir / "spawn_points.csv")
                write_points_csv(pair_route_points, pair_dir / "route_points.csv")
                summary["pairs"][pair_key] = {
                    "scene_profile_id": scene_profile_id,
                    "scene_profile_label": profile_spec.label,
                    "seed": seed,
                    "seed_mode": seed_mode,
                    **build_pair_summary(pair_samples),
                }
                progress.advance(pair_task)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if all_samples:
        write_dataclass_csv(all_samples, output_dir / "all_samples.csv")

    return summary
