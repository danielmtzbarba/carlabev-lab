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

from .common import (
    CARLABEV_REPO,
    CoverageData,
    DEFAULT_MAP_ASSET_SIZE,
    DIFFICULTY_LABELS,
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


def scene_signature(env: CarlaBEV) -> str:
    hero = env.map.hero
    route_x, route_y = env.map.route
    vehicles = []
    for actor in env.map.actor_manager.actors.get("vehicle", []):
        ax, ay, ayaw, av = actor.state
        vehicles.append(
            (
                round(float(ax), 3),
                round(float(ay), 3),
                round(float(ayaw), 5),
                round(float(av), 5),
            )
        )

    payload = {
        "hero": (
            round(float(hero.x), 3),
            round(float(hero.y), 3),
            round(float(hero.yaw), 5),
            round(float(hero.v), 5),
        ),
        "route_head": tuple((int(route_x[i]), int(route_y[i])) for i in range(min(24, len(route_x)))),
        "route_len": int(len(route_x)),
        "vehicles": tuple(vehicles),
        "num_vehicles": int(env.num_vehicles),
        "route_length": round(float(env.len_ego_route), 4),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def extract_sample(
    *,
    difficulty_id: str,
    seed: int,
    sample_index: int,
    applied_seed: int,
    env: CarlaBEV,
    info: dict[str, Any],
) -> SceneSample:
    hero = env.map.hero
    scenario = info.get("scenario", {})
    spawn = info.get("spawn_validation", {})
    return SceneSample(
        difficulty_id=difficulty_id,
        seed=seed,
        sample_index=sample_index,
        applied_seed=applied_seed,
        scene_signature=scene_signature(env),
        hero_x=round(float(hero.x), 6),
        hero_y=round(float(hero.y), 6),
        hero_yaw=round(float(hero.yaw), 6),
        hero_speed=round(float(hero.v), 6),
        route_length=round(float(env.len_ego_route), 6),
        num_vehicles=int(env.num_vehicles),
        straight_fraction=float(scenario.get("straight_fraction", math.nan)),
        left_turn_fraction=float(scenario.get("left_turn_fraction", math.nan)),
        right_turn_fraction=float(scenario.get("right_turn_fraction", math.nan)),
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
    signatures = [sample.scene_signature for sample in samples]
    unique_signatures = len(set(signatures))
    unique_hero_poses = len({(round(sample.hero_x), round(sample.hero_y)) for sample in samples})
    counts = defaultdict(int)
    for signature in signatures:
        counts[signature] += 1

    return {
        "num_samples": len(samples),
        "num_unique_scenes": unique_signatures,
        "num_unique_hero_poses": unique_hero_poses,
        "scene_repeat_ratio": 1.0 - (unique_signatures / len(samples) if samples else 0.0),
        "most_common_scene_count": max(counts.values()) if counts else 0,
        "spawn_valid_rate": mean([1.0 if sample.spawn_valid else 0.0 for sample in samples]) if samples else None,
        "mean_route_length": finite_mean([sample.route_length for sample in samples]),
        "mean_num_vehicles": finite_mean([float(sample.num_vehicles) for sample in samples]),
        "mean_straight_fraction": finite_mean([sample.straight_fraction for sample in samples]),
        "mean_left_turn_fraction": finite_mean([sample.left_turn_fraction for sample in samples]),
        "mean_right_turn_fraction": finite_mean([sample.right_turn_fraction for sample in samples]),
        "spawn_reason_counts": {
            reason: sum(1 for sample in samples if sample.spawn_reason == reason)
            for reason in sorted({sample.spawn_reason for sample in samples})
        },
    }


def generate_dataset(
    *,
    seeds: list[int],
    difficulty_ids: list[str],
    samples_per_seed: int,
    seed_mode: str,
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
        "difficulty_ids": difficulty_ids,
        "carlabev_repo": str(CARLABEV_REPO),
        "map_asset": str(town_map_asset_path("Town01", map_asset_size)),
        "pairs": {},
    }

    total_pairs = len(difficulty_ids) * len(seeds)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        pair_task = progress.add_task("Seed/difficulty pairs", total=total_pairs)

        for difficulty_id in difficulty_ids:
            for seed in seeds:
                env = make_env(frame_size)
                pair_samples: list[SceneSample] = []
                pair_spawn_points: list[tuple[float, float]] = []
                pair_route_points: list[tuple[float, float]] = []
                pair_key = f"{difficulty_id}__seed_{seed}"
                pair_dir = output_dir / difficulty_id / f"seed_{seed}"
                frame_indices = set(representative_indices(samples_per_seed, save_frames_per_pair))
                scene_task = progress.add_task(f"{difficulty_id} seed={seed}", total=samples_per_seed)

                try:
                    for sample_index in range(samples_per_seed):
                        applied_seed = seed if seed_mode == "fixed" else seed + sample_index
                        options = build_random_navigation_options(RandomNavigationReset(difficulty_id=difficulty_id))
                        frame, info = env.reset(seed=applied_seed, options=options)
                        sample = extract_sample(
                            difficulty_id=difficulty_id,
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
                    "difficulty_id": difficulty_id,
                    "difficulty_label": DIFFICULTY_LABELS.get(difficulty_id, difficulty_id),
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
