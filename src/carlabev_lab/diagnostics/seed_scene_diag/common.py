from __future__ import annotations

import csv
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import ImageFont


def add_carlabev_repo_to_path() -> Path:
    candidate_paths = []

    env_repo = os.environ.get("CARLABEV_REPO")
    if env_repo:
        candidate_paths.append(Path(env_repo).expanduser())

    lab_repo = Path(__file__).resolve().parents[2]
    candidate_paths.extend(
        [
            lab_repo.with_name("carlabev-env"),
            Path("/home/danielmtz/Projects/carlabev-env"),
            Path("/Users/danielmtz/Data/projects/driverless/carlabev-env"),
        ]
    )

    for repo_root in candidate_paths:
        if (repo_root / "CarlaBEV").exists():
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            return repo_root

    checked = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Could not locate carlabev-env repo. Checked: {checked}")


CARLABEV_REPO = add_carlabev_repo_to_path()

@dataclass(frozen=True)
class SceneProfileSpec:
    profile_id: str
    label: str
    reset_kwargs: dict[str, Any]


SCENE_PROFILES: dict[str, SceneProfileSpec] = {
    "no_traffic": SceneProfileSpec(
        profile_id="no_traffic",
        label="No traffic",
        reset_kwargs={
            "route_extent": "medium",
            "route_dist_range": (50, 130),
            "speed_profile": "medium",
            "num_vehicles": 0,
            "num_vehicles_near_ego": 0,
        },
    ),
    "easy": SceneProfileSpec(
        profile_id="easy",
        label="Easy traffic",
        reset_kwargs={
            "route_extent": "medium",
            "route_dist_range": (50, 130),
            "speed_profile": "medium",
            "num_vehicles": 2,
            "num_vehicles_near_ego": 2,
            "traffic_role_profile": "lead",
            "guaranteed_candidate_role": "lead",
        },
    ),
    "medium": SceneProfileSpec(
        profile_id="medium",
        label="Medium traffic",
        reset_kwargs={
            "route_extent": "medium",
            "route_dist_range": (50, 130),
            "speed_profile": "medium",
            "num_vehicles": 4,
            "num_vehicles_near_ego": 4,
            "traffic_role_profile": "mix",
            "guaranteed_candidate_role": "mix",
        },
    ),
    "hard": SceneProfileSpec(
        profile_id="hard",
        label="Hard traffic",
        reset_kwargs={
            "route_extent": "medium",
            "route_dist_range": (50, 130),
            "speed_profile": "medium",
            "num_vehicles": 6,
            "num_vehicles_near_ego": 6,
            "traffic_role_profile": "mix",
            "guaranteed_candidate_role": "mix",
        },
    ),
}

DEFAULT_MAP_ASSET_SIZE = 128
ROUTE_HEATMAP_SIGMA = 4
HEATMAP_ALPHA = 235
COMPARISON_MARGIN = 24
COMPARISON_HEADER = 118
LEGEND_WIDTH = 240
LEGEND_HEIGHT_SPAWN = 230
LEGEND_HEIGHT_ROUTE = 68
PANEL_BACKGROUND = (244, 240, 233, 255)
TITLE_COLOR = (26, 31, 36, 255)
SUBTITLE_COLOR = (78, 86, 94, 255)
LABEL_COLOR = (48, 55, 64, 255)
LEGEND_TEXT_COLOR = (42, 48, 56, 255)
SPAWN_CLUSTER_COLORS = [
    (215, 48, 39, 235),
    (252, 141, 89, 235),
    (254, 224, 144, 235),
    (224, 243, 248, 235),
    (171, 217, 233, 235),
    (116, 173, 209, 235),
    (69, 117, 180, 235),
    (49, 54, 149, 235),
    (123, 50, 148, 235),
    (178, 24, 43, 235),
]


@dataclass
class SceneSample:
    scene_profile_id: str
    seed: int
    sample_index: int
    applied_seed: int
    scene_signature: str
    route_signature: str
    hero_x: float
    hero_y: float
    hero_yaw: float
    hero_speed: float
    route_length: float
    num_vehicles: int
    route_profile: str
    route_turn_count: int
    route_intersection_like: bool
    straight_fraction: float
    left_turn_fraction: float
    right_turn_fraction: float
    spawn_valid: bool
    spawn_reason: str
    reset_attempts: int | None


@dataclass
class CoverageData:
    spawn_points: list[tuple[float, float]]
    route_points: list[tuple[float, float]]


@dataclass
class SpawnCluster:
    rank: int
    center_x: float
    center_y: float
    count: int
    share: float


def town_map_asset_path(map_name: str, size: int) -> Path:
    return CARLABEV_REPO / "CarlaBEV" / "assets" / map_name / f"{map_name}-{size}-rgb.png"


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def write_dataclass_csv(rows: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [field.name for field in rows[0].__dataclass_fields__.values()]  # type: ignore[attr-defined]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_points_csv(points: list[tuple[float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        writer.writerows(points)


def load_points_csv(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not path.exists():
        return points
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    return points


def load_scene_samples_csv(path: Path) -> list[SceneSample]:
    samples: list[SceneSample] = []
    if not path.exists():
        return samples
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                SceneSample(
                    scene_profile_id=row.get("scene_profile_id", row.get("difficulty_id", "unknown")),
                    seed=int(row["seed"]),
                    sample_index=int(row["sample_index"]),
                    applied_seed=int(row["applied_seed"]),
                    scene_signature=row["scene_signature"],
                    route_signature=row.get("route_signature", row["scene_signature"]),
                    hero_x=float(row["hero_x"]),
                    hero_y=float(row["hero_y"]),
                    hero_yaw=float(row["hero_yaw"]),
                    hero_speed=float(row["hero_speed"]),
                    route_length=float(row["route_length"]),
                    num_vehicles=int(row["num_vehicles"]),
                    route_profile=row.get("route_profile", "unknown"),
                    route_turn_count=int(row.get("route_turn_count", 0)),
                    route_intersection_like=row.get("route_intersection_like", "False").lower() == "true",
                    straight_fraction=float(row["straight_fraction"]),
                    left_turn_fraction=float(row["left_turn_fraction"]),
                    right_turn_fraction=float(row["right_turn_fraction"]),
                    spawn_valid=row["spawn_valid"].lower() == "true",
                    spawn_reason=row["spawn_reason"],
                    reset_attempts=int(row["reset_attempts"]) if row["reset_attempts"] not in {"", "None"} else None,
                )
            )
    return samples
