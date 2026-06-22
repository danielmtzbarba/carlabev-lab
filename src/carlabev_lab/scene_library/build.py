from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from CarlaBEV.tools.build_scene_library import build_scene_library as build_env_scene_library

from src.config.scene_benchmarks.registry import (
    get_scene_benchmark_config,
    get_scene_benchmark_profile,
)
from src.config.studies.models import RandomNavigationProtocol, StudyConfig
from src.config.studies.registry import get_study_config
from src.tuning.optuna_utils import DEFAULT_STUDY_PRIME_SEEDS
from src.utils.storage_paths import resolve_artifact_path


@dataclass(frozen=True)
class BackboneBuildPlan:
    backbone_id: str
    seed_owner_id: str
    scene_library_path: str
    protocol_ids: tuple[str, ...]
    request_kwargs: dict[str, Any]


def _request_kwargs_from_backbone(
    backbone,
    *,
    scene_benchmark_id: str | None = None,
    scene_split: str | None = None,
) -> dict[str, Any]:
    return {
        "scene_benchmark_id": scene_benchmark_id,
        "scene_profile_id": backbone.scene_profile_id,
        "scene_split": scene_split,
        "difficulty_preset_id": "medium",
        "route_extent": backbone.route_extent,
        "route_dist_range": backbone.route_dist_range,
        "speed_profile": backbone.speed_profile,
        "num_vehicles": backbone.num_vehicles,
        "num_vehicles_near_ego": backbone.num_vehicles_near_ego,
        "traffic_role_profile": backbone.traffic_role_profile,
        "guaranteed_candidate_role": backbone.guaranteed_candidate_role,
        "ego_route_graph": backbone.ego_route_graph,
    }


def _normalize_request_kwargs(
    protocol: RandomNavigationProtocol,
    *,
    study_id: str,
) -> tuple[str, str, dict[str, Any]]:
    source = protocol.scene_source
    if source is not None and source.mode == "benchmark":
        profile = get_scene_benchmark_profile(source.benchmark_id, source.scene_profile_id)
        request_kwargs = _request_kwargs_from_backbone(
            profile.backbone,
            scene_benchmark_id=source.benchmark_id,
            scene_split=source.split,
        )
        stable_backbone_id = f"{source.scene_profile_id}:{source.split}"
        return source.benchmark_id, stable_backbone_id, request_kwargs
    assert protocol.backbone is not None
    request_kwargs = _request_kwargs_from_backbone(protocol.backbone)
    digest = hashlib.sha256(json.dumps(request_kwargs, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    stable_backbone_id = request_kwargs["scene_profile_id"] or f"study_private_{digest}"
    return study_id, stable_backbone_id, request_kwargs


def _backbone_signature(
    seed_owner_id: str,
    backbone_id: str,
    scene_library_path: str,
    request_kwargs: dict[str, Any],
) -> str:
    payload = {
        "seed_owner_id": seed_owner_id,
        "backbone_id": backbone_id,
        "scene_library_path": scene_library_path,
        "request_kwargs": request_kwargs,
    }
    return json.dumps(payload, sort_keys=True)


def _iter_selected_protocols(
    study: StudyConfig,
    *,
    include_train: bool,
    include_eval: bool,
    protocol_ids: set[str] | None,
):
    if include_train:
        for protocol_id, protocol in study.train_protocols.items():
            if protocol_ids is None or protocol_id in protocol_ids:
                yield protocol_id, protocol
    if include_eval:
        for protocol_id, protocol in study.eval_protocols.items():
            if protocol_ids is None or protocol_id in protocol_ids:
                yield protocol_id, protocol


def resolve_study_scene_library_plan(
    study_id: str,
    *,
    include_train: bool = True,
    include_eval: bool = True,
    protocol_ids: list[str] | None = None,
) -> list[BackboneBuildPlan]:
    study = get_study_config(study_id)
    selected_ids = set(protocol_ids) if protocol_ids else None
    grouped: dict[str, dict[str, Any]] = {}

    for protocol_id, protocol in _iter_selected_protocols(
        study,
        include_train=include_train,
        include_eval=include_eval,
        protocol_ids=selected_ids,
    ):
        if protocol.mode != "random_navigation":
            continue
        if protocol.scene_source is not None and protocol.scene_source.mode == "benchmark":
            benchmark = get_scene_benchmark_config(protocol.scene_source.benchmark_id)
            scene_library = benchmark.scene_library
        else:
            scene_library = protocol.scene_library
        if scene_library is None or scene_library.path is None:
            raise ValueError(
                f"Random-navigation protocol {protocol_id!r} in study {study_id!r} does not declare a scene-library path."
            )
        seed_owner_id, backbone_id, request_kwargs = _normalize_request_kwargs(
            protocol,
            study_id=study_id,
        )
        signature = _backbone_signature(
            seed_owner_id,
            backbone_id,
            scene_library.path,
            request_kwargs,
        )
        if signature not in grouped:
            grouped[signature] = {
                "seed_owner_id": seed_owner_id,
                "backbone_id": backbone_id,
                "scene_library_path": scene_library.path,
                "request_kwargs": request_kwargs,
                "protocol_ids": [],
            }
        grouped[signature]["protocol_ids"].append(protocol_id)

    plans: list[BackboneBuildPlan] = []
    for item in grouped.values():
        protocol_names = tuple(sorted(item["protocol_ids"]))
        plans.append(
            BackboneBuildPlan(
                backbone_id=item["backbone_id"],
                seed_owner_id=item["seed_owner_id"],
                scene_library_path=item["scene_library_path"],
                protocol_ids=protocol_names,
                request_kwargs=item["request_kwargs"],
            )
        )
    return plans


def resolve_benchmark_scene_library_plan(
    benchmark_id: str,
    *,
    profiles: list[str] | None = None,
    include_train: bool = True,
    include_eval: bool = True,
) -> tuple[list[BackboneBuildPlan], list[int], int]:
    benchmark = get_scene_benchmark_config(benchmark_id)
    selected_profiles = set(profiles) if profiles else None
    plans: list[BackboneBuildPlan] = []
    splits = []
    if include_train:
        splits.append("train")
    if include_eval:
        splits.append("eval")
    for scene_profile_id, profile in benchmark.profiles.items():
        if selected_profiles is not None and scene_profile_id not in selected_profiles:
            continue
        for split in splits:
            request_kwargs = _request_kwargs_from_backbone(
                profile.backbone,
                scene_benchmark_id=benchmark.benchmark_id,
                scene_split=split,
            )
            plans.append(
                BackboneBuildPlan(
                    backbone_id=f"{scene_profile_id}:{split}",
                    seed_owner_id=benchmark_id,
                    scene_library_path=str(benchmark.scene_library.path),
                    protocol_ids=(f"{scene_profile_id}_{split}",),
                    request_kwargs=request_kwargs,
                )
            )
    return plans, list(benchmark.prime_seeds), int(benchmark.episodes_per_seed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CarlaBEV scene-library databases from study-private or benchmark-backed random-navigation protocols."
    )
    parser.add_argument("--study-id", default=None)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=None)
    parser.add_argument(
        "--prime-seeds",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--protocol-ids", nargs="+", default=None)
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument(
        "--scene-library-path",
        default=None,
        help="Optional override output path for every selected build plan.",
    )
    parser.add_argument("--include-train", action="store_true", default=False)
    parser.add_argument("--include-eval", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def override_scene_library_paths(
    plans: list[BackboneBuildPlan],
    scene_library_path: str | None,
) -> list[BackboneBuildPlan]:
    if scene_library_path is None:
        return plans
    return [
        replace(plan, scene_library_path=scene_library_path)
        for plan in plans
    ]


def _render_plan(
    *,
    target_id: str,
    target_kind: str,
    plans: list[BackboneBuildPlan],
    prime_seeds: list[int],
    episodes_per_seed: int,
) -> dict[str, Any]:
    return {
        f"{target_kind}_id": target_id,
        "target_kind": target_kind,
        "scene_library_paths": sorted({plan.scene_library_path for plan in plans}),
        "backbone_count": len(plans),
        "prime_seeds": list(prime_seeds),
        "episodes_per_seed": int(episodes_per_seed),
        "total_build_calls": len(plans) * len(prime_seeds),
        "total_requested_scenes": len(plans) * len(prime_seeds) * int(episodes_per_seed),
        "backbones": [asdict(plan) for plan in plans],
    }


def _execute_plan(
    *,
    target_id: str,
    target_kind: str,
    plans: list[BackboneBuildPlan],
    prime_seeds: list[int],
    episodes_per_seed: int,
) -> dict[str, Any]:
    executions: list[dict[str, Any]] = []
    for plan in plans:
        db_path = resolve_artifact_path(plan.scene_library_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        for study_seed in prime_seeds:
            summary = build_env_scene_library(
                episodes=episodes_per_seed,
                study_seed=study_seed,
                study_id=plan.seed_owner_id,
                backbone_id=plan.backbone_id,
                scene_library_path=str(db_path),
                **plan.request_kwargs,
            )
            executions.append(
                {
                    "backbone_id": plan.backbone_id,
                    "protocol_ids": list(plan.protocol_ids),
                    "study_seed": study_seed,
                    "summary": summary,
                }
            )
            logger.info(
                "{} {} | {} | seed {} | db {} | misses {} | hits {} | unique {}",
                target_kind,
                target_id,
                plan.backbone_id,
                study_seed,
                db_path,
                summary["misses"],
                summary["hits"],
                summary["unique_scene_keys"],
            )
    return {
        f"{target_kind}_id": target_id,
        "target_kind": target_kind,
        "executions": executions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.study_id) == bool(args.benchmark_id):
        raise SystemExit("Pass exactly one of --study-id or --benchmark-id.")
    include_train = args.include_train or not args.include_eval
    include_eval = args.include_eval or not args.include_train

    if args.benchmark_id is not None:
        plans, benchmark_prime_seeds, benchmark_episodes = resolve_benchmark_scene_library_plan(
            args.benchmark_id,
            profiles=args.profiles,
            include_train=include_train,
            include_eval=include_eval,
        )
        target_id = args.benchmark_id
        target_kind = "benchmark"
        prime_seeds = (
            list(DEFAULT_STUDY_PRIME_SEEDS)
            if args.prime_seeds is None
            else list(args.prime_seeds)
        )
        if args.prime_seeds is None:
            prime_seeds = benchmark_prime_seeds
        episodes_per_seed = (
            int(benchmark_episodes)
            if args.episodes_per_seed is None
            else int(args.episodes_per_seed)
        )
    else:
        plans = resolve_study_scene_library_plan(
            args.study_id,
            include_train=include_train,
            include_eval=include_eval,
            protocol_ids=args.protocol_ids,
        )
        target_id = args.study_id
        target_kind = "study"
        prime_seeds = (
            list(DEFAULT_STUDY_PRIME_SEEDS)
            if args.prime_seeds is None
            else list(args.prime_seeds)
        )
        episodes_per_seed = 1000 if args.episodes_per_seed is None else int(args.episodes_per_seed)
    plans = override_scene_library_paths(plans, args.scene_library_path)
    payload = _render_plan(
        target_id=target_id,
        target_kind=target_kind,
        plans=plans,
        prime_seeds=prime_seeds,
        episodes_per_seed=episodes_per_seed,
    )

    if args.dry_run:
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            logger.info(
                "{} {} | backbones {} | prime_seeds {} | episodes_per_seed {} | build_calls {} | requested_scenes {}",
                target_kind,
                target_id,
                payload["backbone_count"],
                len(payload["prime_seeds"]),
                payload["episodes_per_seed"],
                payload["total_build_calls"],
                payload["total_requested_scenes"],
            )
            for plan in plans:
                logger.info(
                    "{} | protocols {} | db {}",
                    plan.backbone_id,
                    ",".join(plan.protocol_ids),
                    resolve_artifact_path(plan.scene_library_path),
                )
        return 0

    result = _execute_plan(
        target_id=target_id,
        target_kind=target_kind,
        plans=plans,
        prime_seeds=prime_seeds,
        episodes_per_seed=episodes_per_seed,
    )
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
