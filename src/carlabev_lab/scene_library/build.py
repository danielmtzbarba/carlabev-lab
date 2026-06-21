from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from CarlaBEV.tools.build_scene_library import build_scene_library as build_env_scene_library

from src.config.studies.models import RandomNavigationProtocol, StudyConfig
from src.config.studies.registry import get_study_config
from src.tuning.optuna_utils import DEFAULT_STUDY_PRIME_SEEDS
from src.utils.storage_paths import resolve_artifact_path


@dataclass(frozen=True)
class BackboneBuildPlan:
    backbone_id: str
    scene_library_path: str
    protocol_ids: tuple[str, ...]
    request_kwargs: dict[str, Any]


def _normalize_request_kwargs(protocol: RandomNavigationProtocol) -> dict[str, Any]:
    backbone = protocol.backbone
    return {
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


def _backbone_signature(scene_library_path: str, request_kwargs: dict[str, Any]) -> str:
    payload = {"scene_library_path": scene_library_path, "request_kwargs": request_kwargs}
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
        scene_library = protocol.scene_library
        if scene_library is None or scene_library.path is None:
            raise ValueError(
                f"Random-navigation protocol {protocol_id!r} in study {study_id!r} does not declare a scene-library path."
            )
        request_kwargs = _normalize_request_kwargs(protocol)
        signature = _backbone_signature(scene_library.path, request_kwargs)
        if signature not in grouped:
            grouped[signature] = {
                "scene_library_path": scene_library.path,
                "request_kwargs": request_kwargs,
                "protocol_ids": [],
            }
        grouped[signature]["protocol_ids"].append(protocol_id)

    plans: list[BackboneBuildPlan] = []
    for index, item in enumerate(grouped.values()):
        protocol_names = tuple(sorted(item["protocol_ids"]))
        plans.append(
            BackboneBuildPlan(
                backbone_id=f"backbone_{index}",
                scene_library_path=item["scene_library_path"],
                protocol_ids=protocol_names,
                request_kwargs=item["request_kwargs"],
            )
        )
    return plans


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build study-owned CarlaBEV scene-library databases from random-navigation backbones."
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=1000)
    parser.add_argument(
        "--prime-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_STUDY_PRIME_SEEDS),
    )
    parser.add_argument("--protocol-ids", nargs="+", default=None)
    parser.add_argument("--include-train", action="store_true", default=False)
    parser.add_argument("--include-eval", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _render_plan(
    *,
    study_id: str,
    plans: list[BackboneBuildPlan],
    prime_seeds: list[int],
    episodes_per_seed: int,
) -> dict[str, Any]:
    return {
        "study_id": study_id,
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
    study_id: str,
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
                study_id=study_id,
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
                "Study {} | {} | seed {} | db {} | misses {} | hits {} | unique {}",
                study_id,
                plan.backbone_id,
                study_seed,
                db_path,
                summary["misses"],
                summary["hits"],
                summary["unique_scene_keys"],
            )
    return {"study_id": study_id, "executions": executions}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    include_train = args.include_train or not args.include_eval
    include_eval = args.include_eval or not args.include_train

    plans = resolve_study_scene_library_plan(
        args.study_id,
        include_train=include_train,
        include_eval=include_eval,
        protocol_ids=args.protocol_ids,
    )
    payload = _render_plan(
        study_id=args.study_id,
        plans=plans,
        prime_seeds=list(args.prime_seeds),
        episodes_per_seed=args.episodes_per_seed,
    )

    if args.dry_run:
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            logger.info(
                "Study {} | backbones {} | prime_seeds {} | episodes_per_seed {} | build_calls {} | requested_scenes {}",
                payload["study_id"],
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
        study_id=args.study_id,
        plans=plans,
        prime_seeds=list(args.prime_seeds),
        episodes_per_seed=args.episodes_per_seed,
    )
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
