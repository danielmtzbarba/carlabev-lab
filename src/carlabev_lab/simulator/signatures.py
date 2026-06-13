from __future__ import annotations

import json
import math
from typing import Any
import hashlib


def scene_signature(env) -> str:
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
        "route_head": tuple(
            (int(route_x[i]), int(route_y[i])) for i in range(min(24, len(route_x)))
        ),
        "route_len": int(len(route_x)),
        "vehicles": tuple(vehicles),
        "num_vehicles": int(env.num_vehicles),
        "route_length": round(float(env.len_ego_route), 4),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def route_signature(env) -> str:
    route_x, route_y = env.map.route
    payload = {
        "route": tuple(
            (round(float(route_x[i]), 3), round(float(route_y[i]), 3))
            for i in range(min(len(route_x), len(route_y)))
        ),
        "route_length": round(float(env.len_ego_route), 4),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _scenario_context(env, info: dict[str, Any] | None = None) -> dict[str, Any]:
    if hasattr(env, "_scenario_context") and getattr(env, "_scenario_context"):
        return dict(getattr(env, "_scenario_context"))
    if info is not None and isinstance(info.get("scenario"), dict):
        return dict(info["scenario"])
    return {}


def extract_scene_route_metadata(env, info: dict[str, Any] | None = None) -> dict[str, Any]:
    context = _scenario_context(env, info)
    return {
        "scene_signature": scene_signature(env),
        "route_signature": route_signature(env),
        "straight_fraction": float(context.get("straight_fraction", math.nan)),
        "left_turn_fraction": float(context.get("left_turn_fraction", math.nan)),
        "right_turn_fraction": float(context.get("right_turn_fraction", math.nan)),
    }
