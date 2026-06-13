from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.carlabev_lab.simulator.signatures import (
    extract_scene_route_metadata,
    route_signature,
    scene_signature,
)


def _fake_env():
    hero = SimpleNamespace(x=10.0, y=20.0, yaw=0.25, v=3.0)
    vehicle = SimpleNamespace(state=(1.0, 2.0, 0.5, 4.0))
    actor_manager = SimpleNamespace(actors={"vehicle": [vehicle]})
    map_obj = SimpleNamespace(hero=hero, route=([0, 1, 2], [3, 4, 5]), actor_manager=actor_manager)
    return SimpleNamespace(
        map=map_obj,
        num_vehicles=1,
        len_ego_route=12.5,
        _scenario_context={
            "straight_fraction": 0.6,
            "left_turn_fraction": 0.3,
            "right_turn_fraction": 0.1,
        },
    )


@pytest.mark.unit
def test_route_and_scene_signatures_are_stable():
    env = _fake_env()
    assert route_signature(env) == route_signature(env)
    assert scene_signature(env) == scene_signature(env)


@pytest.mark.unit
def test_extract_scene_route_metadata_uses_scenario_context():
    env = _fake_env()
    payload = extract_scene_route_metadata(env)
    assert payload["route_signature"]
    assert payload["scene_signature"]
    assert payload["straight_fraction"] == pytest.approx(0.6)
    assert payload["left_turn_fraction"] == pytest.approx(0.3)
    assert payload["right_turn_fraction"] == pytest.approx(0.1)
