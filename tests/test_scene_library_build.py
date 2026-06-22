import pytest
from CarlaBEV.src.managers.scene_library import SceneLibrary

from src.carlabev_lab.scene_library.build import (
    override_scene_library_paths,
    resolve_benchmark_scene_library_plan,
    resolve_study_scene_library_plan,
)
from src.carlabev_lab.scene_library.merge import merge_scene_library_shards


@pytest.mark.unit
def test_resolve_study_scene_library_plan_dedupes_identical_backbones():
    plans = resolve_study_scene_library_plan("PPO_NAVIGATION")

    assert len(plans) == 3
    assert any(plan.scene_library_path.endswith("ppo_navigation.db") for plan in plans)


@pytest.mark.unit
def test_resolve_study_scene_library_plan_can_filter_protocol_ids():
    plans = resolve_study_scene_library_plan(
        "PPO_NAVIGATION_DIFFICULTY",
        protocol_ids=["easy_train", "easy_eval"],
    )

    assert len(plans) == 2
    assert [plan.backbone_id for plan in plans] == ["easy:train", "easy:eval"]
    assert plans[0].protocol_ids == ("easy_train",)
    assert plans[1].protocol_ids == ("easy_eval",)
    assert all(plan.request_kwargs["scene_benchmark_id"] == "navigation_medium_v1" for plan in plans)
    assert all(plan.request_kwargs["scene_profile_id"] == "easy" for plan in plans)
    assert plans[0].request_kwargs["scene_split"] == "train"
    assert plans[1].request_kwargs["scene_split"] == "eval"
    assert all(plan.request_kwargs["num_vehicles"] == 2 for plan in plans)
    assert all(plan.request_kwargs["guaranteed_candidate_role"] == "lead" for plan in plans)


@pytest.mark.unit
def test_resolve_benchmark_scene_library_plan_builds_train_and_eval_slices():
    plans, prime_seeds, episodes_per_seed = resolve_benchmark_scene_library_plan(
        "navigation_medium_v1",
        profiles=["medium"],
    )

    assert [plan.backbone_id for plan in plans] == ["medium:train", "medium:eval"]
    assert all(plan.request_kwargs["scene_benchmark_id"] == "navigation_medium_v1" for plan in plans)
    assert plans[0].request_kwargs["scene_split"] == "train"
    assert plans[1].request_kwargs["scene_split"] == "eval"
    assert all(plan.scene_library_path.endswith("ppo_navigation_difficulty.db") for plan in plans)
    assert prime_seeds == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert episodes_per_seed == 1000


@pytest.mark.unit
def test_override_scene_library_paths_retargets_all_plans():
    plans, _, _ = resolve_benchmark_scene_library_plan(
        "navigation_medium_v1",
        profiles=["medium"],
    )

    overridden = override_scene_library_paths(plans, "assets/scene_libraries/shards/test.db")

    assert [plan.backbone_id for plan in overridden] == ["medium:train", "medium:eval"]
    assert all(plan.scene_library_path == "assets/scene_libraries/shards/test.db" for plan in overridden)
    assert all(plan.request_kwargs == src.request_kwargs for plan, src in zip(overridden, plans, strict=True))


@pytest.mark.unit
def test_merge_scene_library_shards_combines_rows_by_scene_key(tmp_path):
    shard_a = tmp_path / "navigation_medium_v1.medium.train.seed_2.db"
    shard_b = tmp_path / "navigation_medium_v1.medium.train.seed_3.db"
    output = tmp_path / "navigation_medium_v1.medium.train.merged.db"

    for db_path, scene_key, scene_seed in (
        (shard_a, "scene-a", 2),
        (shard_b, "scene-b", 3),
    ):
        library = SceneLibrary(str(db_path), generator_version="role_traffic_v1")
        library.insert_scene(
            scene_key=scene_key,
            scene_descriptor={
                "scene": "rdm",
                "scene_benchmark_id": "navigation_medium_v1",
                "scene_profile_id": "medium",
                "scene_split": "train",
                "difficulty_preset_id": "medium",
                "route_profile": "multi_turn",
                "main_actor_role": "lead",
                "scene_seed": scene_seed,
                "spawned_num_vehicles": 1,
                "spawned_num_vehicles_near_ego": 1,
                "spawned_role_counts": {"lead": 1},
            },
            scenario_context={"scene": "rdm"},
            actors_payload={"agent": None, "vehicle": [], "pedestrian": [], "target": [], "traffic_light": []},
            spawn_validation={"valid": True, "reason": "ok"},
            resolved_options={"scene": "rdm", "num_vehicles": 1},
        )

    summary = merge_scene_library_shards(
        output_path=str(output),
        input_patterns=[str(tmp_path / "navigation_medium_v1.medium.train.seed_*.db")],
    )

    merged = SceneLibrary(str(output), generator_version="role_traffic_v1")
    rows = merged.get_scene_by_filters(
        scene_benchmark_id="navigation_medium_v1",
        scene_profile_id="medium",
        scene_split="train",
        limit=10,
    )

    assert summary["input_databases"] == 2
    assert summary["rows_scanned"] == 2
    assert summary["rows_inserted"] == 2
    assert summary["rows_skipped_as_duplicates"] == 0
    assert summary["total_rows"] == 2
    assert {row["scene_key"] for row in rows} == {"scene-a", "scene-b"}
