import pytest

from src.carlabev_lab.scene_library.build import (
    resolve_benchmark_scene_library_plan,
    resolve_study_scene_library_plan,
)


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
