import pytest

from src.carlabev_lab.scene_library.build import resolve_study_scene_library_plan


@pytest.mark.unit
def test_resolve_study_scene_library_plan_dedupes_identical_backbones():
    plans = resolve_study_scene_library_plan("PPO_NAVIGATION")

    assert [plan.backbone_id for plan in plans] == ["backbone_0", "backbone_1", "backbone_2"]
    assert len(plans) == 3
    assert any(plan.scene_library_path.endswith("ppo_navigation.db") for plan in plans)


@pytest.mark.unit
def test_resolve_study_scene_library_plan_can_filter_protocol_ids():
    plans = resolve_study_scene_library_plan(
        "PPO_NAVIGATION_DIFFICULTY",
        protocol_ids=["easy_train", "easy_eval"],
    )

    assert len(plans) == 1
    assert plans[0].protocol_ids == ("easy_eval", "easy_train")
    assert plans[0].request_kwargs["num_vehicles"] == 2
    assert plans[0].request_kwargs["guaranteed_candidate_role"] == "lead"
