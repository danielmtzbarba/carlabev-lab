import pytest

from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    SceneGenerationBackbone,
    SceneSourceRef,
)


@pytest.mark.unit
def test_masks_require_semantic_mask_channel():
    with pytest.raises(ValueError, match="semantic_mask_ch"):
        ExperimentSpec(
            action_mode="discrete",
            input_type="masks",
            reward_mode="carl",
            fov_mask="off",
            train_protocol_id="train",
            eval_protocol_ids=["eval"],
        )


@pytest.mark.unit
def test_rgb_rejects_semantic_mask_channel():
    with pytest.raises(ValueError, match="must be omitted"):
        ExperimentSpec(
            action_mode="discrete",
            input_type="rgb",
            semantic_mask_ch="6-class",
            reward_mode="carl",
            fov_mask="off",
            train_protocol_id="train",
            eval_protocol_ids=["eval"],
        )


@pytest.mark.unit
def test_temporal_fusion_requires_supported_mask_layout():
    with pytest.raises(ValueError, match="vehicle channel"):
        ExperimentSpec(
            action_mode="discrete",
            input_type="masks",
            semantic_mask_ch="2-class",
            temporal_fusion_mode="vehicle_temporal",
            reward_mode="carl",
            fov_mask="off",
            train_protocol_id="train",
            eval_protocol_ids=["eval"],
        )


@pytest.mark.unit
def test_valid_temporal_fusion_spec_passes():
    spec = ExperimentSpec(
        action_mode="continuous",
        input_type="masks",
        semantic_mask_ch="6-class",
        temporal_fusion_mode="vehicle_weighted",
        reward_mode="carl",
        fov_mask="on",
        train_protocol_id="train",
        eval_protocol_ids=["eval_a", "eval_b"],
    )

    assert spec.action_mode == "continuous"
    assert spec.temporal_fusion_mode == "vehicle_weighted"
    assert spec.eval_protocol_ids == ["eval_a", "eval_b"]


@pytest.mark.unit
def test_scene_generation_backbone_rejects_invalid_near_ego_count():
    with pytest.raises(ValueError, match="num_vehicles_near_ego"):
        SceneGenerationBackbone(num_vehicles=2, num_vehicles_near_ego=3)


@pytest.mark.unit
def test_benchmark_protocol_requires_benchmark_seed_mode():
    with pytest.raises(ValueError, match="benchmark_hashed_episode"):
        RandomNavigationProtocol(
            protocol_id="medium_train",
            mode="random_navigation",
            scene_source=SceneSourceRef(
                mode="benchmark",
                benchmark_id="navigation_medium_v1",
                scene_profile_id="medium",
            ),
        )
