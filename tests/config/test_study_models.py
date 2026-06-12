import pytest

from src.config.studies.models import ExperimentSpec


@pytest.mark.unit
def test_masks_require_semantic_mask_channel():
    with pytest.raises(ValueError, match="semantic_mask_ch"):
        ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="masks",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="train",
            eval_protocol_ids=["eval"],
        )


@pytest.mark.unit
def test_rgb_rejects_semantic_mask_channel():
    with pytest.raises(ValueError, match="must be omitted"):
        ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="rgb",
            semantic_mask_ch="6-class",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="train",
            eval_protocol_ids=["eval"],
        )


@pytest.mark.unit
def test_temporal_fusion_requires_supported_mask_layout():
    with pytest.raises(ValueError, match="vehicle channel"):
        ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="masks",
            semantic_mask_ch="2-class",
            temporal_fusion_mode="vehicle_temporal",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="train",
            eval_protocol_ids=["eval"],
        )


@pytest.mark.unit
def test_valid_temporal_fusion_spec_passes():
    spec = ExperimentSpec(
        action_mode="continuous",
        traffic="on",
        input_type="masks",
        semantic_mask_ch="6-class",
        temporal_fusion_mode="vehicle_weighted",
        reward_mode="carl",
        curriculum="route_only",
        fov_mask="on",
        train_protocol_id="train",
        eval_protocol_ids=["eval_a", "eval_b"],
    )

    assert spec.action_mode == "continuous"
    assert spec.temporal_fusion_mode == "vehicle_weighted"
    assert spec.eval_protocol_ids == ["eval_a", "eval_b"]
