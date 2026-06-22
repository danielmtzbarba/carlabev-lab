import pytest

from src.config.studies.registry import get_train_protocol
from src.trainers.utils import CurriculumState, resolve_protocol_backbone


@pytest.mark.unit
def test_resolve_protocol_backbone_supports_benchmark_backed_protocols():
    protocol = get_train_protocol("PPO_NAVIGATION_DIFFICULTY", "medium_train")

    backbone = resolve_protocol_backbone(protocol)

    assert backbone.scene_profile_id == "medium"
    assert backbone.num_vehicles == 4
    assert backbone.num_vehicles_near_ego == 4
    assert backbone.speed_profile == "medium"


@pytest.mark.unit
def test_curriculum_state_uses_resolved_benchmark_backbone():
    protocol = get_train_protocol("PPO_NAVIGATION_DIFFICULTY", "hard_train")

    state = CurriculumState(protocol)

    assert state.backbone.scene_profile_id == "hard"
    assert state.max_cars == 6
