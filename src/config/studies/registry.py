from src.config.studies.edge_case_scenarios import EDGE_CASE_SCENARIOS
from src.config.studies.models import ExperimentSpec, ProtocolSpec, StudyConfig
from src.config.studies.ppo_navigation_difficulty_temporal_fusion import (
    PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION,
)
from src.config.studies.ppo_navigation_medium_fov_anchor import (
    PPO_NAVIGATION_MEDIUM_FOV_ANCHOR,
)
from src.config.studies.ppo_navigation_medium_temporal_fusion import (
    PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION,
)
from src.config.studies.ppo_navigation import PPO_NAVIGATION
from src.config.studies.ppo_navigation_semantic_lookahead import (
    PPO_NAVIGATION_SEMANTIC_LOOKAHEAD,
)
from src.config.studies.ppo_navigation_vehicle_temporal_fusion import (
    PPO_NAVIGATION_VEHICLE_TEMPORAL_FUSION,
)


STUDY_REGISTRY: dict[str, StudyConfig] = {
    PPO_NAVIGATION.study_id: PPO_NAVIGATION,
    PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION.study_id: PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION,
    PPO_NAVIGATION_MEDIUM_FOV_ANCHOR.study_id: PPO_NAVIGATION_MEDIUM_FOV_ANCHOR,
    PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION.study_id: PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION,
    PPO_NAVIGATION_SEMANTIC_LOOKAHEAD.study_id: PPO_NAVIGATION_SEMANTIC_LOOKAHEAD,
    PPO_NAVIGATION_VEHICLE_TEMPORAL_FUSION.study_id: PPO_NAVIGATION_VEHICLE_TEMPORAL_FUSION,
    EDGE_CASE_SCENARIOS.study_id: EDGE_CASE_SCENARIOS,
}


def list_study_ids() -> list[str]:
    return sorted(STUDY_REGISTRY.keys())


def get_study_config(study_id: str) -> StudyConfig:
    try:
        return STUDY_REGISTRY[study_id]
    except KeyError as exc:
        available = ", ".join(list_study_ids())
        raise KeyError(f"Unknown study_id '{study_id}'. Available studies: {available}") from exc


def get_experiment_spec(study_id: str, exp_id: int) -> ExperimentSpec:
    study = get_study_config(study_id)
    try:
        return study.experiments[exp_id]
    except KeyError as exc:
        available = ", ".join(str(key) for key in sorted(study.experiments))
        raise KeyError(
            f"Unknown exp_id '{exp_id}' for study '{study_id}'. Available exp_ids: {available}"
        ) from exc


def get_train_protocol(study_id: str, protocol_id: str) -> ProtocolSpec:
    study = get_study_config(study_id)
    try:
        return study.train_protocols[protocol_id]
    except KeyError as exc:
        available = ", ".join(sorted(study.train_protocols))
        raise KeyError(
            f"Unknown train protocol '{protocol_id}' for study '{study_id}'. Available train protocols: {available}"
        ) from exc


def get_eval_protocol(study_id: str, protocol_id: str) -> ProtocolSpec:
    study = get_study_config(study_id)
    try:
        return study.eval_protocols[protocol_id]
    except KeyError as exc:
        available = ", ".join(sorted(study.eval_protocols))
        raise KeyError(
            f"Unknown eval protocol '{protocol_id}' for study '{study_id}'. Available eval protocols: {available}"
        ) from exc
