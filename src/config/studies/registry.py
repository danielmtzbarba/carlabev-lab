from src.config.studies.edge_case_scenarios import EDGE_CASE_SCENARIOS
from src.config.studies.models import ExperimentSpec, StudyConfig
from src.config.studies.ppo_navigation import PPO_NAVIGATION


STUDY_REGISTRY: dict[str, StudyConfig] = {
    PPO_NAVIGATION.study_id: PPO_NAVIGATION,
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
