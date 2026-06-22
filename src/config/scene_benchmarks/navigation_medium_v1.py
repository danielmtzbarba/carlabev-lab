from src.config.studies.models import (
    SceneBenchmarkConfig,
    SceneBenchmarkProfile,
    SceneGenerationBackbone,
    SceneLibraryPolicy,
)
from src.tuning.optuna_utils import DEFAULT_STUDY_PRIME_SEEDS


SCENE_LIBRARY_PATH = "assets/scene_libraries/ppo_navigation_difficulty.db"


def _profile(*, scene_profile_id: str, num_vehicles: int, role: str | None) -> SceneBenchmarkProfile:
    return SceneBenchmarkProfile(
        scene_profile_id=scene_profile_id,
        backbone=SceneGenerationBackbone(
            scene_profile_id=scene_profile_id,
            route_extent="medium",
            route_dist_range=(50, 130),
            speed_profile="medium",
            num_vehicles=num_vehicles,
            num_vehicles_near_ego=num_vehicles,
            traffic_role_profile=role,
            guaranteed_candidate_role=role,
            ego_route_graph="canonical",
        ),
    )


NAVIGATION_MEDIUM_V1 = SceneBenchmarkConfig(
    benchmark_id="navigation_medium_v1",
    description=(
        "Canonical medium-route navigation benchmark with fixed route extent, speed profile, "
        "and near-ego traffic profiles shared across navigation ablation studies."
    ),
    scene_library=SceneLibraryPolicy(
        enabled=True,
        read_only=True,
        require_hit=True,
        path=SCENE_LIBRARY_PATH,
        generator_version="role_traffic_v1",
    ),
    prime_seeds=list(DEFAULT_STUDY_PRIME_SEEDS),
    episodes_per_seed=1000,
    profiles={
        "no_traffic": _profile(scene_profile_id="no_traffic", num_vehicles=0, role=None),
        "easy": _profile(scene_profile_id="easy", num_vehicles=2, role="lead"),
        "medium": _profile(scene_profile_id="medium", num_vehicles=4, role="mix"),
        "hard": _profile(scene_profile_id="hard", num_vehicles=6, role="mix"),
    },
)
