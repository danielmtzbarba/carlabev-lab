from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    SceneGenerationBackbone,
    SceneLibraryPolicy,
    StudyConfig,
)


SCENE_LIBRARY_PATH = "assets/scene_libraries/ppo_navigation_difficulty.db"


def _scene_library() -> SceneLibraryPolicy:
    return SceneLibraryPolicy(
        enabled=True,
        read_only=True,
        require_hit=True,
        path=SCENE_LIBRARY_PATH,
        generator_version="role_traffic_v1",
    )


def _backbone(*, scene_profile_id: str, num_vehicles: int, role: str | None) -> SceneGenerationBackbone:
    return SceneGenerationBackbone(
        scene_profile_id=scene_profile_id,
        route_extent="medium",
        route_dist_range=(50, 130),
        speed_profile="medium",
        num_vehicles=num_vehicles,
        num_vehicles_near_ego=num_vehicles,
        traffic_role_profile=role,
        guaranteed_candidate_role=role,
        ego_route_graph="canonical",
    )


def _protocol(
    protocol_id: str,
    *,
    scene_profile_id: str,
    num_vehicles: int,
    role: str | None,
) -> RandomNavigationProtocol:
    return RandomNavigationProtocol(
        protocol_id=protocol_id,
        mode="random_navigation",
        backbone=_backbone(scene_profile_id=scene_profile_id, num_vehicles=num_vehicles, role=role),
        use_curriculum=False,
        curriculum_axis="none",
        scene_library=_scene_library(),
    )


def _exp(*, exp_id: int, train_protocol_id: str, eval_protocol_id: str, tags: list[str]) -> ExperimentSpec:
    return ExperimentSpec(
        action_mode="discrete",
        action_profile_id="discrete9_v1",
        input_type="masks",
        semantic_mask_ch="4-class",
        temporal_fusion_mode="stack",
        reward_mode="carl",
        reward_profile_id="carl_base_v1",
        fov_mask="off",
        fov_anchor="center",
        train_protocol_id=train_protocol_id,
        eval_protocol_ids=[eval_protocol_id],
        tags=tags,
        notes=f"Traffic-density ablation exp {exp_id} with fixed 4-class/stack/center backbone.",
    )


PPO_NAVIGATION_DIFFICULTY = StudyConfig(
    study_id="PPO_NAVIGATION_DIFFICULTY",
    description=(
        "Navigation traffic-density ablation that isolates near-ego traffic load "
        "while holding route extent, speed profile, semantic classes, temporal fusion, and FOV anchor fixed."
    ),
    optuna_study_name="PPO_NAVIGATION_DIFFICULTY",
    db_path="results/ppo_navigation_difficulty_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "near-ego traffic density ablation",
        "fixed_backbone": {
            "route_extent": "medium",
            "route_dist_range": [50, 130],
            "speed_profile": "medium",
            "semantic_mask_ch": "4-class",
            "temporal_fusion_mode": "stack",
            "fov_anchor": "center",
            "action_profile_id": "discrete9_v1",
            "reward_profile_id": "carl_base_v1",
        },
        "scene_library_path": SCENE_LIBRARY_PATH,
    },
    train_protocols={
        "no_traffic_train": _protocol("no_traffic_train", scene_profile_id="no_traffic", num_vehicles=0, role=None),
        "easy_train": _protocol("easy_train", scene_profile_id="easy", num_vehicles=2, role="lead"),
        "medium_train": _protocol("medium_train", scene_profile_id="medium", num_vehicles=4, role="mix"),
        "hard_train": _protocol("hard_train", scene_profile_id="hard", num_vehicles=6, role="mix"),
    },
    eval_protocols={
        "no_traffic_eval": _protocol("no_traffic_eval", scene_profile_id="no_traffic", num_vehicles=0, role=None),
        "easy_eval": _protocol("easy_eval", scene_profile_id="easy", num_vehicles=2, role="lead"),
        "medium_eval": _protocol("medium_eval", scene_profile_id="medium", num_vehicles=4, role="mix"),
        "hard_eval": _protocol("hard_eval", scene_profile_id="hard", num_vehicles=6, role="mix"),
    },
    experiments={
        1: _exp(
            exp_id=1,
            train_protocol_id="no_traffic_train",
            eval_protocol_id="no_traffic_eval",
            tags=["traffic-density", "no-traffic"],
        ),
        2: _exp(
            exp_id=2,
            train_protocol_id="easy_train",
            eval_protocol_id="easy_eval",
            tags=["traffic-density", "easy"],
        ),
        3: _exp(
            exp_id=3,
            train_protocol_id="medium_train",
            eval_protocol_id="medium_eval",
            tags=["traffic-density", "medium"],
        ),
        4: _exp(
            exp_id=4,
            train_protocol_id="hard_train",
            eval_protocol_id="hard_eval",
            tags=["traffic-density", "hard"],
        ),
    },
)
