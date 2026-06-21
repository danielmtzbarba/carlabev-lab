from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    SceneGenerationBackbone,
    SceneLibraryPolicy,
    StudyConfig,
)


SCENE_LIBRARY_PATH = "assets/scene_libraries/ppo_navigation_medium_semantic_classes.db"


def _scene_library() -> SceneLibraryPolicy:
    return SceneLibraryPolicy(
        enabled=True,
        read_only=True,
        require_hit=True,
        path=SCENE_LIBRARY_PATH,
        generator_version="role_traffic_v1",
    )


MEDIUM_BACKBONE = SceneGenerationBackbone(
    route_extent="medium",
    route_dist_range=(50, 130),
    speed_profile="medium",
    num_vehicles=4,
    num_vehicles_near_ego=4,
    traffic_role_profile="mix",
    guaranteed_candidate_role="mix",
    ego_route_graph="canonical",
)


def _exp(*, exp_id: int, semantic_mask_ch: str) -> ExperimentSpec:
    return ExperimentSpec(
        action_mode="discrete",
        action_profile_id="discrete9_v1",
        input_type="masks",
        semantic_mask_ch=semantic_mask_ch,
        temporal_fusion_mode="stack",
        reward_mode="carl",
        reward_profile_id="carl_base_v1",
        fov_mask="off",
        fov_anchor="center",
        train_protocol_id="medium_train",
        eval_protocol_ids=["medium_eval"],
        tags=["medium", "semantic_classes", semantic_mask_ch],
        notes=f"Fixed medium backbone semantic class ablation exp {exp_id}.",
    )


PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES = StudyConfig(
    study_id="PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES",
    description=(
        "Medium-backbone semantic class ablation that isolates semantic observation granularity "
        "while holding route/traffic, temporal fusion, and FOV anchor fixed."
    ),
    optuna_study_name="PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES",
    db_path="results/ppo_navigation_medium_semantic_classes_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "semantic_mask_ch ablation at fixed medium backbone",
        "scene_library_path": SCENE_LIBRARY_PATH,
    },
    train_protocols={
        "medium_train": RandomNavigationProtocol(
            protocol_id="medium_train",
            mode="random_navigation",
            backbone=MEDIUM_BACKBONE,
            use_curriculum=False,
            curriculum_axis="none",
            scene_library=_scene_library(),
        ),
    },
    eval_protocols={
        "medium_eval": RandomNavigationProtocol(
            protocol_id="medium_eval",
            mode="random_navigation",
            backbone=MEDIUM_BACKBONE,
            use_curriculum=False,
            curriculum_axis="none",
            scene_library=_scene_library(),
        ),
    },
    experiments={
        1: _exp(exp_id=1, semantic_mask_ch="binary"),
        2: _exp(exp_id=2, semantic_mask_ch="2-class"),
        3: _exp(exp_id=3, semantic_mask_ch="4-class"),
        4: _exp(exp_id=4, semantic_mask_ch="5-class"),
        5: _exp(exp_id=5, semantic_mask_ch="6-class"),
        6: _exp(exp_id=6, semantic_mask_ch="7-class"),
    },
)
