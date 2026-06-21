from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    SceneGenerationBackbone,
    SceneLibraryPolicy,
    StudyConfig,
)


SCENE_LIBRARY_PATH = "assets/scene_libraries/ppo_navigation_medium_temporal_fusion.db"


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


PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION = StudyConfig(
    study_id="PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION",
    description=(
        "Focused navigation ablation at fixed medium near-ego traffic that isolates "
        "the effect of temporal fusion mode."
    ),
    optuna_study_name="PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION",
    db_path="results/ppo_navigation_medium_temporal_fusion_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "temporal_fusion_mode ablation at fixed medium backbone",
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
        1: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="stack",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            fov_mask="off",
            fov_anchor="center",
            train_protocol_id="medium_train",
            eval_protocol_ids=["medium_eval"],
            tags=["medium", "temporal_fusion", "stack", "center"],
            notes="Fixed medium backbone with stacked semantic masks only.",
        ),
        2: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="vehicle_temporal",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            fov_mask="off",
            fov_anchor="center",
            train_protocol_id="medium_train",
            eval_protocol_ids=["medium_eval"],
            tags=["medium", "temporal_fusion", "vehicle_temporal", "center"],
            notes="Fixed medium backbone with separate recent vehicle-history channels.",
        ),
        3: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="vehicle_weighted",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            fov_mask="off",
            fov_anchor="center",
            train_protocol_id="medium_train",
            eval_protocol_ids=["medium_eval"],
            tags=["medium", "temporal_fusion", "vehicle_weighted", "center"],
            notes="Fixed medium backbone with weighted vehicle-history fusion.",
        ),
    },
)
