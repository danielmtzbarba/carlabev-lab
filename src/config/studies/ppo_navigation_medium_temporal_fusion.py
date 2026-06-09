from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    StudyConfig,
)


PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION = StudyConfig(
    study_id="PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION",
    description=(
        "Focused navigation ablation at medium traffic difficulty that isolates "
        "the effect of temporal fusion mode."
    ),
    optuna_study_name="PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION",
    db_path="results/ppo_navigation_medium_temporal_fusion_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "parent_study_id": "PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION",
        "parent_exp_id": 13,
        "design": "temporal_fusion_mode ablation at fixed medium difficulty",
        "notes": (
            "All runs hold the backbone fixed: rt_medium_v1 difficulty, discrete9 "
            "action profile, 4-class semantic masks, centered ego anchor, CARL base "
            "reward profile, and curriculum off."
        ),
    },
    train_protocols={
        "random_nav_train": RandomNavigationProtocol(
            protocol_id="random_nav_train",
            mode="random_navigation",
            initial_num_vehicles=25,
            initial_route_dist_range=(50, 130),
            use_curriculum=False,
        ),
    },
    eval_protocols={
        "random_nav_eval": RandomNavigationProtocol(
            protocol_id="random_nav_eval",
            mode="random_navigation",
            initial_num_vehicles=25,
            initial_route_dist_range=(50, 130),
            eval_num_vehicles=25,
            eval_route_dist_range=(50, 130),
            use_curriculum=False,
        ),
    },
    experiments={
        1: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            traffic="on",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="stack",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            curriculum="off",
            difficulty_id="rt_medium_v1",
            fov_mask="off",
            fov_anchor="center",
            train_protocol_id="random_nav_train",
            eval_protocol_ids=["random_nav_eval"],
            tags=["medium", "stack", "center"],
            notes="Fixed medium-difficulty backbone with stacked semantic masks only.",
        ),
        2: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            traffic="on",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="vehicle_temporal",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            curriculum="off",
            difficulty_id="rt_medium_v1",
            fov_mask="off",
            fov_anchor="center",
            train_protocol_id="random_nav_train",
            eval_protocol_ids=["random_nav_eval"],
            tags=["medium", "vehicle_temporal", "center"],
            notes="Fixed medium-difficulty backbone with separate recent vehicle-history channels.",
        ),
        3: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            traffic="on",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="vehicle_weighted",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            curriculum="off",
            difficulty_id="rt_medium_v1",
            fov_mask="off",
            fov_anchor="center",
            train_protocol_id="random_nav_train",
            eval_protocol_ids=["random_nav_eval"],
            tags=["medium", "vehicle_weighted", "center"],
            notes="Fixed medium-difficulty backbone with weighted vehicle-history fusion.",
        ),
    },
)
