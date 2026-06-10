from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    StudyConfig,
)


PPO_NAVIGATION_MEDIUM_FOV_ANCHOR = StudyConfig(
    study_id="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR",
    description=(
        "Focused navigation ablation at medium traffic difficulty that isolates "
        "the effect of centered versus lookahead FOV anchoring."
    ),
    optuna_study_name="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR",
    db_path="results/ppo_navigation_medium_fov_anchor_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "fov_anchor pairwise ablation at fixed medium difficulty",
        "notes": (
            "All runs hold the backbone fixed: rt_medium_v1 difficulty, discrete9 "
            "action profile, 4-class semantic masks, stack temporal mode, CARL base "
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
            tags=["medium", "fov_anchor", "stack", "center"],
            notes="Fixed medium-difficulty backbone with centered ego anchor.",
        ),
        2: ExperimentSpec(
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
            fov_anchor="lookahead_75",
            train_protocol_id="random_nav_train",
            eval_protocol_ids=["random_nav_eval"],
            tags=["medium", "fov_anchor", "stack", "lookahead_75"],
            notes="Fixed medium-difficulty backbone with lookahead_75 ego anchor.",
        ),
    },
)
