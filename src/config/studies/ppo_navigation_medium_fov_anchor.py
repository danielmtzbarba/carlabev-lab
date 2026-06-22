from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    SceneSourceRef,
    StudyConfig,
)


PPO_NAVIGATION_MEDIUM_FOV_ANCHOR = StudyConfig(
    study_id="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR",
    description=(
        "Focused navigation ablation at fixed medium near-ego traffic that isolates "
        "the effect of centered versus lookahead FOV anchoring."
    ),
    optuna_study_name="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR",
    db_path="results/ppo_navigation_medium_fov_anchor_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "fov_anchor pairwise ablation at fixed medium backbone",
        "scene_benchmark_id": "navigation_medium_v1",
    },
    train_protocols={
        "medium_train": RandomNavigationProtocol(
            protocol_id="medium_train",
            mode="random_navigation",
            reset_seed_mode="benchmark_hashed_episode",
            scene_source=SceneSourceRef(
                mode="benchmark",
                benchmark_id="navigation_medium_v1",
                scene_profile_id="medium",
                split="train",
            ),
            use_curriculum=False,
            curriculum_axis="none",
        ),
    },
    eval_protocols={
        "medium_eval": RandomNavigationProtocol(
            protocol_id="medium_eval",
            mode="random_navigation",
            reset_seed_mode="benchmark_hashed_episode",
            scene_source=SceneSourceRef(
                mode="benchmark",
                benchmark_id="navigation_medium_v1",
                scene_profile_id="medium",
                split="eval",
            ),
            use_curriculum=False,
            curriculum_axis="none",
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
            tags=["medium", "fov_anchor", "stack", "center"],
            notes="Fixed medium backbone with centered ego anchor.",
        ),
        2: ExperimentSpec(
            action_mode="discrete",
            action_profile_id="discrete9_v1",
            input_type="masks",
            semantic_mask_ch="4-class",
            temporal_fusion_mode="stack",
            reward_mode="carl",
            reward_profile_id="carl_base_v1",
            fov_mask="off",
            fov_anchor="lookahead_75",
            train_protocol_id="medium_train",
            eval_protocol_ids=["medium_eval"],
            tags=["medium", "fov_anchor", "stack", "lookahead_75"],
            notes="Fixed medium backbone with lookahead_75 ego anchor.",
        ),
    },
)
