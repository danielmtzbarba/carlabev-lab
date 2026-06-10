from src.config.studies.models import ExperimentSpec, RandomNavigationProtocol, StudyConfig


def _exp(*, exp_id: int, semantic_mask_ch: str) -> ExperimentSpec:
    return ExperimentSpec(
        action_mode="discrete",
        action_profile_id="discrete9_v1",
        traffic="on",
        input_type="masks",
        semantic_mask_ch=semantic_mask_ch,
        temporal_fusion_mode="stack",
        reward_mode="carl",
        reward_profile_id="carl_base_v1",
        curriculum="off",
        difficulty_id="rt_medium_v1",
        fov_mask="off",
        fov_anchor="center",
        train_protocol_id="random_nav_train",
        eval_protocol_ids=["random_nav_eval"],
        tags=["medium", "semantic_classes", semantic_mask_ch],
        notes=f"Medium-difficulty semantic class ablation exp {exp_id}.",
    )


PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES = StudyConfig(
    study_id="PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES",
    description=(
        "Medium-difficulty semantic class ablation that isolates semantic observation granularity "
        "while holding temporal fusion and FOV anchor fixed."
    ),
    optuna_study_name="PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES",
    db_path="results/ppo_navigation_medium_semantic_classes_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "semantic_mask_ch ablation at fixed medium difficulty",
        "fixed_backbone": {
            "difficulty_id": "rt_medium_v1",
            "temporal_fusion_mode": "stack",
            "fov_anchor": "center",
            "action_profile_id": "discrete9_v1",
            "reward_profile_id": "carl_base_v1",
            "curriculum": "off",
        },
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
        1: _exp(exp_id=1, semantic_mask_ch="binary"),
        2: _exp(exp_id=2, semantic_mask_ch="2-class"),
        3: _exp(exp_id=3, semantic_mask_ch="4-class"),
        4: _exp(exp_id=4, semantic_mask_ch="5-class"),
        5: _exp(exp_id=5, semantic_mask_ch="6-class"),
        6: _exp(exp_id=6, semantic_mask_ch="7-class"),
    },
)
