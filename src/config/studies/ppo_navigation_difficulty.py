from src.config.studies.models import ExperimentSpec, RandomNavigationProtocol, StudyConfig


def _exp(*, exp_id: int, difficulty_id: str, traffic: str, tags: list[str]) -> ExperimentSpec:
    return ExperimentSpec(
        action_mode="discrete",
        action_profile_id="discrete9_v1",
        traffic=traffic,
        input_type="masks",
        semantic_mask_ch="4-class",
        temporal_fusion_mode="stack",
        reward_mode="carl",
        reward_profile_id="carl_base_v1",
        curriculum="off",
        difficulty_id=difficulty_id,
        fov_mask="off",
        fov_anchor="center",
        train_protocol_id="random_nav_train",
        eval_protocol_ids=["random_nav_eval"],
        tags=tags,
        notes=f"Difficulty ablation exp {exp_id} with fixed 4-class/stack/center backbone.",
    )


PPO_NAVIGATION_DIFFICULTY = StudyConfig(
    study_id="PPO_NAVIGATION_DIFFICULTY",
    description=(
        "Navigation difficulty ablation that isolates the effect of random-traffic difficulty "
        "while holding semantic classes, temporal fusion, and FOV anchor fixed."
    ),
    optuna_study_name="PPO_NAVIGATION_DIFFICULTY",
    db_path="results/ppo_navigation_difficulty_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "design": "difficulty_id ablation",
        "fixed_backbone": {
            "semantic_mask_ch": "4-class",
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
        1: _exp(exp_id=1, difficulty_id="rt_no_traffic_v1", traffic="off", tags=["difficulty", "no-traffic"]),
        2: _exp(exp_id=2, difficulty_id="rt_easy_v1", traffic="on", tags=["difficulty", "easy"]),
        3: _exp(exp_id=3, difficulty_id="rt_medium_v1", traffic="on", tags=["difficulty", "medium"]),
        4: _exp(exp_id=4, difficulty_id="rt_hard_v1", traffic="on", tags=["difficulty", "hard"]),
    },
)
