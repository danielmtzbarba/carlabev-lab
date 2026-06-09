from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    StudyConfig,
)


def _exp(
    *,
    difficulty_id: str,
    traffic: str,
    temporal_fusion_mode: str,
    fov_anchor: str,
    tags: list[str],
) -> ExperimentSpec:
    return ExperimentSpec(
        action_mode="discrete",
        action_profile_id="discrete9_v1",
        traffic=traffic,
        input_type="masks",
        semantic_mask_ch="4-class",
        temporal_fusion_mode=temporal_fusion_mode,
        reward_mode="carl",
        reward_profile_id="carl_base_v1",
        curriculum="off",
        difficulty_id=difficulty_id,
        fov_mask="off",
        fov_anchor=fov_anchor,
        train_protocol_id="random_nav_train",
        eval_protocol_ids=["random_nav_eval"],
        tags=tags,
    )


PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION = StudyConfig(
    study_id="PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION",
    description=(
        "Navigation study that fixes the strongest 1M screening backbone and sweeps "
        "random-traffic difficulty against temporal fusion mode and FOV anchor."
    ),
    optuna_study_name="PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION",
    db_path="results/ppo_navigation_difficulty_temporal_fusion_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "parent_study_id": "PPO_NAVIGATION_SEMANTIC_LOOKAHEAD",
        "parent_exp_id": 5,
        "design": "difficulty_id x temporal_fusion_mode x fov_anchor full factorial",
        "notes": (
            "All runs hold the backbone fixed: discrete9 action profile, 4-class "
            "semantic masks, CARL base reward profile, and no curriculum. The "
            "experiment isolates traffic difficulty, temporal fusion, and FOV anchor."
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
        1: _exp(
            difficulty_id="rt_no_traffic_v1",
            traffic="off",
            temporal_fusion_mode="stack",
            fov_anchor="center",
            tags=["difficulty", "no-traffic", "stack", "center"],
        ),
        2: _exp(
            difficulty_id="rt_no_traffic_v1",
            traffic="off",
            temporal_fusion_mode="stack",
            fov_anchor="lookahead_75",
            tags=["difficulty", "no-traffic", "stack", "lookahead_75"],
        ),
        3: _exp(
            difficulty_id="rt_no_traffic_v1",
            traffic="off",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="center",
            tags=["difficulty", "no-traffic", "vehicle_temporal", "center"],
        ),
        4: _exp(
            difficulty_id="rt_no_traffic_v1",
            traffic="off",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="lookahead_75",
            tags=["difficulty", "no-traffic", "vehicle_temporal", "lookahead_75"],
        ),
        5: _exp(
            difficulty_id="rt_no_traffic_v1",
            traffic="off",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="center",
            tags=["difficulty", "no-traffic", "vehicle_weighted", "center"],
        ),
        6: _exp(
            difficulty_id="rt_no_traffic_v1",
            traffic="off",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="lookahead_75",
            tags=["difficulty", "no-traffic", "vehicle_weighted", "lookahead_75"],
        ),
        7: _exp(
            difficulty_id="rt_easy_v1",
            traffic="on",
            temporal_fusion_mode="stack",
            fov_anchor="center",
            tags=["difficulty", "easy", "stack", "center"],
        ),
        8: _exp(
            difficulty_id="rt_easy_v1",
            traffic="on",
            temporal_fusion_mode="stack",
            fov_anchor="lookahead_75",
            tags=["difficulty", "easy", "stack", "lookahead_75"],
        ),
        9: _exp(
            difficulty_id="rt_easy_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="center",
            tags=["difficulty", "easy", "vehicle_temporal", "center"],
        ),
        10: _exp(
            difficulty_id="rt_easy_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="lookahead_75",
            tags=["difficulty", "easy", "vehicle_temporal", "lookahead_75"],
        ),
        11: _exp(
            difficulty_id="rt_easy_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="center",
            tags=["difficulty", "easy", "vehicle_weighted", "center"],
        ),
        12: _exp(
            difficulty_id="rt_easy_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="lookahead_75",
            tags=["difficulty", "easy", "vehicle_weighted", "lookahead_75"],
        ),
        13: _exp(
            difficulty_id="rt_medium_v1",
            traffic="on",
            temporal_fusion_mode="stack",
            fov_anchor="center",
            tags=["difficulty", "medium", "stack", "center"],
        ),
        14: _exp(
            difficulty_id="rt_medium_v1",
            traffic="on",
            temporal_fusion_mode="stack",
            fov_anchor="lookahead_75",
            tags=["difficulty", "medium", "stack", "lookahead_75"],
        ),
        15: _exp(
            difficulty_id="rt_medium_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="center",
            tags=["difficulty", "medium", "vehicle_temporal", "center"],
        ),
        16: _exp(
            difficulty_id="rt_medium_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="lookahead_75",
            tags=["difficulty", "medium", "vehicle_temporal", "lookahead_75"],
        ),
        17: _exp(
            difficulty_id="rt_medium_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="center",
            tags=["difficulty", "medium", "vehicle_weighted", "center"],
        ),
        18: _exp(
            difficulty_id="rt_medium_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="lookahead_75",
            tags=["difficulty", "medium", "vehicle_weighted", "lookahead_75"],
        ),
        19: _exp(
            difficulty_id="rt_hard_v1",
            traffic="on",
            temporal_fusion_mode="stack",
            fov_anchor="center",
            tags=["difficulty", "hard", "stack", "center"],
        ),
        20: _exp(
            difficulty_id="rt_hard_v1",
            traffic="on",
            temporal_fusion_mode="stack",
            fov_anchor="lookahead_75",
            tags=["difficulty", "hard", "stack", "lookahead_75"],
        ),
        21: _exp(
            difficulty_id="rt_hard_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="center",
            tags=["difficulty", "hard", "vehicle_temporal", "center"],
        ),
        22: _exp(
            difficulty_id="rt_hard_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_temporal",
            fov_anchor="lookahead_75",
            tags=["difficulty", "hard", "vehicle_temporal", "lookahead_75"],
        ),
        23: _exp(
            difficulty_id="rt_hard_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="center",
            tags=["difficulty", "hard", "vehicle_weighted", "center"],
        ),
        24: _exp(
            difficulty_id="rt_hard_v1",
            traffic="on",
            temporal_fusion_mode="vehicle_weighted",
            fov_anchor="lookahead_75",
            tags=["difficulty", "hard", "vehicle_weighted", "lookahead_75"],
        ),
    },
)
