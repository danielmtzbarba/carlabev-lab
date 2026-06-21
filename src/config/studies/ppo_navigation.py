from src.config.studies.models import (
    ExperimentSpec,
    RandomNavigationProtocol,
    SceneGenerationBackbone,
    SceneLibraryPolicy,
    StudyConfig,
    TuningConfig,
    TuningStageConfig,
)


SCENE_LIBRARY_PATH = "assets/scene_libraries/ppo_navigation.db"


def _scene_library() -> SceneLibraryPolicy:
    return SceneLibraryPolicy(
        enabled=True,
        read_only=True,
        require_hit=True,
        path=SCENE_LIBRARY_PATH,
        generator_version="role_traffic_v1",
    )


def _backbone(*, num_vehicles: int, route_dist_range: tuple[int, int]) -> SceneGenerationBackbone:
    return SceneGenerationBackbone(
        route_dist_range=route_dist_range,
        speed_profile="medium",
        num_vehicles=num_vehicles,
        num_vehicles_near_ego=num_vehicles,
        traffic_role_profile="mix" if num_vehicles > 0 else None,
        guaranteed_candidate_role="mix" if num_vehicles > 0 else None,
        ego_route_graph="canonical",
    )


def _protocol(
    protocol_id: str,
    *,
    num_vehicles: int,
    route_dist_range: tuple[int, int],
    use_curriculum: bool = False,
    curriculum_axis: str = "none",
) -> RandomNavigationProtocol:
    return RandomNavigationProtocol(
        protocol_id=protocol_id,
        mode="random_navigation",
        backbone=_backbone(num_vehicles=num_vehicles, route_dist_range=route_dist_range),
        use_curriculum=use_curriculum,
        curriculum_axis=curriculum_axis,
        scene_library=_scene_library(),
    )


PPO_NAVIGATION = StudyConfig(
    study_id="PPO_NAVIGATION",
    description="Primary PPO navigation study covering action space, near-ego traffic, input, reward, curriculum, and FOV masking variants.",
    optuna_study_name="PPO_NAVIGATION",
    db_path="results/ppo_navigation_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "navigation",
        "notes": "Migrated from the legacy global EXPERIMENT_TABLE navigation matrix.",
        "scene_library_path": SCENE_LIBRARY_PATH,
    },
    tuning=TuningConfig(
        algorithm="cnn-ppo",
        objective_metric="normalized_score",
        sampler="tpe",
        pruner="median",
        num_seeds=3,
        eval_episodes=30,
        eval_final_episodes=100,
        stages={
            "policy_dynamics": TuningStageConfig(
                stage_id="policy_dynamics",
                description="Tune learning rate, GAE lambda, and discount factor.",
                n_trials=100,
                total_timesteps=1_100_000,
                selection_top_k=10,
                save_model=False,
                capture_video=False,
            ),
            "rollout_geometry": TuningStageConfig(
                stage_id="rollout_geometry",
                description="Tune PPO rollout horizon and minibatch/update geometry.",
                n_trials=60,
                total_timesteps=1_100_000,
                inherits_from=["policy_dynamics"],
                selection_top_k=10,
                save_model=False,
                capture_video=False,
            ),
            "loss_regularization": TuningStageConfig(
                stage_id="loss_regularization",
                description="Tune PPO clipping, entropy, value loss, and decay schedules.",
                n_trials=100,
                total_timesteps=1_100_000,
                inherits_from=["policy_dynamics", "rollout_geometry"],
                selection_top_k=5,
                save_model=False,
                capture_video=False,
            ),
            "network_capacity": TuningStageConfig(
                stage_id="network_capacity",
                description="Tune convolutional channel scale and fully connected size.",
                n_trials=50,
                total_timesteps=2_100_000,
                inherits_from=[
                    "policy_dynamics",
                    "rollout_geometry",
                    "loss_regularization",
                ],
                selection_top_k=5,
                save_model=False,
                capture_video=False,
            ),
        },
    ),
    train_protocols={
        "no_traffic_fixed_train": _protocol(
            "no_traffic_fixed_train",
            num_vehicles=0,
            route_dist_range=(50, 150),
        ),
        "no_traffic_route_curriculum_train": _protocol(
            "no_traffic_route_curriculum_train",
            num_vehicles=0,
            route_dist_range=(50, 150),
            use_curriculum=True,
            curriculum_axis="route_distance",
        ),
        "traffic_fixed_train": _protocol(
            "traffic_fixed_train",
            num_vehicles=4,
            route_dist_range=(50, 150),
        ),
        "traffic_vehicles_curriculum_train": _protocol(
            "traffic_vehicles_curriculum_train",
            num_vehicles=4,
            route_dist_range=(50, 150),
            use_curriculum=True,
            curriculum_axis="near_ego_traffic",
        ),
        "traffic_route_curriculum_train": _protocol(
            "traffic_route_curriculum_train",
            num_vehicles=4,
            route_dist_range=(50, 150),
            use_curriculum=True,
            curriculum_axis="route_distance",
        ),
        "traffic_both_curriculum_train": _protocol(
            "traffic_both_curriculum_train",
            num_vehicles=4,
            route_dist_range=(50, 150),
            use_curriculum=True,
            curriculum_axis="both",
        ),
    },
    eval_protocols={
        "navigation_eval": _protocol(
            "navigation_eval",
            num_vehicles=4,
            route_dist_range=(250, 500),
        ),
    },
    experiments={
        1: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        2: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        3: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        4: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        5: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        6: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        7: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        8: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        9: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        10: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_both_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        11: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        12: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_both_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        13: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        14: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="shaping", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_both_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        15: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_fixed_train", eval_protocol_ids=["navigation_eval"]),
        16: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_both_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        17: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_vehicles_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        18: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        19: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_both_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        20: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_vehicles_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        21: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        22: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_both_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        23: ExperimentSpec(action_mode="discrete", input_type="rgb", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        24: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="no_traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        25: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_fixed_train", eval_protocol_ids=["navigation_eval"], notes="Navigation baseline without curriculum."),
        26: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        27: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="on", fov_anchor="center", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        28: ExperimentSpec(action_mode="continuous", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="center", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        29: ExperimentSpec(action_mode="continuous", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="on", fov_anchor="center", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"]),
        30: ExperimentSpec(action_mode="discrete", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="lookahead_75", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"], notes="Route-only navigation with lookahead_75 FOV anchor."),
        31: ExperimentSpec(action_mode="continuous", input_type="masks", semantic_mask_ch="6-class", reward_mode="carl", fov_mask="off", fov_anchor="lookahead_75", train_protocol_id="traffic_route_curriculum_train", eval_protocol_ids=["navigation_eval"], notes="Continuous-control route-only navigation with lookahead_75 FOV anchor."),
    },
)
