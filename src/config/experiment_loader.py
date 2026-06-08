import os
import random
import time
import traceback

import tyro
import yaml
from CarlaBEV.config import validate_run_config

from src.config.base_config import ArgsCarlaBEV, to_carlabev_run_config
from src.config.studies.models import ExperimentSpec, StudyConfig
from src.config.studies.registry import (
    get_eval_protocol,
    get_experiment_spec,
    get_study_config,
    get_train_protocol,
)


def get_study_name(study_id: str) -> str:
    return get_study_config(study_id).optuna_study_name


def get_study_db_path(study_id: str) -> str:
    return get_study_config(study_id).db_path


def apply_experiment_config(
    args: ArgsCarlaBEV,
    exp_id: int,
    study_id: str | None = None,
) -> ArgsCarlaBEV:
    study: StudyConfig = get_study_config(study_id or args.study_id)
    experiment: ExperimentSpec = get_experiment_spec(study.study_id, exp_id)

    args.study_id = study.study_id
    args.exp_id = exp_id
    args.algorithm = study.default_algorithm
    args.train_protocol_id = experiment.train_protocol_id
    args.eval_protocol_ids = list(experiment.eval_protocol_ids)

    env = args.env

    env.action_mode = experiment.action_mode
    env.fov_masked = experiment.fov_mask == "on"
    if experiment.fov_anchor == "center":
        env.ego_anchor_x_frac = 0.5
        env.ego_anchor_y_frac = 0.5
    elif experiment.fov_anchor == "lookahead_75":
        env.ego_anchor_x_frac = 0.5
        env.ego_anchor_y_frac = 0.75
    else:
        raise ValueError(f"Unsupported fov_anchor={experiment.fov_anchor!r}")
    env.traffic_enabled = experiment.traffic == "on"
    env.input_type = experiment.input_type
    env.semantic_mask_ch = (
        experiment.semantic_mask_ch
        if experiment.semantic_mask_ch is not None
        else env.semantic_mask_ch
    )
    env.temporal_fusion_mode = experiment.temporal_fusion_mode
    env.reward_mode = experiment.reward_mode

    if experiment.curriculum == "off":
        env.curriculum_enabled = False
    else:
        env.curriculum_enabled = True
        if experiment.curriculum == "vehicles_only":
            env.curriculum_mode = "vehicles"
        elif experiment.curriculum == "route_only":
            env.curriculum_mode = "route"
        else:
            env.curriculum_mode = "both"

    args.exp_name = (
        f"{study.study_id}_exp-{exp_id}_{args.algorithm}"
        f"_act-{experiment.action_mode}"
        f"_traffic-{experiment.traffic}"
        f"_input-{experiment.input_type}"
        f"_sem-{args.env.semantic_mask_ch if experiment.input_type == 'masks' else 'rgb'}"
        f"_tfuse-{args.env.temporal_fusion_mode}"
        f"_rwd-{experiment.reward_mode}"
        f"_curr-{experiment.curriculum}"
        f"_fovmask-{experiment.fov_mask}"
        f"_fovanchor-{experiment.fov_anchor}"
    )

    return args


def save_run_config(args: ArgsCarlaBEV):
    study = get_study_config(args.study_id)
    experiment = get_experiment_spec(args.study_id, args.exp_id)
    train_protocol = get_train_protocol(args.study_id, args.train_protocol_id)
    eval_protocols = [
        get_eval_protocol(args.study_id, protocol_id)
        for protocol_id in args.eval_protocol_ids
    ]

    out_dir = os.path.join("runs", args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "study": study.model_dump(mode="json", exclude={"experiments"}),
        "experiment": {
            "exp_id": args.exp_id,
            **experiment.model_dump(
                mode="json",
                exclude={"action_space", "reward_type"},
            ),
        },
        "train_protocol": train_protocol.model_dump(mode="json"),
        "eval_protocols": [
            protocol.model_dump(mode="json") for protocol in eval_protocols
        ],
        "args": args.to_dict(),
        "carlabev_run_config": to_carlabev_run_config(args).model_dump(mode="json"),
        "compatibility": {
            "legacy_env_aliases": args.legacy_aliases(),
            "legacy_experiment_aliases": {
                "action_space": experiment.action_space,
                "reward_type": experiment.reward_type,
            },
        },
    }
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"Saved: runs/{args.exp_name}/config.yaml")


def load_experiment():
    parser = tyro.extras.subcommand_type_from_defaults({"exp": ArgsCarlaBEV()})
    args = tyro.cli(parser)

    print(f"⚙ Selecting study ID = {args.study_id}")
    print(f"⚙ Selecting experiment ID = {args.exp_id}")

    args = apply_experiment_config(args, args.exp_id, study_id=args.study_id)
    validate_run_config(to_carlabev_run_config(args))
    save_run_config(args)
    return args


def run_experiment(args: ArgsCarlaBEV, trial=None, seed_idx: int = None) -> float:
    """Universal execution wrapper for both single standalone trains and Optuna searches."""
    import optuna
    import torch

    from CarlaBEV.envs import make_env

    from src.trainers import build_trainer
    from src.utils.logger import DRLogger

    study = get_study_config(args.study_id)
    experiment = get_experiment_spec(args.study_id, args.exp_id)

    if trial is None:
        print(
            f"🔗 Integrating standalone run into Optuna SQLite Database "
            f"(Study: {study.study_id}, Exp ID: {args.exp_id})"
        )
        db_path = get_study_db_path(args.study_id)
        storage_name = f"sqlite:///{db_path}"
        optuna_study = None

        for _ in range(20):
            try:
                optuna_study = optuna.create_study(
                    storage=storage_name,
                    load_if_exists=True,
                    direction="maximize",
                    study_name=get_study_name(args.study_id),
                )
                break
            except Exception as exc:
                print(
                    f"Study creation collision or lock detected: {exc}. "
                    f"Retrying in a few seconds..."
                )
                time.sleep(random.uniform(2, 6))

        if optuna_study is None:
            raise RuntimeError(
                "Failed to create or load the Optuna study after multiple attempts "
                "due to database locking."
            )

        enqueued_params = {
            "learning_rate": args.ppo.learning_rate,
            "gae_lambda": args.ppo.gae_lambda,
            "gamma": args.ppo.gamma,
            "num_steps": args.ppo.num_steps,
            "update_epochs": args.ppo.update_epochs,
            "num_minibatches": args.ppo.num_minibatches,
        }
        optuna_study.enqueue_trial(enqueued_params)

        def _manual_objective(t):
            return run_experiment(args, trial=t, seed_idx=seed_idx)

        optuna_study.optimize(_manual_objective, n_trials=1)
        return

    if seed_idx is not None:
        args.exp_name = f"{args.exp_name}_optuna_trial_{trial.number}_seed_{args.seed}"
    else:
        args.exp_name = f"{args.exp_name}_optuna_trial_{trial.number}"

    save_run_config(args)

    args.logging.db_path = get_study_db_path(args.study_id)
    args.logging.trial_number = trial.number

    trial.set_user_attr("study_id", study.study_id)
    trial.set_user_attr("study_name", study.optuna_study_name)
    trial.set_user_attr("study_metadata", study.metadata)
    trial.set_user_attr("base_exp_id", args.exp_id)
    trial.set_user_attr("train_protocol_id", args.train_protocol_id)
    trial.set_user_attr("eval_protocol_ids", list(args.eval_protocol_ids))
    trial.set_user_attr("action_mode", args.env.action_mode)
    trial.set_user_attr("traffic_enabled", args.env.traffic_enabled)
    trial.set_user_attr("input_type", args.env.input_type)
    trial.set_user_attr("semantic_mask_ch", args.env.semantic_mask_ch)
    trial.set_user_attr("temporal_fusion_mode", args.env.temporal_fusion_mode)
    trial.set_user_attr("fov_masked", args.env.fov_masked)
    trial.set_user_attr("ego_anchor_x_frac", args.env.ego_anchor_x_frac)
    trial.set_user_attr("ego_anchor_y_frac", args.env.ego_anchor_y_frac)
    trial.set_user_attr("fov_anchor", experiment.fov_anchor if 'experiment' in locals() else None)
    trial.set_user_attr("reward_mode", args.env.reward_mode)
    trial.set_user_attr(
        "curriculum",
        args.env.curriculum_mode if args.env.curriculum_enabled else "off",
    )

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    validate_run_config(to_carlabev_run_config(args))
    envs = make_env(to_carlabev_run_config(args))
    logger = DRLogger(config=args, stats_interval=100)
    logger.set_running()
    logger.msg(f"Environments - {args.env.env_id}:{args.num_envs} built.")

    try:
        trainer = build_trainer(args.algorithm)
        logger.msg(f"Trainer built for algorithm: {args.algorithm}")

        final_score = trainer(args, envs, logger, device, trial=trial)
        logger.mark_completed()
        return final_score
    except optuna.TrialPruned:
        logger.update_status(state="pruned", finished=True)
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.msg(f"Run failed: {type(exc).__name__}: {exc}")
        logger.mark_failed(exc, tb)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    load_experiment()
