import os
import random
import time
import traceback
import json
from pathlib import Path

import tyro
import yaml
from CarlaBEV.config import resolve_env_profiles, validate_run_config

from src.config.base_config import (
    ArgsCarlaBEV,
    LEGACY_ACTION_PROFILE_IDS,
    LEGACY_REWARD_PROFILE_IDS,
    to_carlabev_run_config,
)
from src.config.studies.models import ExperimentSpec, StudyConfig
from src.config.studies.registry import (
    get_eval_protocol,
    get_experiment_spec,
    get_study_config,
    get_train_protocol,
)
from src.utils.run_paths import RunPaths, build_run_id, build_run_label


def get_study_name(study_id: str) -> str:
    return get_study_config(study_id).optuna_study_name


def get_study_db_path(study_id: str) -> str:
    return get_study_config(study_id).db_path


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.") from exc


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
    env.action_profile_id = experiment.action_profile_id or LEGACY_ACTION_PROFILE_IDS[
        experiment.action_mode
    ]
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
    env.reward_profile_id = experiment.reward_profile_id or LEGACY_REWARD_PROFILE_IDS[
        experiment.reward_mode
    ]
    env.difficulty_id = experiment.difficulty_id

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

    run_paths = RunPaths(study_id=study.study_id, exp_id=exp_id, trial_number=None, seed=args.seed)
    args.run_label = run_paths.run_label
    args.run_id = build_run_id(study.study_id, exp_id, seed=args.seed)
    args.exp_name = args.run_id
    args.run_dir = str(run_paths.run_dir)

    return args


def save_run_config(args: ArgsCarlaBEV):
    study = get_study_config(args.study_id)
    experiment = get_experiment_spec(args.study_id, args.exp_id)
    train_protocol = get_train_protocol(args.study_id, args.train_protocol_id)
    eval_protocols = [
        get_eval_protocol(args.study_id, protocol_id)
        for protocol_id in args.eval_protocol_ids
    ]

    out_dir = args.run_dir
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
        "carlabev_env_profiles": resolve_env_profiles(to_carlabev_run_config(args).env),
        "selected_profiles": {
            "difficulty_id": args.env.difficulty_id,
            "action_profile_id": args.env.action_profile_id,
            "reward_profile_id": args.env.reward_profile_id,
        },
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
    latest_payload = {
        "study_id": args.study_id,
        "exp_id": args.exp_id,
        "run_label": args.run_label,
        "run_id": args.run_id,
        "run_dir": args.run_dir,
        "seed": args.seed,
        "trial_number": args.logging.trial_number,
    }
    latest_path = RunPaths(
        study_id=args.study_id,
        exp_id=args.exp_id,
        trial_number=args.logging.trial_number,
        seed=args.seed,
    ).latest_pointer_path
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(latest_payload, handle, indent=2, sort_keys=True)
    print(f"Saved: {os.path.join(out_dir, 'config.yaml')}", flush=True)


def write_startup_marker(
    args: ArgsCarlaBEV,
    *,
    stage: str,
    extra: dict | None = None,
) -> Path:
    experiment_root = RunPaths(
        study_id=args.study_id,
        exp_id=args.exp_id,
        trial_number=None,
        seed=args.seed,
    ).experiment_root
    experiment_root.mkdir(parents=True, exist_ok=True)
    marker_path = experiment_root / f"startup_seed_{args.seed}.json"
    payload = {
        "stage": stage,
        "study_id": args.study_id,
        "exp_id": args.exp_id,
        "seed": args.seed,
        "run_label": getattr(args, "run_label", None),
        "run_id": getattr(args, "run_id", None),
        "run_dir": getattr(args, "run_dir", None),
        "pid": os.getpid(),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload["extra"] = extra
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return marker_path


def load_experiment():
    parser = tyro.extras.subcommand_type_from_defaults({"exp": ArgsCarlaBEV()})
    args = tyro.cli(parser)

    print(f"⚙ Selecting study ID = {args.study_id}", flush=True)
    print(f"⚙ Selecting experiment ID = {args.exp_id}", flush=True)

    args = apply_experiment_config(args, args.exp_id, study_id=args.study_id)
    validate_run_config(to_carlabev_run_config(args))
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
        startup_marker = write_startup_marker(
            args,
            stage="manual_run_requested",
        )
        print(
            f"🔗 Integrating standalone run into Optuna SQLite Database "
            f"(Study: {study.study_id}, Exp ID: {args.exp_id})",
            flush=True,
        )
        print(f"📝 Startup marker written to {startup_marker}", flush=True)
        db_path = get_study_db_path(args.study_id)
        storage_name = f"sqlite:///{db_path}"
        sqlite_busy_timeout = _get_env_int("OPTUNA_SQLITE_BUSY_TIMEOUT_SECONDS", 120)
        max_wait_seconds = _get_env_int("OPTUNA_STUDY_ACQUIRE_MAX_WAIT_SECONDS", 900)
        retry_min_seconds = _get_env_int("OPTUNA_STUDY_ACQUIRE_RETRY_MIN_SECONDS", 5)
        retry_max_seconds = _get_env_int("OPTUNA_STUDY_ACQUIRE_RETRY_MAX_SECONDS", 15)
        if retry_min_seconds > retry_max_seconds:
            raise ValueError(
                "OPTUNA_STUDY_ACQUIRE_RETRY_MIN_SECONDS must be <= "
                "OPTUNA_STUDY_ACQUIRE_RETRY_MAX_SECONDS."
            )

        storage = optuna.storages.RDBStorage(
            url=storage_name,
            engine_kwargs={"connect_args": {"timeout": sqlite_busy_timeout}},
        )
        optuna_study = None
        deadline = time.monotonic() + max_wait_seconds
        attempt = 0

        while time.monotonic() < deadline:
            attempt += 1
            try:
                write_startup_marker(
                    args,
                    stage="optuna_create_study_attempt",
                    extra={
                        "attempt": attempt,
                        "storage_name": storage_name,
                        "sqlite_busy_timeout_seconds": sqlite_busy_timeout,
                        "max_wait_seconds": max_wait_seconds,
                    },
                )
                print(
                    f"🔄 Optuna study acquisition attempt {attempt} "
                    f"for study={study.study_id} exp_id={args.exp_id} seed={args.seed}",
                    flush=True,
                )
                optuna_study = optuna.create_study(
                    storage=storage,
                    load_if_exists=True,
                    direction="maximize",
                    study_name=get_study_name(args.study_id),
                    pruner=optuna.pruners.NopPruner(),
                )
                write_startup_marker(
                    args,
                    stage="optuna_study_ready",
                    extra={
                        "attempt": attempt,
                        "storage_name": storage_name,
                        "elapsed_seconds": round(max_wait_seconds - max(deadline - time.monotonic(), 0), 2),
                    },
                )
                break
            except Exception as exc:
                remaining_seconds = max(0.0, deadline - time.monotonic())
                sleep_seconds = min(
                    random.uniform(retry_min_seconds, retry_max_seconds),
                    remaining_seconds,
                )
                print(
                    f"Study creation collision or lock detected: {exc}. "
                    f"Retrying in {sleep_seconds:.1f}s "
                    f"(attempt={attempt}, remaining={remaining_seconds:.1f}s)...",
                    flush=True,
                )
                write_startup_marker(
                    args,
                    stage="optuna_create_study_retry",
                    extra={
                        "attempt": attempt,
                        "error": str(exc),
                        "sleep_seconds": round(sleep_seconds, 2),
                        "remaining_seconds": round(remaining_seconds, 2),
                    },
                )
                if sleep_seconds <= 0:
                    break
                time.sleep(sleep_seconds)

        if optuna_study is None:
            write_startup_marker(
                args,
                stage="optuna_study_acquire_failed",
                extra={
                    "attempts": attempt,
                    "max_wait_seconds": max_wait_seconds,
                    "storage_name": storage_name,
                },
            )
            raise RuntimeError(
                "Failed to create or load the Optuna study within the configured "
                "startup wait budget due to database locking/contention."
            )

        def _manual_objective(t):
            return run_experiment(args, trial=t, seed_idx=seed_idx)

        optuna_study.optimize(_manual_objective, n_trials=1)
        return

    run_paths = RunPaths(
        study_id=study.study_id,
        exp_id=args.exp_id,
        trial_number=trial.number,
        seed=args.seed,
    )
    run_paths.ensure_dirs()
    args.run_label = run_paths.run_label
    args.run_id = run_paths.run_id
    args.exp_name = args.run_id
    args.run_dir = str(run_paths.run_dir)

    args.logging.db_path = get_study_db_path(args.study_id)
    args.logging.trial_number = trial.number

    save_run_config(args)

    trial.set_user_attr("study_id", study.study_id)
    trial.set_user_attr("study_name", study.optuna_study_name)
    trial.set_user_attr("study_metadata", study.metadata)
    trial.set_user_attr("base_exp_id", args.exp_id)
    trial.set_user_attr("seed", args.seed)
    trial.set_user_attr("train_protocol_id", args.train_protocol_id)
    trial.set_user_attr("eval_protocol_ids", list(args.eval_protocol_ids))
    trial.set_user_attr("action_mode", args.env.action_mode)
    trial.set_user_attr("action_profile_id", args.env.action_profile_id)
    trial.set_user_attr("traffic_enabled", args.env.traffic_enabled)
    trial.set_user_attr("difficulty_id", args.env.difficulty_id)
    trial.set_user_attr("input_type", args.env.input_type)
    trial.set_user_attr("semantic_mask_ch", args.env.semantic_mask_ch)
    trial.set_user_attr("temporal_fusion_mode", args.env.temporal_fusion_mode)
    trial.set_user_attr("fov_masked", args.env.fov_masked)
    trial.set_user_attr("ego_anchor_x_frac", args.env.ego_anchor_x_frac)
    trial.set_user_attr("ego_anchor_y_frac", args.env.ego_anchor_y_frac)
    trial.set_user_attr("fov_anchor", experiment.fov_anchor if 'experiment' in locals() else None)
    trial.set_user_attr("reward_mode", args.env.reward_mode)
    trial.set_user_attr("reward_profile_id", args.env.reward_profile_id)
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
