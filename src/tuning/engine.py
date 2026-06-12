from __future__ import annotations

import copy
from typing import Any

import numpy as np
import optuna

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import get_study_db_path, run_experiment
from src.config.studies.models import TuningConfig, TuningStageConfig, TuningStageName
from src.tuning.optuna_utils import OptunaArgs


STAGE_SEQUENCE: tuple[TuningStageName, ...] = (
    "policy_dynamics",
    "rollout_geometry",
    "loss_regularization",
    "network_capacity",
)

LEGACY_STAGE_ALIASES: dict[str, TuningStageName] = {
    "1": "policy_dynamics",
    "2a": "rollout_geometry",
    "2b": "loss_regularization",
    "3": "network_capacity",
}

BASE_CHANNELS = [32, 64, 64]


def normalize_stage_name(stage: str) -> TuningStageName:
    normalized = LEGACY_STAGE_ALIASES.get(stage, stage)
    if normalized not in STAGE_SEQUENCE:
        available = ", ".join(STAGE_SEQUENCE)
        raise ValueError(f"Unknown tuning stage {stage!r}. Available stages: {available}")
    return normalized


def stage_sort_key(stage: str) -> int:
    normalized = LEGACY_STAGE_ALIASES.get(stage, stage)
    try:
        return STAGE_SEQUENCE.index(normalized)  # type: ignore[arg-type]
    except ValueError:
        return len(STAGE_SEQUENCE)


def trial_stage_name(trial: optuna.trial.FrozenTrial | optuna.Trial) -> str:
    raw = str(trial.user_attrs.get("tuning_stage") or trial.user_attrs.get("phase") or "unknown")
    return LEGACY_STAGE_ALIASES.get(raw, raw)


def stage_title(stage: str) -> str:
    titles = {
        "policy_dynamics": "Policy Dynamics",
        "rollout_geometry": "Rollout Geometry",
        "loss_regularization": "Loss Regularization",
        "network_capacity": "Network Capacity",
    }
    return titles.get(stage, stage)


def apply_tuning_overrides(args: ArgsCarlaBEV, overrides: dict[str, Any]) -> ArgsCarlaBEV:
    for key, value in overrides.items():
        target: Any = args
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return args


def sample_stage_params(trial: optuna.Trial, stage: TuningStageName) -> dict[str, Any]:
    if stage == "policy_dynamics":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "gamma": trial.suggest_float("gamma", 0.98, 0.9995),
        }
    if stage == "rollout_geometry":
        return {
            "num_steps": trial.suggest_categorical("num_steps", [128, 256, 512]),
            "update_epochs": trial.suggest_int("update_epochs", 3, 8),
            "num_minibatches": trial.suggest_categorical("num_minibatches", [2, 4, 8]),
        }
    if stage == "loss_regularization":
        return {
            "clip_coef_start": trial.suggest_float("clip_coef_start", 0.1, 0.25),
            "ent_coef_start": trial.suggest_float("ent_coef_start", 5e-4, 3e-2, log=True),
            "vf_coef_start": trial.suggest_float("vf_coef_start", 0.4, 0.9),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
            "ent_decay_factor": trial.suggest_float("ent_decay_factor", 0.1, 0.5),
            "vf_decay_factor": trial.suggest_float("vf_decay_factor", 0.5, 1.0),
            "clip_decay_factor": trial.suggest_float("clip_decay_factor", 0.4, 0.9),
        }
    if stage == "network_capacity":
        return {
            "channel_multiplier": trial.suggest_categorical("channel_multiplier", [0.5, 1.0, 1.5, 2.0]),
            "fc_size": trial.suggest_categorical("fc_size", [256, 512, 1024]),
        }
    raise ValueError(f"Unsupported tuning stage {stage!r}")


def runtime_overrides_for_stage(stage: TuningStageName, params: dict[str, Any]) -> dict[str, Any]:
    if stage == "policy_dynamics":
        return {
            "ppo.learning_rate": params["learning_rate"],
            "ppo.gae_lambda": params["gae_lambda"],
            "ppo.gamma": params["gamma"],
        }
    if stage == "rollout_geometry":
        return {
            "ppo.num_steps": params["num_steps"],
            "ppo.update_epochs": params["update_epochs"],
            "ppo.num_minibatches": params["num_minibatches"],
        }
    if stage == "loss_regularization":
        return {
            "ppo.clip_coef_start": params["clip_coef_start"],
            "ppo.ent_coef_start": params["ent_coef_start"],
            "ppo.vf_coef_start": params["vf_coef_start"],
            "ppo.max_grad_norm": params["max_grad_norm"],
            "ppo.ent_decay_factor": params["ent_decay_factor"],
            "ppo.vf_decay_factor": params["vf_decay_factor"],
            "ppo.clip_decay_factor": params["clip_decay_factor"],
        }
    if stage == "network_capacity":
        multiplier = float(params["channel_multiplier"])
        return {
            "ppo.channels": [int(channel * multiplier) for channel in BASE_CHANNELS],
            "ppo.fc_size": params["fc_size"],
        }
    raise ValueError(f"Unsupported tuning stage {stage!r}")


def resolved_stage_config(
    tuning_config: TuningConfig,
    stage: TuningStageName,
    cli_args: OptunaArgs,
) -> tuple[TuningConfig, TuningStageConfig]:
    base = tuning_config.stages[stage]
    stage_cfg = base.model_copy(
        update={
            "n_trials": cli_args.n_trials if cli_args.n_trials is not None else base.n_trials,
            "total_timesteps": (
                cli_args.total_timesteps
                if cli_args.total_timesteps is not None
                else base.total_timesteps
            ),
        }
    )
    tuning_cfg = tuning_config.model_copy(
        update={
            "num_seeds": cli_args.num_seeds if cli_args.num_seeds is not None else tuning_config.num_seeds,
            "eval_episodes": (
                cli_args.eval_episodes
                if cli_args.eval_episodes is not None
                else tuning_config.eval_episodes
            ),
            "eval_final_episodes": (
                cli_args.eval_final_episodes
                if cli_args.eval_final_episodes is not None
                else tuning_config.eval_final_episodes
            ),
        }
    )
    return tuning_cfg, stage_cfg


def completed_stage_trials(
    study: optuna.Study,
    stage: TuningStageName,
) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        if trial_stage_name(trial) == stage
    ]


def inherited_stage_overrides(
    study: optuna.Study,
    tuning_config: TuningConfig,
    stage_cfg: TuningStageConfig,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for source_stage in stage_cfg.inherits_from:
        source_cfg = tuning_config.stages[source_stage]
        stage_trials = completed_stage_trials(study, source_stage)
        if not stage_trials:
            raise RuntimeError(
                f"Cannot run stage {stage_cfg.stage_id!r} without completed trials "
                f"from dependency stage {source_stage!r}."
            )
        stage_trials.sort(key=lambda trial: float(trial.value or float("-inf")), reverse=True)
        top_trials = stage_trials[: source_cfg.selection_top_k]
        best_trial = top_trials[0]
        overrides.update(runtime_overrides_for_stage(source_stage, best_trial.params))
    return overrides


def run_stage_objective(
    trial: optuna.Trial,
    *,
    base_args: ArgsCarlaBEV,
    study_id: str,
    tuning_config: TuningConfig,
    stage_cfg: TuningStageConfig,
    inherited_overrides: dict[str, Any],
) -> float:
    sampled_params = sample_stage_params(trial, stage_cfg.stage_id)
    stage_overrides = runtime_overrides_for_stage(stage_cfg.stage_id, sampled_params)

    scores = []
    base_seed = base_args.seed
    trial_offset = trial.number * 10000

    trial.set_user_attr("tuning_stage", stage_cfg.stage_id)
    trial.set_user_attr("phase", stage_cfg.stage_id)

    for seed_idx in range(tuning_config.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + trial_offset + seed_idx

        apply_tuning_overrides(args, inherited_overrides)
        apply_tuning_overrides(args, stage_overrides)

        args.ppo.total_timesteps = stage_cfg.total_timesteps
        args.eval_episodes = tuning_config.eval_episodes
        args.eval_final_episodes = tuning_config.eval_final_episodes
        args.logging.db_path = get_study_db_path(study_id)
        args.logging.trial_number = trial.number
        args.save_model = stage_cfg.save_model
        args.capture_video = stage_cfg.capture_video

        score = run_experiment(args, trial=trial, seed_idx=seed_idx)
        scores.append(score)

    return float(np.mean(scores))
