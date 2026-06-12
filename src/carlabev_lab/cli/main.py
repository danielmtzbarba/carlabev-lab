from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Sequence

import tyro
from CarlaBEV.config import validate_run_config

from src.config.base_config import ArgsCarlaBEV, to_carlabev_run_config
from src.config.experiment_loader import apply_experiment_config, run_experiment
from src.eval.eval_ppo import evaluate_ppo
from src.utils.logger import DRLogger
from src.utils.run_paths import RunPaths


EXPERIMENT_PARSER = tyro.extras.subcommand_type_from_defaults({"exp": ArgsCarlaBEV()})

USAGE = """Usage:
  carlabev-lab train exp [ARGS...]
  carlabev-lab eval exp [ARGS...]
  carlabev-lab tune run [ARGS...]
  carlabev-lab tune analyze [ARGS...]
  carlabev-lab results top-trials [ARGS...]
  carlabev-lab results top-experiments [ARGS...]
  carlabev-lab results leaderboard [ARGS...]
  carlabev-lab results confirmation-summary [ARGS...]
  carlabev-lab results plots [ARGS...]
  carlabev-lab results medium-report [ARGS...]
  carlabev-lab db trial-states [ARGS...]
  carlabev-lab db inspect [ARGS...]
  carlabev-lab db clean-stale [ARGS...]
  carlabev-lab db delete-trials [ARGS...]
  carlabev-lab diagnostics seed-scenes [ARGS...]
  carlabev-lab diagnostics pruning [ARGS...]

Legacy aliases remain available:
  drl run train exp [ARGS...]
  drl run eval exp [ARGS...]
"""


def _invoke_module_main(module_name: str, argv: Sequence[str]) -> object:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise ValueError(f"Module {module_name!r} does not expose a main() function.")

    old_argv = sys.argv
    sys.argv = [module_name, *argv]
    try:
        return module.main()
    finally:
        sys.argv = old_argv


def _parse_experiment_args(argv: Sequence[str]) -> ArgsCarlaBEV:
    args = tyro.cli(EXPERIMENT_PARSER, args=list(argv))
    print(f"⚙ Selecting study ID = {args.study_id}", flush=True)
    print(f"⚙ Selecting experiment ID = {args.exp_id}", flush=True)

    args = apply_experiment_config(args, args.exp_id, study_id=args.study_id)
    validate_run_config(to_carlabev_run_config(args))
    return args


def run_train_command(argv: Sequence[str]) -> object:
    cfg = _parse_experiment_args(argv)
    return run_experiment(cfg)


def run_eval_command(argv: Sequence[str]) -> dict[str, object]:
    cfg = _parse_experiment_args(argv)
    latest_pointer = RunPaths(
        study_id=cfg.study_id,
        exp_id=cfg.exp_id,
        trial_number=None,
        seed=cfg.seed,
    ).latest_pointer_path
    if latest_pointer.exists():
        with open(latest_pointer, "r", encoding="utf-8") as handle:
            latest = json.load(handle)
        cfg.run_dir = latest["run_dir"]

    model_path = os.path.join(cfg.run_dir, "checkpoints", "ppo_final.pt")
    payload = evaluate_ppo(
        cfg=cfg,
        model_path=model_path,
        num_episodes=1000,
        num_envs=14,
        render=False,
        device="cuda",
        file_name="final/manual_eval.npy",
    )
    logger = DRLogger(cfg)
    try:
        logger.log_evaluation(payload["aggregate"], 0)
    finally:
        logger.close()
    return payload


def run_tune_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.tuning.optuna_tuner", argv)


def run_tune_analysis_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.tuning.optuna_analysis", argv)


def run_results_top_trials_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.reporting.top_trials", argv)


def run_results_top_experiments_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.reporting.top_experiments", argv)


def run_results_leaderboard_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.reporting.leaderboard", argv)


def run_results_confirmation_summary_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.reporting.confirmation_summary", argv)


def run_results_plots_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.reporting.study_plots", argv)


def run_results_medium_report_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.reporting.medium_report", argv)


def run_db_trial_states_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.db.trial_states", argv)


def run_db_inspect_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.db.inspect", argv)


def run_db_clean_stale_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.db.clean_stale", argv)


def run_db_delete_trials_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.db.delete_trials", argv)


def run_diagnostics_seed_scenes_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.diagnostics.seed_scenes", argv)


def run_diagnostics_pruning_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.diagnostics.pruning", argv)


def _pop_command(argv: list[str], context: str) -> str:
    if not argv:
        raise SystemExit(f"Missing command after {context}.\n\n{USAGE}")
    return argv.pop(0)


def _normalize_legacy_args(argv: list[str]) -> list[str]:
    if len(argv) >= 2 and argv[0] == "run" and argv[1] in {"train", "eval"}:
        return argv[1:]
    return argv


def main(argv: Sequence[str] | None = None) -> object:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return None

    args = _normalize_legacy_args(args)
    group = _pop_command(args, "carlabev-lab")

    if group == "train":
        return run_train_command(args)
    if group == "eval":
        return run_eval_command(args)

    if group == "tune":
        command = _pop_command(args, "carlabev-lab tune")
        if command == "run":
            return run_tune_command(args)
        if command == "analyze":
            return run_tune_analysis_command(args)

    if group == "results":
        command = _pop_command(args, "carlabev-lab results")
        if command == "top-trials":
            return run_results_top_trials_command(args)
        if command == "top-experiments":
            return run_results_top_experiments_command(args)
        if command == "leaderboard":
            return run_results_leaderboard_command(args)
        if command == "confirmation-summary":
            return run_results_confirmation_summary_command(args)
        if command == "plots":
            return run_results_plots_command(args)
        if command == "medium-report":
            return run_results_medium_report_command(args)

    if group == "db":
        command = _pop_command(args, "carlabev-lab db")
        if command == "trial-states":
            return run_db_trial_states_command(args)
        if command == "inspect":
            return run_db_inspect_command(args)
        if command == "clean-stale":
            return run_db_clean_stale_command(args)
        if command == "delete-trials":
            return run_db_delete_trials_command(args)

    if group == "diagnostics":
        command = _pop_command(args, "carlabev-lab diagnostics")
        if command == "seed-scenes":
            return run_diagnostics_seed_scenes_command(args)
        if command == "pruning":
            return run_diagnostics_pruning_command(args)

    raise SystemExit(f"Unknown command: {' '.join([group, *args])}\n\n{USAGE}")
