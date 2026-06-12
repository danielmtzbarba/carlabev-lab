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
  drl run train exp [ARGS...]
  drl run eval exp [ARGS...]
  drl tune run [ARGS...]
  drl tune analyze [ARGS...]
  drl results top-trials [ARGS...]
  drl results top-experiments [ARGS...]
  drl results leaderboard [ARGS...]
  drl results confirmation-summary [ARGS...]
  drl results plots [ARGS...]
  drl db trial-states [ARGS...]
  drl db inspect [ARGS...]
  drl db clean-stale [ARGS...]
  drl db delete-trials [ARGS...]
  drl diagnostics seed-scenes [ARGS...]
  drl diagnostics pruning [ARGS...]
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
    return _invoke_module_main("scripts.print_top_study_results", argv)


def run_results_top_experiments_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.print_top_experiments_by_seed_average", argv)


def run_results_leaderboard_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.print_normalized_study_leaderboard", argv)


def run_results_confirmation_summary_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.print_confirmation_experiment_summary", argv)


def run_results_plots_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.generate_study_result_plots", argv)


def run_db_trial_states_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.print_study_trial_states", argv)


def run_db_inspect_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.inspect_optuna_db", argv)


def run_db_clean_stale_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.clean_stale_trials", argv)


def run_db_delete_trials_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.hard_clean_optuna_trials", argv)


def run_diagnostics_seed_scenes_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("scripts.analyze_seed_scene_distribution", argv)


def run_diagnostics_pruning_command(argv: Sequence[str]) -> object:
    module = importlib.import_module("scripts.analyze_pruning")
    return module.check_median()


def _pop_command(argv: list[str], context: str) -> str:
    if not argv:
        raise SystemExit(f"Missing command after {context}.\n\n{USAGE}")
    return argv.pop(0)


def main(argv: Sequence[str] | None = None) -> object:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return None

    group = _pop_command(args, "drl")

    if group == "run":
        command = _pop_command(args, "drl run")
        if command == "train":
            return run_train_command(args)
        if command == "eval":
            return run_eval_command(args)

    if group == "tune":
        command = _pop_command(args, "drl tune")
        if command == "run":
            return run_tune_command(args)
        if command == "analyze":
            return run_tune_analysis_command(args)

    if group == "results":
        command = _pop_command(args, "drl results")
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

    if group == "db":
        command = _pop_command(args, "drl db")
        if command == "trial-states":
            return run_db_trial_states_command(args)
        if command == "inspect":
            return run_db_inspect_command(args)
        if command == "clean-stale":
            return run_db_clean_stale_command(args)
        if command == "delete-trials":
            return run_db_delete_trials_command(args)

    if group == "diagnostics":
        command = _pop_command(args, "drl diagnostics")
        if command == "seed-scenes":
            return run_diagnostics_seed_scenes_command(args)
        if command == "pruning":
            return run_diagnostics_pruning_command(args)

    raise SystemExit(f"Unknown command: {' '.join([group, *args])}\n\n{USAGE}")


if __name__ == "__main__":
    main()
