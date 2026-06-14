from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Sequence

USAGE = """Usage:
  drl train exp [ARGS...]
  drl eval exp [ARGS...]
  drl tune run [ARGS...]
  drl tune analyze [ARGS...]
  drl results top-trials [ARGS...]
  drl results top-experiments [ARGS...]
  drl results leaderboard [ARGS...]
  drl results confirmation-summary [ARGS...]
  drl results plots [ARGS...]
  drl results medium-report [ARGS...]
  drl db trial-states [ARGS...]
  drl db inspect [ARGS...]
  drl db clean-stale [ARGS...]
  drl db delete-trials [ARGS...]
  drl diagnostics seed-scenes [ARGS...]
  drl diagnostics pruning [ARGS...]
  drl world-model collect exp [ARGS...]
  drl world-model stage [ARGS...]
  drl world-model benchmark [ARGS...]
  drl world-model inspect [ARGS...]
  drl world-model summary [ARGS...]
  drl world-model train [ARGS...]
  drl world-model validate [ARGS...]

Compatibility aliases remain available:
  drl run train exp [ARGS...]
  drl run eval exp [ARGS...]
"""


validate_run_config = None
evaluate_ppo = None
DRLogger = None
RunPaths = None
_EXPERIMENT_PARSER = None


def _import_attr(module_name: str, attr_name: str) -> object:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _get_experiment_parser() -> object:
    global _EXPERIMENT_PARSER
    if _EXPERIMENT_PARSER is None:
        tyro = importlib.import_module("tyro")
        args_cls = _import_attr("src.config.base_config", "ArgsCarlaBEV")
        _EXPERIMENT_PARSER = tyro.extras.subcommand_type_from_defaults(
            {"exp": args_cls()}
        )
    return _EXPERIMENT_PARSER


def _get_validate_run_config():
    return validate_run_config or _import_attr("CarlaBEV.config", "validate_run_config")


def _get_to_carlabev_run_config():
    return _import_attr("src.config.base_config", "to_carlabev_run_config")


def _get_apply_experiment_config():
    return _import_attr("src.config.experiment_loader", "apply_experiment_config")


def _get_run_experiment():
    return _import_attr("src.config.experiment_loader", "run_experiment")


def _get_evaluate_ppo():
    return evaluate_ppo or _import_attr("src.eval.eval_ppo", "evaluate_ppo")


def _get_dr_logger():
    return DRLogger or _import_attr("src.utils.logger", "DRLogger")


def _get_run_paths():
    return RunPaths or _import_attr("src.utils.run_paths", "RunPaths")


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


def _parse_experiment_args(argv: Sequence[str]):
    tyro = importlib.import_module("tyro")
    parser = _get_experiment_parser()
    apply_experiment_config = _get_apply_experiment_config()
    to_carlabev_run_config = _get_to_carlabev_run_config()
    validate = _get_validate_run_config()

    args = tyro.cli(parser, args=list(argv))
    print(f"⚙ Selecting study ID = {args.study_id}", flush=True)
    print(f"⚙ Selecting experiment ID = {args.exp_id}", flush=True)

    args = apply_experiment_config(args, args.exp_id, study_id=args.study_id)
    validate(to_carlabev_run_config(args))
    return args


def run_train_command(argv: Sequence[str]) -> object:
    cfg = _parse_experiment_args(argv)
    return _get_run_experiment()(cfg)


def run_eval_command(argv: Sequence[str]) -> dict[str, object]:
    evaluate = _get_evaluate_ppo()
    logger_cls = _get_dr_logger()
    run_paths_cls = _get_run_paths()
    cfg = _parse_experiment_args(argv)
    latest_pointer = run_paths_cls(
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
    payload = evaluate(
        cfg=cfg,
        model_path=model_path,
        num_episodes=1000,
        num_envs=14,
        render=False,
        device="cuda",
        file_name="final/manual_eval.npy",
    )
    logger = logger_cls(cfg)
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


def run_world_model_collect_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.collect", argv)


def run_world_model_stage_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.stage", argv)


def run_world_model_benchmark_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.benchmark", argv)


def run_world_model_inspect_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.inspect", argv)


def run_world_model_summary_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.summary", argv)


def run_world_model_train_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.train", argv)


def run_world_model_validate_command(argv: Sequence[str]) -> object:
    return _invoke_module_main("src.carlabev_lab.world_model.validate", argv)


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
    group = _pop_command(args, "drl")

    if group == "train":
        return run_train_command(args)
    if group == "eval":
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
        if command == "medium-report":
            return run_results_medium_report_command(args)

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

    if group == "world-model":
        command = _pop_command(args, "drl world-model")
        if command == "collect":
            return run_world_model_collect_command(args)
        if command == "stage":
            return run_world_model_stage_command(args)
        if command == "benchmark":
            return run_world_model_benchmark_command(args)
        if command == "inspect":
            return run_world_model_inspect_command(args)
        if command == "summary":
            return run_world_model_summary_command(args)
        if command == "train":
            return run_world_model_train_command(args)
        if command == "validate":
            return run_world_model_validate_command(args)

    raise SystemExit(f"Unknown command: {' '.join([group, *args])}\n\n{USAGE}")


if __name__ == "__main__":
    main()
