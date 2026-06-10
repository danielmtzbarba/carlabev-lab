from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import optuna
from rich.console import Console
from rich.table import Table
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.experiment_loader import get_study_db_path, get_study_name


@dataclass
class Args:
    study_id: str = "PPO_NAVIGATION"
    top_k: int = 5
    exp_id: int | None = None


def _fmt_float(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def _compact_params(params: dict[str, object], max_items: int = 4) -> str:
    items = list(params.items())[:max_items]
    rendered = [
        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in items
    ]
    if len(params) > max_items:
        rendered.append("...")
    return ", ".join(rendered) if rendered else "-"


def _trial_sort_score(trial: optuna.trial.FrozenTrial) -> float:
    attrs = trial.user_attrs
    if "final_normalized_score" in attrs:
        return float(attrs["final_normalized_score"])
    if trial.value is not None:
        return float(trial.value)
    return float("-inf")


def _scoped_completed_trials(study: optuna.Study, exp_id: int | None) -> list[optuna.trial.FrozenTrial]:
    trials = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if exp_id is not None and trial.user_attrs.get("base_exp_id") != exp_id:
            continue
        trials.append(trial)
    return sorted(trials, key=_trial_sort_score, reverse=True)


def main() -> None:
    args = tyro.cli(Args)
    console = Console()

    storage_name = f"sqlite:///{get_study_db_path(args.study_id)}"
    study = optuna.load_study(
        study_name=get_study_name(args.study_id),
        storage=storage_name,
    )

    completed_trials = _scoped_completed_trials(study, args.exp_id)
    selected_trials = completed_trials[: args.top_k]

    title = f"Top {len(selected_trials)} Results for {args.study_id}"
    if args.exp_id is not None:
        title += f" (exp_id={args.exp_id})"

    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Trial", justify="right")
    table.add_column("Norm Score", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Exp", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Phase", justify="center")
    table.add_column("Success", justify="right")
    table.add_column("Collision", justify="right")
    table.add_column("Unfinished", justify="right")
    table.add_column("Comfort", justify="right")
    table.add_column("Comfort Viol", justify="right")
    table.add_column("Harsh Brake", justify="right")
    table.add_column("Mean Return", justify="right")
    table.add_column("Params", justify="left")

    if not selected_trials:
        console.print(f"[yellow]No completed trials found for study_id={args.study_id}.[/yellow]")
        return

    for rank, trial in enumerate(selected_trials, start=1):
        attrs = trial.user_attrs
        table.add_row(
            str(rank),
            str(trial.number),
            _fmt_float(attrs.get("final_normalized_score"), 3),
            _fmt_float(trial.value, 4),
            str(attrs.get("base_exp_id", "-")),
            str(attrs.get("seed", "-")),
            str(attrs.get("phase", "-")).strip('"'),
            _fmt_float(attrs.get("final_success_rate")),
            _fmt_float(attrs.get("final_collision_rate")),
            _fmt_float(attrs.get("final_unfinished_rate")),
            _fmt_float(attrs.get("final_comfort_score")),
            _fmt_float(attrs.get("final_comfort_violation_rate")),
            _fmt_float(attrs.get("final_harsh_brake_rate")),
            _fmt_float(attrs.get("final_mean_return")),
            _compact_params(trial.params),
        )

    console.print(table)

    summary = {
        "study_id": args.study_id,
        "exp_id": args.exp_id,
        "top_k": args.top_k,
        "completed_trial_count": len(completed_trials),
        "displayed_trial_numbers": [trial.number for trial in selected_trials],
    }
    console.print_json(json.dumps(summary))


if __name__ == "__main__":
    main()
