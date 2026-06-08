from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import statistics
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


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    args = tyro.cli(Args)
    console = Console()

    storage_name = f"sqlite:///{get_study_db_path(args.study_id)}"
    study = optuna.load_study(
        study_name=get_study_name(args.study_id),
        storage=storage_name,
    )

    grouped: dict[int, list[optuna.trial.FrozenTrial]] = defaultdict(list)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        exp_id = trial.user_attrs.get("base_exp_id")
        if exp_id is None:
            continue
        grouped[int(exp_id)].append(trial)

    rows = []
    for exp_id, trials in grouped.items():
        scores = [float(trial.value) for trial in trials if trial.value is not None]
        successes = [
            float(trial.user_attrs["final_success_rate"])
            for trial in trials
            if "final_success_rate" in trial.user_attrs
        ]
        collisions = [
            float(trial.user_attrs["final_collision_rate"])
            for trial in trials
            if "final_collision_rate" in trial.user_attrs
        ]
        unfinished = [
            float(trial.user_attrs["final_unfinished_rate"])
            for trial in trials
            if "final_unfinished_rate" in trial.user_attrs
        ]
        mean_returns = [
            float(trial.user_attrs["final_mean_return"])
            for trial in trials
            if "final_mean_return" in trial.user_attrs
        ]

        rows.append(
            {
                "exp_id": exp_id,
                "n": len(trials),
                "score_mean": _safe_mean(scores),
                "score_std": _safe_std(scores),
                "success_mean": _safe_mean(successes),
                "collision_mean": _safe_mean(collisions),
                "unfinished_mean": _safe_mean(unfinished),
                "return_mean": _safe_mean(mean_returns),
                "return_std": _safe_std(mean_returns),
                "trial_numbers": [trial.number for trial in trials],
            }
        )

    rows.sort(
        key=lambda row: (
            row["score_mean"] if row["score_mean"] is not None else float("-inf"),
            -(row["collision_mean"] if row["collision_mean"] is not None else 1.0),
        ),
        reverse=True,
    )

    selected = rows[: args.top_k]

    table = Table(
        title=f"Top {len(selected)} Experiments by Seed-Averaged Score for {args.study_id}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right")
    table.add_column("Exp", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Score Mean", justify="right")
    table.add_column("Score Std", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Collision", justify="right")
    table.add_column("Unfinished", justify="right")
    table.add_column("Return Mean", justify="right")
    table.add_column("Return Std", justify="right")
    table.add_column("Trials", justify="left")

    if not selected:
        console.print(f"[yellow]No completed trials found for study_id={args.study_id}.[/yellow]")
        return

    for rank, row in enumerate(selected, start=1):
        table.add_row(
            str(rank),
            str(row["exp_id"]),
            str(row["n"]),
            _fmt(row["score_mean"], 4),
            _fmt(row["score_std"], 4),
            _fmt(row["success_mean"]),
            _fmt(row["collision_mean"]),
            _fmt(row["unfinished_mean"]),
            _fmt(row["return_mean"]),
            _fmt(row["return_std"]),
            ", ".join(str(trial_number) for trial_number in row["trial_numbers"]),
        )

    console.print(table)
    console.print_json(
        json.dumps(
            {
                "study_id": args.study_id,
                "top_k": args.top_k,
                "completed_experiment_count": len(rows),
                "displayed_exp_ids": [row["exp_id"] for row in selected],
            }
        )
    )


if __name__ == "__main__":
    main()
