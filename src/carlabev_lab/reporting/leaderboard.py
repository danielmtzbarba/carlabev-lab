from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import statistics

import optuna
from rich.console import Console
from rich.table import Table
import tyro

from src.config.experiment_loader import get_study_db_path, get_study_name


@dataclass
class Args:
    study_id: str = "PPO_NAVIGATION"
    top_k: int = 10


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
        normalized_scores = [
            float(trial.user_attrs["final_normalized_score"])
            for trial in trials
            if "final_normalized_score" in trial.user_attrs
        ]
        comforts = [
            float(trial.user_attrs["final_comfort_score"])
            for trial in trials
            if "final_comfort_score" in trial.user_attrs
        ]
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
        straight = [
            float(trial.user_attrs["final_straight_fraction"])
            for trial in trials
            if "final_straight_fraction" in trial.user_attrs
        ]
        left = [
            float(trial.user_attrs["final_left_turn_fraction"])
            for trial in trials
            if "final_left_turn_fraction" in trial.user_attrs
        ]
        right = [
            float(trial.user_attrs["final_right_turn_fraction"])
            for trial in trials
            if "final_right_turn_fraction" in trial.user_attrs
        ]
        comfort_violations = [
            float(trial.user_attrs["final_comfort_violation_rate"])
            for trial in trials
            if "final_comfort_violation_rate" in trial.user_attrs
        ]
        harsh_brakes = [
            float(trial.user_attrs["final_harsh_brake_rate"])
            for trial in trials
            if "final_harsh_brake_rate" in trial.user_attrs
        ]
        returns = [
            float(trial.user_attrs["final_mean_return"])
            for trial in trials
            if "final_mean_return" in trial.user_attrs
        ]

        rows.append(
            {
                "exp_id": exp_id,
                "n": len(trials),
                "normalized_score_mean": _safe_mean(normalized_scores),
                "normalized_score_std": _safe_std(normalized_scores),
                "comfort_mean": _safe_mean(comforts),
                "success_mean": _safe_mean(successes),
                "collision_mean": _safe_mean(collisions),
                "unfinished_mean": _safe_mean(unfinished),
                "straight_mean": _safe_mean(straight),
                "left_mean": _safe_mean(left),
                "right_mean": _safe_mean(right),
                "comfort_violation_mean": _safe_mean(comfort_violations),
                "harsh_brake_mean": _safe_mean(harsh_brakes),
                "return_mean": _safe_mean(returns),
                "trial_numbers": [trial.number for trial in trials],
            }
        )

    rows.sort(
        key=lambda row: (
            row["normalized_score_mean"] if row["normalized_score_mean"] is not None else float("-inf"),
            -(row["collision_mean"] if row["collision_mean"] is not None else 1.0),
        ),
        reverse=True,
    )
    selected = rows[: args.top_k]

    table = Table(
        title=f"Normalized Score Leaderboard for {args.study_id}",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Rank", justify="right")
    table.add_column("Exp", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Norm Mean", justify="right")
    table.add_column("Norm Std", justify="right")
    table.add_column("Comfort", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Collision", justify="right")
    table.add_column("Unfinished", justify="right")
    table.add_column("Straight", justify="right")
    table.add_column("Left", justify="right")
    table.add_column("Right", justify="right")
    table.add_column("Comfort Viol", justify="right")
    table.add_column("Harsh Brake", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Trials", justify="left")

    if not selected:
        console.print(f"[yellow]No completed trials found for study_id={args.study_id}.[/yellow]")
        return

    for rank, row in enumerate(selected, start=1):
        table.add_row(
            str(rank),
            str(row["exp_id"]),
            str(row["n"]),
            _fmt(row["normalized_score_mean"]),
            _fmt(row["normalized_score_std"]),
            _fmt(row["comfort_mean"]),
            _fmt(row["success_mean"]),
            _fmt(row["collision_mean"]),
            _fmt(row["unfinished_mean"]),
            _fmt(row["straight_mean"]),
            _fmt(row["left_mean"]),
            _fmt(row["right_mean"]),
            _fmt(row["comfort_violation_mean"]),
            _fmt(row["harsh_brake_mean"]),
            _fmt(row["return_mean"]),
            ", ".join(str(trial_number) for trial_number in row["trial_numbers"]),
        )

    console.print(table)
    console.print_json(
        json.dumps(
            {
                "study_id": args.study_id,
                "top_k": args.top_k,
                "displayed_exp_ids": [row["exp_id"] for row in selected],
            }
        )
    )


if __name__ == "__main__":
    main()
