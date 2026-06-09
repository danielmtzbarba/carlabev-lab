from __future__ import annotations

from collections import Counter
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


def main() -> None:
    args = tyro.cli(Args)
    console = Console()

    storage_name = f"sqlite:///{get_study_db_path(args.study_id)}"
    study = optuna.load_study(
        study_name=get_study_name(args.study_id),
        storage=storage_name,
    )

    counts = Counter(trial.state.name for trial in study.trials)

    summary = Table(title=f"Trial States for {args.study_id}", header_style="bold cyan")
    summary.add_column("State")
    summary.add_column("Count", justify="right")
    for state in sorted(counts):
        summary.add_row(state, str(counts[state]))
    console.print(summary)

    details = Table(title="Trials", header_style="bold cyan")
    details.add_column("Trial", justify="right")
    details.add_column("State")
    details.add_column("Exp", justify="right")
    details.add_column("Seed", justify="right")
    details.add_column("Value", justify="right")

    for trial in study.trials:
        details.add_row(
            str(trial.number),
            trial.state.name,
            str(trial.user_attrs.get("base_exp_id", "-")),
            str(trial.user_attrs.get("seed", "-")),
            "-" if trial.value is None else f"{trial.value:.4f}",
        )
    console.print(details)

    console.print_json(
        json.dumps(
            {
                "study_id": args.study_id,
                "counts": dict(counts),
                "trial_count": len(study.trials),
            }
        )
    )


if __name__ == "__main__":
    main()
