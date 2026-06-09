from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys

from rich.console import Console
from rich.table import Table
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.experiment_loader import get_study_db_path


@dataclass
class Args:
    db_path: str | None = None
    study_id: str | None = None


def _resolve_db_path(args: Args) -> Path:
    if args.study_id is not None:
        return Path(get_study_db_path(args.study_id))
    if args.db_path is not None:
        return Path(args.db_path)
    raise ValueError("Provide either --db-path or --study-id.")


def main() -> None:
    args = tyro.cli(Args)
    console = Console()
    db_path = _resolve_db_path(args)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        studies = conn.execute(
            "SELECT study_id, study_name FROM studies ORDER BY study_id"
        ).fetchall()

        studies_table = Table(title=f"Studies in {db_path}", header_style="bold cyan")
        studies_table.add_column("Study ID", justify="right")
        studies_table.add_column("Study Name")
        for row in studies:
            studies_table.add_row(str(row["study_id"]), row["study_name"])
        console.print(studies_table)

        state_rows = conn.execute(
            """
            SELECT s.study_name, t.state, COUNT(*) AS count
            FROM trials t
            JOIN studies s ON s.study_id = t.study_id
            GROUP BY s.study_name, t.state
            ORDER BY s.study_name, t.state
            """
        ).fetchall()
        counts_by_study: dict[str, dict[str, int]] = defaultdict(dict)
        all_states: set[str] = set()
        for row in state_rows:
            counts_by_study[row["study_name"]][row["state"]] = int(row["count"])
            all_states.add(row["state"])

        states = sorted(all_states)
        summary = Table(title="Trial States by Study", header_style="bold cyan")
        summary.add_column("Study Name")
        for state in states:
            summary.add_column(state, justify="right")
        summary.add_column("TOTAL", justify="right")

        for study in studies:
            study_name = study["study_name"]
            row_counts = counts_by_study.get(study_name, {})
            total = sum(row_counts.values())
            summary.add_row(
                study_name,
                *[str(row_counts.get(state, 0)) for state in states],
                str(total),
            )
        console.print(summary)

        detail_rows = conn.execute(
            """
            SELECT
                t.trial_id,
                t.number,
                t.state,
                t.datetime_start,
                t.datetime_complete,
                s.study_name
            FROM trials t
            JOIN studies s ON s.study_id = t.study_id
            ORDER BY s.study_name, t.number
            """
        ).fetchall()

        details = Table(title="Trial Listing", header_style="bold cyan")
        details.add_column("Study")
        details.add_column("Trial", justify="right")
        details.add_column("Trial ID", justify="right")
        details.add_column("State")
        details.add_column("Started")
        details.add_column("Completed")

        for row in detail_rows:
            details.add_row(
                row["study_name"],
                str(row["number"]),
                str(row["trial_id"]),
                row["state"],
                row["datetime_start"] or "-",
                row["datetime_complete"] or "-",
            )
        console.print(details)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
