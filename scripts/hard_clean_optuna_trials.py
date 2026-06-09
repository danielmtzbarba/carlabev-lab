from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sqlite3
import sys

from rich.console import Console
from rich.table import Table
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.experiment_loader import get_study_db_path, get_study_name


DEPENDENT_TRIAL_TABLES = [
    "trial_heartbeats",
    "trial_intermediate_values",
    "trial_params",
    "trial_system_attributes",
    "trial_user_attributes",
    "trial_values",
]


@dataclass
class Args:
    study_id: str | None = None
    db_path: str | None = None
    study_name: str | None = None
    delete_states: tuple[str, ...] = ("PRUNED", "FAIL", "WAITING", "RUNNING")
    backup: bool = True


def _resolve_db_and_study(args: Args) -> tuple[Path, str | None]:
    if args.study_id is not None:
        return Path(get_study_db_path(args.study_id)), get_study_name(args.study_id)
    if args.db_path is not None:
        return Path(args.db_path), args.study_name
    raise ValueError("Provide either --study-id or --db-path.")


def _backup_db(db_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_suffix(db_path.suffix + f".backup-{timestamp}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _fetch_study_id(conn: sqlite3.Connection, study_name: str | None) -> tuple[int | None, list[tuple[int, str]]]:
    rows = conn.execute(
        "SELECT study_id, study_name FROM studies ORDER BY study_id"
    ).fetchall()
    if study_name is None:
        if len(rows) == 1:
            return int(rows[0][0]), rows
        return None, rows

    for study_id, name in rows:
        if name == study_name:
            return int(study_id), rows
    raise KeyError(f"Study name {study_name!r} not found in DB.")


def main() -> None:
    args = tyro.cli(Args)
    console = Console()

    db_path, study_name = _resolve_db_and_study(args)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    if args.backup:
        backup_path = _backup_db(db_path)
        console.print(f"[green]Backup created:[/green] {backup_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        study_id, studies = _fetch_study_id(conn, study_name)

        if study_id is None:
            table = Table(title="Studies in DB", header_style="bold cyan")
            table.add_column("Study ID", justify="right")
            table.add_column("Study Name")
            for row_study_id, row_name in studies:
                table.add_row(str(row_study_id), row_name)
            console.print(table)
            raise ValueError(
                "DB contains multiple studies. Re-run with --study-name or use --study-id."
            )

        placeholders = ", ".join("?" for _ in args.delete_states)
        trial_rows = conn.execute(
            f"""
            SELECT trial_id, number, state
            FROM trials
            WHERE study_id = ?
              AND state IN ({placeholders})
            ORDER BY trial_id
            """,
            (study_id, *args.delete_states),
        ).fetchall()

        if not trial_rows:
            console.print("[yellow]No matching trials to delete.[/yellow]")
            return

        trial_ids = [int(row[0]) for row in trial_rows]
        preview = Table(
            title=f"Trials to Delete from {study_name or f'study_id={study_id}'}",
            header_style="bold red",
        )
        preview.add_column("Trial ID", justify="right")
        preview.add_column("Number", justify="right")
        preview.add_column("State")
        for trial_id, number, state in trial_rows:
            preview.add_row(str(trial_id), str(number), state)
        console.print(preview)

        qmarks = ", ".join("?" for _ in trial_ids)
        with conn:
            for table_name in DEPENDENT_TRIAL_TABLES:
                conn.execute(
                    f"DELETE FROM {table_name} WHERE trial_id IN ({qmarks})",
                    trial_ids,
                )
            conn.execute(
                f"DELETE FROM trials WHERE trial_id IN ({qmarks})",
                trial_ids,
            )

        console.print(
            f"[green]Deleted {len(trial_ids)} trials[/green] from "
            f"{study_name or f'study_id={study_id}'} in {db_path}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
