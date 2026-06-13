from __future__ import annotations

from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
import tyro

from src.world_model.inspect import inspect_dataset_path


@dataclass
class InspectDatasetArgs:
    path: str
    shard_index: int = 0


def _render_summary(console: Console, payload: dict[str, object]) -> None:
    summary = payload.get("summary")
    dataset_dir = str(payload["dataset_dir"])
    shard_path = str(payload["inspected_shard_path"])

    table = Table(title="World Model Dataset", show_header=False, header_style="bold")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Dataset Dir", dataset_dir)
    table.add_row("Shard", shard_path)
    if isinstance(summary, dict):
        table.add_row("Study", str(summary.get("study_id")))
        table.add_row("Experiment", str(summary.get("exp_id")))
        table.add_row("Policy", str(summary.get("policy")))
        table.add_row("Split", str(summary.get("split")))
        table.add_row("Seed", str(summary.get("seed")))
        table.add_row("Shard Count", str(summary.get("shard_count")))
        table.add_row("Transitions", str(summary.get("total_transitions")))
    console.print(table)


def _render_arrays(console: Console, payload: dict[str, object]) -> None:
    arrays = payload["arrays"]
    table = Table(title="Shard Arrays", header_style="bold magenta")
    table.add_column("Name", style="bold")
    table.add_column("Shape", justify="right")
    table.add_column("DType", justify="right")
    table.add_column("Rows", justify="right")

    for name in sorted(arrays):
        meta = arrays[name]
        shape = meta["shape"]
        rows = shape[0] if shape else 1
        table.add_row(
            str(name),
            str(tuple(shape)),
            str(meta["dtype"]),
            str(rows),
        )
    console.print(table)


def main() -> None:
    args = tyro.cli(InspectDatasetArgs)
    payload = inspect_dataset_path(args.path, shard_index=args.shard_index)
    console = Console()
    _render_summary(console, payload)
    _render_arrays(console, payload)
    return None


if __name__ == "__main__":
    main()
