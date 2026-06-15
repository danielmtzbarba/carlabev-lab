from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
import tyro

from src.world_model.contracts import WorldModelSequenceConfig
from src.world_model.validate import validate_datasets


@dataclass
class ValidateDatasetArgs:
    paths: list[str] = field(default_factory=list)
    chunk_lengths: list[int] = field(default_factory=lambda: [1, 8, 16])
    stride: int = 1
    expected_num_actions: int = 9
    cache_sequence_indices: bool = True
    sequence_cache_dir: str | None = None


def _render_overview(console: Console, payload: dict[str, object]) -> None:
    table = Table(title="World Model Dataset Validation", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    for label, key in (
        ("Dataset Roots", "dataset_roots"),
        ("Source Names", "source_names"),
        ("Datasets", "total_datasets"),
        ("Shards", "total_shards"),
        ("Transitions", "total_transitions"),
        ("Episodes", "total_episodes"),
        ("Mean Episode Length", "mean_episode_length"),
        ("P50 Episode Length", "p50_episode_length"),
        ("P95 Episode Length", "p95_episode_length"),
        ("Max Episode Length", "max_episode_length"),
        ("Unique Routes (Episode)", "unique_routes_episode"),
        ("Unique Scenes (Episode)", "unique_scenes_episode"),
        ("Route Uniqueness Rate (Episode)", "route_uniqueness_rate_episode"),
        ("Scene Uniqueness Rate (Episode)", "scene_uniqueness_rate_episode"),
        ("Unique Routes (Transition)", "unique_routes_transition"),
        ("Unique Scenes (Transition)", "unique_scenes_transition"),
        ("Route Uniqueness Rate (Transition)", "route_uniqueness_rate_transition"),
        ("Scene Uniqueness Rate (Transition)", "scene_uniqueness_rate_transition"),
        ("Done Rate", "mean_done_rate"),
        ("Terminated Rate", "mean_terminated_rate"),
        ("Truncated Rate", "mean_truncated_rate"),
    ):
        value = payload[key]
        if isinstance(value, list):
            rendered = "\n".join(str(item) for item in value)
        elif isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = str(value)
        table.add_row(label, rendered)
    console.print(table)


def _render_counts(console: Console, title: str, payload: dict[str, int], *, count_label: str) -> None:
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Name", style="bold")
    table.add_column(count_label, justify="right")
    for name, count in payload.items():
        table.add_row(str(name), str(count))
    console.print(table)


def _render_windows(console: Console, payload: dict[int, int]) -> None:
    table = Table(title="Valid Sequence Windows", header_style="bold magenta")
    table.add_column("Chunk Length", style="bold")
    table.add_column("Count", justify="right")
    for chunk_length, count in sorted(payload.items()):
        table.add_row(str(chunk_length), str(count))
    console.print(table)


def _render_warnings(console: Console, warnings: list[str]) -> None:
    if not warnings:
        return
    table = Table(title="Warnings", header_style="bold yellow")
    table.add_column("Message")
    for message in warnings:
        table.add_row(message)
    console.print(table)


def main() -> None:
    args = tyro.cli(ValidateDatasetArgs)
    if not args.paths:
        raise SystemExit("Provide at least one dataset path via --paths.")
    cfg = WorldModelSequenceConfig(
        chunk_length=1,
        stride=args.stride,
        expected_num_actions=args.expected_num_actions,
    )
    report = validate_datasets(
        args.paths,
        cfg=cfg,
        chunk_lengths=tuple(args.chunk_lengths),
        cache_sequence_indices=args.cache_sequence_indices,
        sequence_cache_dir=args.sequence_cache_dir,
    )
    payload = report.model_dump()
    console = Console()
    _render_overview(console, payload)
    _render_counts(console, "Action Histogram", payload["action_histogram"], count_label="Count")
    _render_counts(
        console,
        "Dataset Transition Counts",
        payload["dataset_transition_counts"],
        count_label="Transitions",
    )
    _render_windows(console, payload["valid_windows"])
    _render_warnings(console, payload["warnings"])
    return None


if __name__ == "__main__":
    main()
