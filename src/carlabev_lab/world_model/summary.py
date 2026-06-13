from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
import tyro

from src.world_model.summary import summarize_dataset


@dataclass
class SummaryDatasetArgs:
    path: str


def _render_overview(console: Console, payload: dict[str, object]) -> None:
    table = Table(title="World Model Dataset Summary", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    for label, key in (
        ("Dataset Dir", "dataset_dir"),
        ("Study", "study_id"),
        ("Experiment", "exp_id"),
        ("Policy", "policy"),
        ("Split", "split"),
        ("Seed", "seed"),
        ("Shard Count", "shard_count"),
        ("Transitions", "total_transitions"),
        ("Episodes", "total_episodes"),
        ("Mean Episode Length", "mean_episode_length"),
        ("Max Episode Length", "max_episode_length"),
        ("Mean Reward", "mean_reward"),
        ("Done Rate", "done_rate"),
        ("Terminated Rate", "terminated_rate"),
        ("Unique Routes", "unique_routes"),
        ("Unique Scenes", "unique_scenes"),
        ("Route Repeat Ratio", "route_repeat_ratio"),
        ("Scene Repeat Ratio", "scene_repeat_ratio"),
        ("Mean Straight Fraction", "mean_straight_fraction"),
        ("Mean Left Fraction", "mean_left_turn_fraction"),
        ("Mean Right Fraction", "mean_right_turn_fraction"),
    ):
        value = payload[key]
        if isinstance(value, float):
            if "rate" in key or "ratio" in key or "fraction" in key:
                rendered = f"{value:.3f}"
            else:
                rendered = f"{value:.3f}"
        else:
            rendered = str(value)
        table.add_row(label, rendered)
    console.print(table)


def _render_actions(console: Console, payload: dict[str, object]) -> None:
    histogram = payload["action_histogram"]
    table = Table(title="Action Histogram", header_style="bold magenta")
    table.add_column("Action", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Share", justify="right")
    total = int(payload["total_transitions"])
    for action, count in histogram.items():
        share = (int(count) / total) if total else 0.0
        table.add_row(str(action), str(count), f"{share:.3f}")
    console.print(table)


def main() -> None:
    args = tyro.cli(SummaryDatasetArgs)
    payload = summarize_dataset(args.path)
    console = Console()
    _render_overview(console, payload)
    _render_actions(console, payload)
    return None


if __name__ == "__main__":
    main()
