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
        ("Checkpoint", "checkpoint_path"),
        ("Source Run Dir", "source_run_dir"),
        ("Shard Count", "shard_count"),
        ("Transitions", "total_transitions"),
        ("Episodes", "total_episodes"),
        ("Mean Episode Length", "mean_episode_length"),
        ("Max Episode Length", "max_episode_length"),
        ("Mean Reward", "mean_reward"),
        ("Done Rate", "done_rate"),
        ("Terminated Rate", "terminated_rate"),
        ("Unique Routes (Episode)", "unique_routes_episode"),
        ("Unique Scenes (Episode)", "unique_scenes_episode"),
        ("Route Uniqueness Rate", "route_uniqueness_rate_episode"),
        ("Scene Uniqueness Rate", "scene_uniqueness_rate_episode"),
        ("Unique Routes (Transition)", "unique_routes_transition"),
        ("Unique Scenes (Transition)", "unique_scenes_transition"),
        ("Mean Transitions / Route", "mean_transitions_per_route"),
        ("Max Transitions / Route", "max_transitions_per_route"),
        ("Mean Transitions / Scene", "mean_transitions_per_scene"),
        ("Max Transitions / Scene", "max_transitions_per_scene"),
        ("Mean Straight Fraction", "mean_straight_fraction"),
        ("Mean Left Fraction", "mean_left_turn_fraction"),
        ("Mean Right Fraction", "mean_right_turn_fraction"),
    ):
        value = payload[key]
        if value is None:
            continue
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
