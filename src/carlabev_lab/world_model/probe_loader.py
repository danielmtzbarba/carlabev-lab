from __future__ import annotations

from rich.console import Console
from rich.table import Table
import tyro

from src.utils.common_logging import configure_logging
from src.world_model.probe_loader import WorldModelLoaderProbeConfig, probe_world_model_loader


def _render_overview(console: Console, summary) -> None:
    table = Table(title="World Model Loader Probe", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    rows = [
        ("Run Dir", summary.run_dir),
        ("Device", summary.device),
        ("Observation Shape", str(summary.obs_shape) if summary.obs_shape is not None else "-"),
        ("JSON Results", summary.json_path),
        ("CSV Results", summary.csv_path),
    ]
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(table)


def _render_results(console: Console, summary) -> None:
    table = Table(title="Loader Candidates", header_style="bold magenta")
    table.add_column("Chunk", justify="right")
    table.add_column("Batch", justify="right")
    table.add_column("Workers", justify="right")
    table.add_column("Pin")
    table.add_column("Persist")
    table.add_column("Prefetch")
    table.add_column("Status")
    table.add_column("Samples/s", justify="right")
    table.add_column("First Batch (s)", justify="right")
    table.add_column("Peak MB", justify="right")

    sorted_results = sorted(
        summary.results,
        key=lambda result: (
            result.status != "ok",
            -(result.samples_per_second or 0.0),
            result.first_batch_seconds or float("inf"),
        ),
    )
    for result in sorted_results:
        table.add_row(
            str(result.chunk_length),
            str(result.batch_size),
            str(result.num_workers),
            str(result.pin_memory),
            "-" if result.persistent_workers is None else str(result.persistent_workers),
            "-" if result.prefetch_factor is None else str(result.prefetch_factor),
            result.status,
            "-" if result.samples_per_second is None else f"{result.samples_per_second:.2f}",
            "-" if result.first_batch_seconds is None else f"{result.first_batch_seconds:.3f}",
            "-" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.1f}",
        )
    console.print(table)


def _render_recommendation(console: Console, summary) -> None:
    ok_results = [result for result in summary.results if result.status == "ok"]
    if not ok_results:
        console.print("[bold red]No successful loader candidate completed.[/bold red]")
        return
    best = max(
        ok_results,
        key=lambda result: (
            result.samples_per_second or 0.0,
            -(result.first_batch_seconds or float("inf")),
        ),
    )
    console.print(
        "Best loader candidate: "
        f"chunk_length={best.chunk_length} batch_size={best.batch_size} "
        f"num_workers={best.num_workers} pin_memory={best.pin_memory} "
        f"persistent_workers={best.persistent_workers} prefetch_factor={best.prefetch_factor} "
        f"samples/s={best.samples_per_second:.2f}"
    )


def main() -> None:
    configure_logging()
    cfg = tyro.cli(WorldModelLoaderProbeConfig)
    if not cfg.data.dataset_paths:
        raise SystemExit("Provide at least one dataset path via --data.dataset-paths.")
    summary = probe_world_model_loader(cfg, show_progress=False)
    console = Console()
    _render_overview(console, summary)
    _render_results(console, summary)
    _render_recommendation(console, summary)
    return None


if __name__ == "__main__":
    main()
