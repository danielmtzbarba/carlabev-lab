from __future__ import annotations

from rich.console import Console
from rich.table import Table
import tyro

from src.utils.common_logging import configure_logging
from src.world_model.benchmark import WorldModelBenchmarkConfig, benchmark_world_model


def _render_overview(console: Console, summary) -> None:
    table = Table(title="World Model Benchmark", show_header=False)
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
    table = Table(title="Benchmark Sweep", header_style="bold magenta")
    table.add_column("Chunk", justify="right")
    table.add_column("Batch", justify="right")
    table.add_column("Status")
    table.add_column("Samples/s", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Peak MB", justify="right")
    table.add_column("Last Loss", justify="right")
    table.add_column("Measured Batches", justify="right")

    sorted_results = sorted(
        summary.results,
        key=lambda result: (
            result.status != "ok",
            -(result.samples_per_second or 0.0),
            -(result.tokens_per_second or 0.0),
        ),
    )
    for result in sorted_results:
        table.add_row(
            str(result.chunk_length),
            str(result.batch_size),
            result.status,
            "-" if result.samples_per_second is None else f"{result.samples_per_second:.2f}",
            "-" if result.tokens_per_second is None else f"{result.tokens_per_second:.2f}",
            "-" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.1f}",
            "-" if result.last_loss is None else f"{result.last_loss:.4f}",
            str(result.measured_batches),
        )
    console.print(table)


def _render_recommendation(console: Console, summary) -> None:
    ok_results = [result for result in summary.results if result.status == "ok"]
    if not ok_results:
        console.print("[bold red]No successful benchmark configuration completed.[/bold red]")
        return
    best = max(
        ok_results,
        key=lambda result: (
            result.tokens_per_second or 0.0,
            result.samples_per_second or 0.0,
            result.batch_size,
            result.chunk_length,
        ),
    )
    console.print(
        "Best throughput candidate: "
        f"chunk_length={best.chunk_length} batch_size={best.batch_size} "
        f"tokens/s={best.tokens_per_second:.2f} peak_mb={best.peak_memory_mb or 0.0:.1f}"
    )


def main() -> None:
    configure_logging()
    cfg = tyro.cli(WorldModelBenchmarkConfig)
    if not cfg.data.dataset_paths:
        raise SystemExit("Provide at least one dataset path via --data.dataset-paths.")
    summary = benchmark_world_model(cfg)
    console = Console()
    _render_overview(console, summary)
    _render_results(console, summary)
    _render_recommendation(console, summary)
    return None


if __name__ == "__main__":
    main()
