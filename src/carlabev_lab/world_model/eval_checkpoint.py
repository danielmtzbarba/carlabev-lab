from __future__ import annotations

from dataclasses import dataclass, field

import tyro
from rich.console import Console
from rich.table import Table

from src.utils.common_logging import configure_logging
from src.world_model.evaluate import (
    WorldModelCheckpointEvalConfig,
    evaluate_world_model_checkpoint,
)


@dataclass
class EvalCheckpointArgs:
    checkpoint_path: str
    dataset_paths: list[str] = field(default_factory=list)
    device: str | None = None
    batch_size: int | None = None
    include_train_split: bool = False
    output_path: str | None = None


def _render(console: Console, result) -> None:
    table = Table(title="World Model Checkpoint Eval", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    rows = [
        ("Checkpoint", result.checkpoint_path),
        ("Output", result.output_path),
        ("Datasets", "\n".join(result.dataset_paths)),
        ("Sources", "\n".join(result.source_names)),
        ("Device", result.device),
        ("Chunk Length", str(result.chunk_length)),
        ("Batch Size", str(result.batch_size)),
        ("Val Loss", f"{result.val_metrics.loss:.4f}"),
        ("Val Pred Loss", f"{result.val_metrics.pred_loss:.4f}"),
        ("Val Reg Loss", f"{result.val_metrics.reg_loss:.4f}"),
        ("Val Fetch ms", f"{result.val_metrics.avg_fetch_ms:.2f}"),
        ("Val Transfer ms", f"{result.val_metrics.avg_transfer_ms:.2f}"),
        ("Val Step ms", f"{result.val_metrics.avg_step_ms:.2f}"),
    ]
    if result.train_metrics is not None:
        rows.extend(
            [
                ("Train Loss", f"{result.train_metrics.loss:.4f}"),
                ("Train Pred Loss", f"{result.train_metrics.pred_loss:.4f}"),
                ("Train Reg Loss", f"{result.train_metrics.reg_loss:.4f}"),
                ("Train Fetch ms", f"{result.train_metrics.avg_fetch_ms:.2f}"),
                ("Train Transfer ms", f"{result.train_metrics.avg_transfer_ms:.2f}"),
                ("Train Step ms", f"{result.train_metrics.avg_step_ms:.2f}"),
            ]
        )
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)


def main() -> None:
    configure_logging()
    args = tyro.cli(EvalCheckpointArgs)
    result = evaluate_world_model_checkpoint(
        WorldModelCheckpointEvalConfig(
            checkpoint_path=args.checkpoint_path,
            dataset_paths=args.dataset_paths or None,
            device=args.device,
            batch_size=args.batch_size,
            include_train_split=args.include_train_split,
            output_path=args.output_path,
        )
    )
    _render(Console(), result)


if __name__ == "__main__":
    main()
