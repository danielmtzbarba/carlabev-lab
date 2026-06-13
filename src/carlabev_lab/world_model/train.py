from __future__ import annotations

import tyro

from src.world_model.config import WorldModelConfig
from src.world_model.train import train_world_model


def main() -> None:
    cfg = tyro.cli(WorldModelConfig)
    if not cfg.data.dataset_paths:
        raise SystemExit("Provide at least one dataset path via --data.dataset-paths.")
    result = train_world_model(cfg)
    print(f"Saved world-model run to {result.run_dir}", flush=True)
    print(f"Best checkpoint: {result.best_checkpoint_path}", flush=True)
    print(
        f"Final losses: train={result.final_train_loss:.6f} val={result.final_val_loss:.6f}",
        flush=True,
    )
    return None


if __name__ == "__main__":
    main()
