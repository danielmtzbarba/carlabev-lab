from __future__ import annotations

import tyro

from src.utils.common_logging import configure_logging
from src.world_model.config import WorldModelConfig
from src.world_model.experiment_loader import (
    WorldModelTrainExperimentArgs,
    build_world_model_train_config_from_experiment,
)
from src.world_model.train import train_world_model


def main() -> None:
    configure_logging()
    argv = __import__("sys").argv[1:]
    if argv and argv[0] == "exp":
        args = tyro.cli(WorldModelTrainExperimentArgs, args=argv[1:])
        cfg = build_world_model_train_config_from_experiment(args)
    else:
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
