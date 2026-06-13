from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from src.config.base_config import ArgsCarlaBEV, to_carlabev_run_config
from src.config.experiment_loader import apply_experiment_config
from src.world_model.collector import collect_dataset


@dataclass
class CollectDatasetArgs:
    study_id: str = "PPO_NAVIGATION"
    exp_id: int = 26
    seed: int = 0
    num_envs: int = 14
    total_transitions: int = 10_000
    steps_per_shard: int = 2_048
    split: str = "train"
    dataset_name: str = "default"
    output_dir: str | None = None
    policy: str = "random"
    checkpoint_path: str | None = None
    device: str = "cpu"
    show_progress: bool = True


def main() -> None:
    parser = tyro.extras.subcommand_type_from_defaults({"exp": CollectDatasetArgs()})
    args = tyro.cli(parser)

    cfg = ArgsCarlaBEV(study_id=args.study_id, exp_id=args.exp_id, seed=args.seed)
    cfg.num_envs = args.num_envs
    cfg = apply_experiment_config(cfg, args.exp_id, study_id=args.study_id)
    cfg.num_envs = args.num_envs
    cfg.capture_video = False
    cfg.capture_every = 0
    cfg.video_output_dir = None
    cfg.video_episode_indices = None
    cfg.train_video_count = 0
    cfg.eval_video_count = 0
    cfg.final_eval_video_count = 0
    to_carlabev_run_config(cfg)

    summary = collect_dataset(
        cfg,
        total_transitions=args.total_transitions,
        steps_per_shard=args.steps_per_shard,
        split=args.split,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        policy=args.policy,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        show_progress=args.show_progress,
    )
    print(f"Saved world-model dataset to {Path(summary.output_dir)}", flush=True)
    print(
        f"Collected {summary.total_transitions} transitions across {summary.shard_count} shards.",
        flush=True,
    )
    return None


if __name__ == "__main__":
    main()
