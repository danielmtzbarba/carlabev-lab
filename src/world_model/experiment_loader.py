from __future__ import annotations

from dataclasses import dataclass

from CarlaBEV.config import validate_run_config

from src.config.base_config import ArgsCarlaBEV, to_carlabev_run_config
from src.config.experiment_loader import apply_experiment_config
from src.world_model.collector import default_output_dir
from src.world_model.config import WorldModelConfig


@dataclass(frozen=True)
class WorldModelStudyPreset:
    dataset_name: str = "default"
    split: str = "train"
    batch_size: int = 32
    chunk_length: int = 8
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    epochs: int = 10
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "bfloat16"
    timing_log_interval: int = 50
    include_metadata: bool = False
    cache_sequence_indices: bool = True


STUDY_WORLD_MODEL_PRESETS: dict[str, WorldModelStudyPreset] = {
    "PPO_NAVIGATION_DIFFICULTY": WorldModelStudyPreset(
        dataset_name="lewm-ppo-difficulty-hpc",
        split="train",
        batch_size=32,
        chunk_length=4,
        num_workers=0,
        pin_memory=True,
        persistent_workers=None,
        prefetch_factor=None,
        epochs=5,
        device="cuda",
        amp=True,
        amp_dtype="bfloat16",
        timing_log_interval=50,
        include_metadata=False,
        cache_sequence_indices=True,
    ),
}


def get_world_model_study_preset(study_id: str) -> WorldModelStudyPreset:
    return STUDY_WORLD_MODEL_PRESETS.get(study_id, WorldModelStudyPreset())


@dataclass
class WorldModelTrainExperimentArgs:
    study_id: str = "PPO_NAVIGATION_DIFFICULTY"
    exp_id: int = 1
    seed: int = 2
    dataset_name: str | None = None
    split: str | None = None
    dataset_path: str | None = None
    run_name: str | None = None
    device: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    chunk_length: int | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    amp: bool | None = None
    amp_dtype: str | None = None
    timing_log_interval: int | None = None
    sequence_cache_dir: str | None = None


def build_world_model_train_config_from_experiment(args: WorldModelTrainExperimentArgs) -> WorldModelConfig:
    preset = get_world_model_study_preset(args.study_id)

    cfg = ArgsCarlaBEV(study_id=args.study_id, exp_id=args.exp_id, seed=args.seed)
    cfg = apply_experiment_config(cfg, args.exp_id, study_id=args.study_id)
    validate_run_config(to_carlabev_run_config(cfg))

    dataset_name = args.dataset_name or preset.dataset_name
    split = args.split or preset.split
    resolved_dataset_path = args.dataset_path or str(default_output_dir(cfg, split=split, dataset_name=dataset_name))
    run_name = args.run_name or _default_run_name(
        study_id=args.study_id,
        exp_id=args.exp_id,
        seed=args.seed,
        dataset_name=dataset_name,
    )

    world_cfg = WorldModelConfig(run_name=run_name)
    world_cfg.data.dataset_paths = [resolved_dataset_path]
    world_cfg.data.batch_size = args.batch_size if args.batch_size is not None else preset.batch_size
    world_cfg.data.chunk_length = args.chunk_length if args.chunk_length is not None else preset.chunk_length
    world_cfg.data.num_workers = args.num_workers if args.num_workers is not None else preset.num_workers
    world_cfg.data.pin_memory = args.pin_memory if args.pin_memory is not None else preset.pin_memory
    world_cfg.data.persistent_workers = (
        args.persistent_workers if args.persistent_workers is not None else preset.persistent_workers
    )
    world_cfg.data.prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else preset.prefetch_factor
    world_cfg.data.include_metadata = preset.include_metadata
    world_cfg.data.cache_sequence_indices = preset.cache_sequence_indices
    world_cfg.data.sequence_cache_dir = args.sequence_cache_dir

    world_cfg.training.epochs = args.epochs if args.epochs is not None else preset.epochs
    world_cfg.training.device = args.device or preset.device
    world_cfg.training.amp = args.amp if args.amp is not None else preset.amp
    world_cfg.training.amp_dtype = args.amp_dtype or preset.amp_dtype
    world_cfg.training.timing_log_interval = (
        args.timing_log_interval if args.timing_log_interval is not None else preset.timing_log_interval
    )
    return world_cfg


def _default_run_name(*, study_id: str, exp_id: int, seed: int, dataset_name: str) -> str:
    safe_dataset = dataset_name.replace("/", "-").replace("_", "-")
    return f"wm-{study_id.lower()}-exp{exp_id}-seed{seed}-{safe_dataset}"
