from __future__ import annotations

from dataclasses import dataclass, field

from CarlaBEV.config import validate_run_config

from src.config.base_config import ArgsCarlaBEV, to_carlabev_run_config
from src.config.experiment_loader import apply_experiment_config
from src.world_model.collector import default_output_dir
from src.world_model.config import WorldModelConfig


@dataclass(frozen=True)
class WorldModelStudyPreset:
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
    split: str = "train"


@dataclass(frozen=True)
class WorldModelDatasetSourceSpec:
    study_id: str
    exp_id: int
    dataset_name: str
    split: str = "train"
    seed: int | None = None


@dataclass(frozen=True)
class WorldModelExperimentSpec:
    name: str
    dataset_sources: tuple[WorldModelDatasetSourceSpec, ...]
    notes: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldModelStudyConfig:
    study_id: str
    description: str
    preset: WorldModelStudyPreset
    experiments: dict[int, WorldModelExperimentSpec]
    metadata: dict[str, object] = field(default_factory=dict)


LEGACY_STUDY_WORLD_MODEL_PRESETS: dict[str, tuple[WorldModelStudyPreset, str]] = {
    "PPO_NAVIGATION_DIFFICULTY": (
        WorldModelStudyPreset(
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
            split="train",
        ),
        "lewm-ppo-difficulty-hpc",
    ),
}


WM_DATA_PHASE1 = WorldModelStudyConfig(
    study_id="WM_DATA_PHASE1",
    description=(
        "Phase 1 world-model dataset-composition study over random, PPO no-traffic, PPO easy, "
        "PPO medium, and mixed-dataset training."
    ),
    preset=WorldModelStudyPreset(
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
        split="train",
    ),
    metadata={
        "owner": "carlabev-lab",
        "kind": "world_model_dataset_ablation",
        "fixed_backbone": {
            "source_study": "PPO_NAVIGATION_DIFFICULTY",
            "semantic_mask_ch": "4-class",
            "temporal_fusion_mode": "stack",
            "fov_anchor": "center",
            "action_profile_id": "discrete9_v1",
        },
        "mixing": "concatenate_dataset_roots_evenly",
        "random_baseline_note": (
            "Random baseline uses the medium-difficulty fixed-backbone dataset family to keep "
            "observation contracts aligned with the PPO datasets."
        ),
    },
    experiments={
        1: WorldModelExperimentSpec(
            name="WM_DATA_RANDOM",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=3,
                    dataset_name="lewm-random-difficulty-hpc",
                ),
            ),
            notes="Random-policy baseline using the medium-difficulty fixed backbone.",
            tags=("random", "baseline"),
        ),
        2: WorldModelExperimentSpec(
            name="WM_DATA_PPO_NO_TRAFFIC",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=1,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
            ),
            notes="Pure PPO no-traffic dataset.",
            tags=("ppo", "no_traffic"),
        ),
        3: WorldModelExperimentSpec(
            name="WM_DATA_PPO_EASY",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=2,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
            ),
            notes="Pure PPO easy-difficulty dataset.",
            tags=("ppo", "easy"),
        ),
        4: WorldModelExperimentSpec(
            name="WM_DATA_PPO_EASY_RANDOM",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=2,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=3,
                    dataset_name="lewm-random-difficulty-hpc",
                ),
            ),
            notes="Easy PPO mixed with random-coverage medium dataset.",
            tags=("ppo", "easy", "random_mix"),
        ),
        5: WorldModelExperimentSpec(
            name="WM_DATA_PPO_MEDIUM",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=3,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
            ),
            notes="Pure PPO medium-difficulty dataset.",
            tags=("ppo", "medium"),
        ),
        6: WorldModelExperimentSpec(
            name="WM_DATA_PPO_MEDIUM_NO_TRAFFIC",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=3,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=1,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
            ),
            notes="Medium PPO mixed with cleaner no-traffic PPO dataset.",
            tags=("ppo", "medium", "no_traffic_mix"),
        ),
        7: WorldModelExperimentSpec(
            name="WM_DATA_PPO_MEDIUM_RANDOM",
            dataset_sources=(
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=3,
                    dataset_name="lewm-ppo-difficulty-hpc",
                ),
                WorldModelDatasetSourceSpec(
                    study_id="PPO_NAVIGATION_DIFFICULTY",
                    exp_id=3,
                    dataset_name="lewm-random-difficulty-hpc",
                ),
            ),
            notes="Medium PPO mixed with medium random dataset.",
            tags=("ppo", "medium", "random_mix"),
        ),
    },
)


WORLD_MODEL_STUDY_REGISTRY: dict[str, WorldModelStudyConfig] = {
    WM_DATA_PHASE1.study_id: WM_DATA_PHASE1,
}


def list_world_model_study_ids() -> list[str]:
    return sorted(WORLD_MODEL_STUDY_REGISTRY.keys())


def get_world_model_study_config(study_id: str) -> WorldModelStudyConfig:
    try:
        return WORLD_MODEL_STUDY_REGISTRY[study_id]
    except KeyError as exc:
        available = ", ".join(list_world_model_study_ids())
        raise KeyError(
            f"Unknown world-model study_id '{study_id}'. Available world-model studies: {available}"
        ) from exc


def get_world_model_experiment_spec(study_id: str, exp_id: int) -> WorldModelExperimentSpec:
    study = get_world_model_study_config(study_id)
    try:
        return study.experiments[exp_id]
    except KeyError as exc:
        available = ", ".join(str(key) for key in sorted(study.experiments))
        raise KeyError(
            f"Unknown world-model exp_id '{exp_id}' for study '{study_id}'. Available exp_ids: {available}"
        ) from exc


@dataclass
class WorldModelTrainExperimentArgs:
    study_id: str = "PPO_NAVIGATION_DIFFICULTY"
    exp_id: int = 1
    seed: int = 2
    dataset_name: str | None = None
    split: str | None = None
    dataset_path: str | None = None
    dataset_paths: list[str] = field(default_factory=list)
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
    if args.study_id in WORLD_MODEL_STUDY_REGISTRY:
        return _build_from_world_model_study(args)
    return _build_from_legacy_source_study(args)


def _build_from_legacy_source_study(args: WorldModelTrainExperimentArgs) -> WorldModelConfig:
    try:
        preset, default_dataset_name = LEGACY_STUDY_WORLD_MODEL_PRESETS[args.study_id]
    except KeyError:
        available = ", ".join(sorted(LEGACY_STUDY_WORLD_MODEL_PRESETS))
        raise KeyError(
            f"Unsupported world-model source study_id '{args.study_id}'. "
            f"Available source studies: {available}. "
            f"World-model studies: {', '.join(list_world_model_study_ids())}"
        ) from None

    cfg = _build_source_env_config(study_id=args.study_id, exp_id=args.exp_id, seed=args.seed)
    split = args.split or preset.split
    dataset_name = args.dataset_name or default_dataset_name
    if args.dataset_paths and args.dataset_path:
        raise ValueError("Use either dataset_path or dataset_paths, not both.")
    if args.dataset_paths:
        resolved_dataset_paths = list(args.dataset_paths)
    elif args.dataset_path:
        resolved_dataset_paths = [args.dataset_path]
    else:
        resolved_dataset_paths = [str(default_output_dir(cfg, split=split, dataset_name=dataset_name))]

    world_cfg = _apply_shared_overrides(
        args,
        preset=preset,
        resolved_dataset_paths=resolved_dataset_paths,
        run_name=args.run_name
        or _default_run_name(study_id=args.study_id, exp_id=args.exp_id, seed=args.seed, experiment_name=dataset_name),
    )
    return world_cfg


def _build_from_world_model_study(args: WorldModelTrainExperimentArgs) -> WorldModelConfig:
    study = get_world_model_study_config(args.study_id)
    experiment = get_world_model_experiment_spec(args.study_id, args.exp_id)
    split_override = args.split or study.preset.split
    if args.dataset_name is not None:
        raise ValueError(
            "`dataset_name` is not supported for world-model dataset studies because each experiment "
            "may reference multiple source dataset families. Use --dataset-paths to override roots explicitly."
        )
    if args.dataset_path and args.dataset_paths:
        raise ValueError("Use either dataset_path or dataset_paths, not both.")
    if args.dataset_path and len(experiment.dataset_sources) != 1:
        raise ValueError(
            "`dataset_path` can only override a single-source world-model experiment. "
            "Use --dataset-paths for mixed experiments."
        )
    if args.dataset_paths and len(args.dataset_paths) != len(experiment.dataset_sources):
        raise ValueError(
            f"`dataset_paths` must match the number of dataset sources for this experiment "
            f"({len(experiment.dataset_sources)})."
        )

    if args.dataset_paths:
        resolved_dataset_paths = list(args.dataset_paths)
    elif args.dataset_path:
        resolved_dataset_paths = [args.dataset_path]
    else:
        resolved_dataset_paths = [
            str(
                default_output_dir(
                    _build_source_env_config(
                        study_id=source.study_id,
                        exp_id=source.exp_id,
                        seed=source.seed if source.seed is not None else args.seed,
                    ),
                    split=split_override if args.split is not None else source.split,
                    dataset_name=source.dataset_name,
                )
            )
            for source in experiment.dataset_sources
        ]

    world_cfg = _apply_shared_overrides(
        args,
        preset=study.preset,
        resolved_dataset_paths=resolved_dataset_paths,
        run_name=args.run_name
        or _default_run_name(
            study_id=args.study_id,
            exp_id=args.exp_id,
            seed=args.seed,
            experiment_name=experiment.name,
        ),
    )
    return world_cfg


def _build_source_env_config(*, study_id: str, exp_id: int, seed: int) -> ArgsCarlaBEV:
    cfg = ArgsCarlaBEV(study_id=study_id, exp_id=exp_id, seed=seed)
    cfg = apply_experiment_config(cfg, exp_id, study_id=study_id)
    validate_run_config(to_carlabev_run_config(cfg))
    return cfg


def _apply_shared_overrides(
    args: WorldModelTrainExperimentArgs,
    *,
    preset: WorldModelStudyPreset,
    resolved_dataset_paths: list[str],
    run_name: str,
) -> WorldModelConfig:
    world_cfg = WorldModelConfig(run_name=run_name)
    world_cfg.data.dataset_paths = resolved_dataset_paths
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


def _default_run_name(*, study_id: str, exp_id: int, seed: int, experiment_name: str) -> str:
    safe_experiment = experiment_name.lower().replace("/", "-").replace("_", "-")
    return f"wm-{study_id.lower()}-exp{exp_id}-seed{seed}-{safe_experiment}"
