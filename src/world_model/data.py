from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.common_logging import get_logger
from src.utils.storage_paths import resolve_artifact_path
from src.world_model.config import WorldModelDataConfig
from src.world_model.contracts import DatasetRootConfig, DatasetSummaryModel, WorldModelSequenceConfig

REQUIRED_ARRAYS = (
    "obs",
    "actions",
    "rewards",
    "dones",
    "terminated",
    "truncated",
    "next_obs",
    "env_index",
    "episode_id",
    "step_in_episode",
    "protocol_id",
    "reset_seed",
    "route_signature",
    "scene_signature",
    "straight_fraction",
    "left_turn_fraction",
    "right_turn_fraction",
)

LOGGER = get_logger("world_model.data")


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: int
    root_path: Path
    source_name: str
    summary: DatasetSummaryModel


@dataclass(frozen=True)
class ShardRecord:
    dataset_id: int
    source_name: str
    shard_index: int
    shard_path: Path
    row_count: int


@dataclass(frozen=True)
class TransitionRef:
    dataset_id: int
    source_name: str
    shard_path: Path
    shard_index: int
    row_index: int
    episode_key: tuple[int, int, int]
    step_in_episode: int
    route_signature: str
    scene_signature: str
    done: bool
    terminated: bool
    truncated: bool
    action_value: int


@dataclass(frozen=True)
class SequenceRef:
    source_name: str
    episode_key: tuple[int, int, int]
    transitions: tuple[TransitionRef, ...]


@dataclass(frozen=True)
class IndexedDataset:
    sources: tuple[DatasetSource, ...]
    shards: tuple[ShardRecord, ...]
    transitions: tuple[TransitionRef, ...]
    episodes: dict[tuple[int, int, int], tuple[TransitionRef, ...]]


@dataclass(frozen=True)
class WorldModelDataArtifacts:
    train_dataset: Dataset
    val_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    indexed: IndexedDataset
    obs_shape: tuple[int, int, int]
    num_actions: int


def _summary_path(dataset_dir: Path) -> Path:
    return dataset_dir / "summary.json"


def _coerce_dataset_dir(path: str | Path) -> Path:
    dataset_dir = resolve_artifact_path(path)
    if dataset_dir.is_file():
        dataset_dir = dataset_dir.parent
    return dataset_dir


def load_dataset_summary(path: str | Path) -> DatasetSummaryModel:
    dataset_dir = _coerce_dataset_dir(path)
    summary_file = _summary_path(dataset_dir)
    if not summary_file.exists():
        raise FileNotFoundError(f"Expected summary.json in dataset directory: {dataset_dir}")
    with open(summary_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DatasetSummaryModel.model_validate(payload)


def _dataset_source_name(cfg: DatasetRootConfig, summary: DatasetSummaryModel) -> str:
    if cfg.source_name is not None:
        return cfg.source_name
    return f"{summary.policy}:{summary.study_id}:exp_{summary.exp_id}:seed_{summary.seed}"


def _validate_shard_arrays(data: dict[str, np.ndarray], shard_path: Path) -> int:
    missing = [name for name in REQUIRED_ARRAYS if name not in data]
    if missing:
        raise ValueError(f"Shard {shard_path} is missing required arrays: {', '.join(missing)}")
    row_count = int(data["obs"].shape[0])
    if row_count <= 0:
        raise ValueError(f"Shard {shard_path} is empty.")
    if tuple(data["obs"].shape) != tuple(data["next_obs"].shape):
        raise ValueError(
            f"Shard {shard_path} has mismatched obs/next_obs shapes: "
            f"{data['obs'].shape} vs {data['next_obs'].shape}"
        )
    for name in REQUIRED_ARRAYS:
        arr = data[name]
        if arr.shape[0] != row_count:
            raise ValueError(
                f"Shard {shard_path} has inconsistent row count for {name}: "
                f"{arr.shape[0]} vs {row_count}"
            )
    return row_count


def _load_shard_arrays(shard_path: Path) -> dict[str, np.ndarray]:
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    _validate_shard_arrays(arrays, shard_path)
    return arrays


def build_index(dataset_roots: list[str | Path | DatasetRootConfig]) -> IndexedDataset:
    LOGGER.info("Building world-model dataset index for %d dataset root(s)", len(dataset_roots))
    sources: list[DatasetSource] = []
    shard_records: list[ShardRecord] = []
    transitions: list[TransitionRef] = []
    episodes: dict[tuple[int, int, int], list[TransitionRef]] = defaultdict(list)

    normalized_roots: list[DatasetRootConfig] = []
    for root in dataset_roots:
        if isinstance(root, DatasetRootConfig):
            normalized_roots.append(root)
        else:
            normalized_roots.append(DatasetRootConfig(path=str(root)))

    for dataset_id, root_cfg in enumerate(normalized_roots):
        dataset_dir = _coerce_dataset_dir(root_cfg.path)
        summary = load_dataset_summary(dataset_dir)
        source_name = _dataset_source_name(root_cfg, summary)
        LOGGER.info(
            "Loading dataset root %s as source=%s transitions=%d shards=%d",
            dataset_dir,
            source_name,
            summary.total_transitions,
            summary.shard_count,
        )
        sources.append(
            DatasetSource(
                dataset_id=dataset_id,
                root_path=dataset_dir,
                source_name=source_name,
                summary=summary,
            )
        )

        for shard_meta in summary.shards:
            shard_path = resolve_artifact_path(shard_meta.path)
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard listed in summary does not exist: {shard_meta.path}")
            arrays = _load_shard_arrays(shard_path)
            row_count = int(arrays["obs"].shape[0])
            if row_count != shard_meta.transitions:
                raise ValueError(
                    f"Shard transition mismatch for {shard_path}: "
                    f"summary={shard_meta.transitions} actual={row_count}"
                )
            shard_records.append(
                ShardRecord(
                    dataset_id=dataset_id,
                    source_name=source_name,
                    shard_index=shard_meta.shard_index,
                    shard_path=shard_path,
                    row_count=row_count,
                )
            )
            if np.issubdtype(arrays["actions"].dtype, np.integer):
                action_values = arrays["actions"].reshape(row_count, -1)[:, 0].astype(np.int64, copy=False)
            else:
                raise ValueError(f"Phase 1 expects discrete integer actions, got {arrays['actions'].dtype}")

            for row_index in range(row_count):
                env_index = int(arrays["env_index"][row_index])
                episode_id = int(arrays["episode_id"][row_index])
                step_in_episode = int(arrays["step_in_episode"][row_index])
                episode_key = (dataset_id, env_index, episode_id)
                ref = TransitionRef(
                    dataset_id=dataset_id,
                    source_name=source_name,
                    shard_path=shard_path,
                    shard_index=shard_meta.shard_index,
                    row_index=row_index,
                    episode_key=episode_key,
                    step_in_episode=step_in_episode,
                    route_signature=str(arrays["route_signature"][row_index]),
                    scene_signature=str(arrays["scene_signature"][row_index]),
                    done=bool(arrays["dones"][row_index]),
                    terminated=bool(arrays["terminated"][row_index]),
                    truncated=bool(arrays["truncated"][row_index]),
                    action_value=int(action_values[row_index]),
                )
                transitions.append(ref)
                episodes[episode_key].append(ref)

    ordered_episodes: dict[tuple[int, int, int], tuple[TransitionRef, ...]] = {}
    for episode_key, refs in episodes.items():
        ordered = tuple(sorted(refs, key=lambda ref: (ref.step_in_episode, ref.shard_index, ref.row_index)))
        ordered_episodes[episode_key] = ordered
    LOGGER.info(
        "Indexed %d transition(s) across %d episode(s) and %d shard(s)",
        len(transitions),
        len(ordered_episodes),
        len(shard_records),
    )
    return IndexedDataset(
        sources=tuple(sources),
        shards=tuple(shard_records),
        transitions=tuple(transitions),
        episodes=ordered_episodes,
    )


class ShardArrayCache:
    def __init__(self, max_items: int = 2):
        self.max_items = max_items
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def get(self, shard_path: Path) -> dict[str, np.ndarray]:
        if shard_path in self._cache:
            arrays = self._cache.pop(shard_path)
            self._cache[shard_path] = arrays
            return arrays
        arrays = _load_shard_arrays(shard_path)
        self._cache[shard_path] = arrays
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return arrays


def _row_to_transition_sample(arrays: dict[str, np.ndarray], ref: TransitionRef) -> dict[str, Any]:
    idx = ref.row_index
    return {
        "obs": torch.from_numpy(np.asarray(arrays["obs"][idx], dtype=np.float32)),
        "action": torch.tensor(int(ref.action_value), dtype=torch.long),
        "next_obs": torch.from_numpy(np.asarray(arrays["next_obs"][idx], dtype=np.float32)),
        "reward": torch.tensor(float(arrays["rewards"][idx]), dtype=torch.float32),
        "done": torch.tensor(bool(arrays["dones"][idx]), dtype=torch.bool),
        "terminated": torch.tensor(bool(arrays["terminated"][idx]), dtype=torch.bool),
        "truncated": torch.tensor(bool(arrays["truncated"][idx]), dtype=torch.bool),
        "metadata": {
            "episode_key": ref.episode_key,
            "step_in_episode": ref.step_in_episode,
            "route_signature": ref.route_signature,
            "scene_signature": ref.scene_signature,
            "source_dataset": ref.source_name,
            "shard_index": ref.shard_index,
            "row_index": ref.row_index,
        },
    }


class WorldModelTransitionDataset(Dataset):
    def __init__(
        self,
        indexed: IndexedDataset,
        *,
        include_metadata: bool = True,
        cache_size: int = 2,
    ) -> None:
        self.indexed = indexed
        self.include_metadata = include_metadata
        self.cache = ShardArrayCache(max_items=cache_size)

    def __len__(self) -> int:
        return len(self.indexed.transitions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.indexed.transitions[index]
        arrays = self.cache.get(ref.shard_path)
        sample = _row_to_transition_sample(arrays, ref)
        if not self.include_metadata:
            sample = {key: value for key, value in sample.items() if key != "metadata"}
        return sample


def build_sequence_refs(indexed: IndexedDataset, *, chunk_length: int, stride: int = 1) -> list[SequenceRef]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows: list[SequenceRef] = []
    for episode_key, refs in indexed.episodes.items():
        if len(refs) < chunk_length:
            continue
        for start in range(0, len(refs) - chunk_length + 1, stride):
            window = refs[start : start + chunk_length]
            expected_steps = list(range(window[0].step_in_episode, window[0].step_in_episode + chunk_length))
            actual_steps = [ref.step_in_episode for ref in window]
            if actual_steps != expected_steps:
                continue
            if any(ref.done for ref in window[:-1]):
                continue
            windows.append(
                SequenceRef(
                    source_name=window[0].source_name,
                    episode_key=episode_key,
                    transitions=tuple(window),
                )
            )
    return windows


class WorldModelSequenceDataset(Dataset):
    def __init__(
        self,
        indexed: IndexedDataset,
        *,
        chunk_length: int,
        stride: int = 1,
        include_metadata: bool = True,
        cache_size: int = 2,
    ) -> None:
        self.indexed = indexed
        self.include_metadata = include_metadata
        self.cache = ShardArrayCache(max_items=cache_size)
        self.sequence_refs = build_sequence_refs(indexed, chunk_length=chunk_length, stride=stride)

    def __len__(self) -> int:
        return len(self.sequence_refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq_ref = self.sequence_refs[index]
        obs_seq = []
        action_seq = []
        next_obs_seq = []
        reward_seq = []
        done_seq = []
        terminated_seq = []
        truncated_seq = []
        for ref in seq_ref.transitions:
            arrays = self.cache.get(ref.shard_path)
            row = _row_to_transition_sample(arrays, ref)
            obs_seq.append(row["obs"])
            action_seq.append(row["action"])
            next_obs_seq.append(row["next_obs"])
            reward_seq.append(row["reward"])
            done_seq.append(row["done"])
            terminated_seq.append(row["terminated"])
            truncated_seq.append(row["truncated"])

        payload: dict[str, Any] = {
            "obs": torch.stack(obs_seq, dim=0),
            "action": torch.stack(action_seq, dim=0),
            "next_obs": torch.stack(next_obs_seq, dim=0),
            "reward": torch.stack(reward_seq, dim=0),
            "done": torch.stack(done_seq, dim=0),
            "terminated": torch.stack(terminated_seq, dim=0),
            "truncated": torch.stack(truncated_seq, dim=0),
            "mask": torch.ones((len(seq_ref.transitions),), dtype=torch.bool),
        }
        if self.include_metadata:
            payload["metadata"] = {
                "episode_key": seq_ref.episode_key,
                "source_dataset": seq_ref.source_name,
                "route_signatures": [ref.route_signature for ref in seq_ref.transitions],
                "scene_signatures": [ref.scene_signature for ref in seq_ref.transitions],
                "step_in_episode": [ref.step_in_episode for ref in seq_ref.transitions],
            }
        return payload


def build_train_val_subsets(
    dataset: Dataset,
    *,
    val_ratio: float = 0.1,
) -> tuple[Dataset, Dataset]:
    total = len(dataset)
    if total == 0:
        return Subset(dataset, []), Subset(dataset, [])
    val_count = int(round(total * val_ratio))
    val_count = min(max(val_count, 0), total)
    train_count = total - val_count
    indices = list(range(total))
    return Subset(dataset, indices[:train_count]), Subset(dataset, indices[train_count:])


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    device: str | None = None,
) -> DataLoader:
    resolved_pin_memory = pin_memory if pin_memory is not None else bool(device and device.startswith("cuda"))
    resolved_persistent_workers = (
        persistent_workers if persistent_workers is not None else num_workers > 0
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": resolved_pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = resolved_persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def build_transition_datasets(
    dataset_roots: list[str | Path | DatasetRootConfig],
    *,
    include_metadata: bool = True,
    val_ratio: float = 0.1,
) -> tuple[Dataset, Dataset, IndexedDataset]:
    indexed = build_index(dataset_roots)
    dataset = WorldModelTransitionDataset(indexed, include_metadata=include_metadata)
    train_dataset, val_dataset = build_train_val_subsets(dataset, val_ratio=val_ratio)
    return train_dataset, val_dataset, indexed


def build_sequence_datasets(
    dataset_roots: list[str | Path | DatasetRootConfig],
    *,
    cfg: WorldModelSequenceConfig,
    include_metadata: bool = True,
) -> tuple[Dataset, Dataset, IndexedDataset]:
    indexed = build_index(dataset_roots)
    train_dataset, val_dataset = build_sequence_datasets_from_indexed(
        indexed,
        cfg=cfg,
        include_metadata=include_metadata,
    )
    return train_dataset, val_dataset, indexed


def build_sequence_datasets_from_indexed(
    indexed: IndexedDataset,
    *,
    cfg: WorldModelSequenceConfig,
    include_metadata: bool = True,
) -> tuple[Dataset, Dataset]:
    dataset = WorldModelSequenceDataset(
        indexed,
        chunk_length=cfg.chunk_length,
        stride=cfg.stride,
        include_metadata=include_metadata,
    )
    train_dataset, val_dataset = build_train_val_subsets(dataset, val_ratio=cfg.val_ratio)
    return train_dataset, val_dataset


def build_world_model_data_from_indexed(
    indexed: IndexedDataset,
    cfg: WorldModelDataConfig,
    *,
    device: str | None = None,
) -> WorldModelDataArtifacts:
    LOGGER.info(
        "Preparing world-model dataloaders batch_size=%d chunk_length=%d stride=%d num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        cfg.batch_size,
        cfg.chunk_length,
        cfg.stride,
        cfg.num_workers,
        cfg.pin_memory,
        cfg.persistent_workers,
        cfg.prefetch_factor,
    )
    sequence_cfg = WorldModelSequenceConfig(
        chunk_length=cfg.chunk_length,
        stride=cfg.stride,
        val_ratio=cfg.val_ratio,
        expected_num_actions=cfg.expected_num_actions,
    )
    train_dataset, val_dataset = build_sequence_datasets_from_indexed(
        indexed,
        cfg=sequence_cfg,
        include_metadata=cfg.include_metadata,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after sequence-window construction.")
    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )
    sample = train_dataset[0]
    obs_shape = tuple(sample["obs"].shape[1:])
    LOGGER.info(
        "Built world-model data train_windows=%d val_windows=%d train_batches=%d val_batches=%d obs_shape=%s",
        len(train_dataset),
        len(val_dataset),
        len(train_loader),
        len(val_loader),
        obs_shape,
    )
    return WorldModelDataArtifacts(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        indexed=indexed,
        obs_shape=obs_shape,
        num_actions=cfg.expected_num_actions,
    )


def build_world_model_data(
    cfg: WorldModelDataConfig,
    *,
    device: str | None = None,
) -> WorldModelDataArtifacts:
    indexed = build_index(
        cfg.dataset_paths,
    )
    return build_world_model_data_from_indexed(indexed, cfg, device=device)
