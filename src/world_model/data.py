from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.common_logging import event_message, get_logger
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
class SequenceWindowCache:
    window_transition_indices: np.ndarray
    path: Path | None
    cache_hit: bool


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
    start = time.perf_counter()
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    _validate_shard_arrays(arrays, shard_path)
    load_ms = (time.perf_counter() - start) * 1000.0
    LOGGER.info(event_message("DATA", "SHARD_LOAD", path=shard_path, rows=int(arrays["obs"].shape[0]), load_ms=load_ms))
    return arrays


def build_index(
    dataset_roots: list[str | Path | DatasetRootConfig],
    *,
    progress=None,
    task_id: int | None = None,
) -> IndexedDataset:
    LOGGER.info(event_message("DATA", "INDEX_START", roots=len(dataset_roots)))
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

    dataset_infos: list[tuple[int, DatasetRootConfig, Path, DatasetSummaryModel, str]] = []
    total_shards = 0
    for dataset_id, root_cfg in enumerate(normalized_roots):
        dataset_dir = _coerce_dataset_dir(root_cfg.path)
        summary = load_dataset_summary(dataset_dir)
        source_name = _dataset_source_name(root_cfg, summary)
        dataset_infos.append((dataset_id, root_cfg, dataset_dir, summary, source_name))
        total_shards += summary.shard_count

    if progress is not None and task_id is not None:
        progress.reset(
            task_id,
            total=max(total_shards, 1),
            completed=0,
            visible=True,
            candidate="dataset index",
            stage="loading shards",
        )

    for dataset_id, root_cfg, dataset_dir, summary, source_name in dataset_infos:
        LOGGER.info(
            event_message(
                "DATA",
                "ROOT_LOAD",
                path=dataset_dir,
                source=source_name,
                transitions=summary.total_transitions,
                shards=summary.shard_count,
            )
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
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

    ordered_episodes: dict[tuple[int, int, int], tuple[TransitionRef, ...]] = {}
    for episode_key, refs in episodes.items():
        ordered = tuple(sorted(refs, key=lambda ref: (ref.step_in_episode, ref.shard_index, ref.row_index)))
        ordered_episodes[episode_key] = ordered
    LOGGER.info(
        event_message(
            "DATA",
            "INDEX_DONE",
            transitions=len(transitions),
            episodes=len(ordered_episodes),
            shards=len(shard_records),
        )
    )
    if progress is not None and task_id is not None:
        progress.update(task_id, stage="index complete", visible=False)
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


def build_sequence_window_indices(
    indexed: IndexedDataset,
    *,
    chunk_length: int,
    stride: int = 1,
) -> np.ndarray:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    episode_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for transition_index, ref in enumerate(indexed.transitions):
        episode_indices[ref.episode_key].append(transition_index)

    windows: list[list[int]] = []
    for episode_key, transition_indices in episode_indices.items():
        ordered_indices = sorted(
            transition_indices,
            key=lambda idx: (
                indexed.transitions[idx].step_in_episode,
                indexed.transitions[idx].shard_index,
                indexed.transitions[idx].row_index,
            ),
        )
        if len(ordered_indices) < chunk_length:
            continue
        for start in range(0, len(ordered_indices) - chunk_length + 1, stride):
            window_indices = ordered_indices[start : start + chunk_length]
            refs = [indexed.transitions[idx] for idx in window_indices]
            expected_steps = list(range(refs[0].step_in_episode, refs[0].step_in_episode + chunk_length))
            actual_steps = [ref.step_in_episode for ref in refs]
            if actual_steps != expected_steps:
                continue
            if any(ref.done for ref in refs[:-1]):
                continue
            windows.append(window_indices)

    if not windows:
        return np.empty((0, chunk_length), dtype=np.int64)
    return np.asarray(windows, dtype=np.int64)


def _sequence_cache_dir(indexed: IndexedDataset, explicit_cache_dir: str | None) -> Path:
    if explicit_cache_dir:
        return resolve_artifact_path(explicit_cache_dir)
    if len(indexed.sources) == 1:
        return indexed.sources[0].root_path / ".wm_cache"
    return resolve_artifact_path("datasets/world_model/.wm_cache")


def _sequence_cache_fingerprint(indexed: IndexedDataset) -> str:
    payload = {
        "sources": [
            {
                "dataset_id": source.dataset_id,
                "root_path": str(source.root_path),
                "source_name": source.source_name,
                "study_id": source.summary.study_id,
                "exp_id": source.summary.exp_id,
                "seed": source.summary.seed,
                "policy": source.summary.policy,
                "split": source.summary.split,
                "total_transitions": source.summary.total_transitions,
                "shard_count": source.summary.shard_count,
            }
            for source in indexed.sources
        ],
        "shards": [
            {
                "dataset_id": shard.dataset_id,
                "shard_index": shard.shard_index,
                "path": str(shard.shard_path),
                "row_count": shard.row_count,
                "size": shard.shard_path.stat().st_size,
                "mtime_ns": shard.shard_path.stat().st_mtime_ns,
            }
            for shard in indexed.shards
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sequence_cache_path(
    indexed: IndexedDataset,
    *,
    chunk_length: int,
    stride: int,
    explicit_cache_dir: str | None,
) -> Path:
    cache_dir = _sequence_cache_dir(indexed, explicit_cache_dir)
    fingerprint = _sequence_cache_fingerprint(indexed)
    file_name = f"sequence_windows_chunk{chunk_length}_stride{stride}_{fingerprint}.npz"
    return cache_dir / file_name


def load_or_build_sequence_window_cache(
    indexed: IndexedDataset,
    *,
    chunk_length: int,
    stride: int = 1,
    enabled: bool = True,
    cache_dir: str | None = None,
) -> SequenceWindowCache:
    if not enabled:
        return SequenceWindowCache(
            window_transition_indices=build_sequence_window_indices(
                indexed,
                chunk_length=chunk_length,
                stride=stride,
            ),
            path=None,
            cache_hit=False,
        )

    cache_path = _sequence_cache_path(
        indexed,
        chunk_length=chunk_length,
        stride=stride,
        explicit_cache_dir=cache_dir,
    )
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            window_transition_indices = np.asarray(payload["window_transition_indices"], dtype=np.int64)
            cached_chunk_length = int(payload["chunk_length"])
            cached_stride = int(payload["stride"])
        if window_transition_indices.ndim != 2 or window_transition_indices.shape[1] != chunk_length:
            raise ValueError(
                f"Sequence cache {cache_path} has invalid shape {window_transition_indices.shape} "
                f"for chunk_length={chunk_length}"
            )
        if cached_chunk_length != chunk_length or cached_stride != stride:
            raise ValueError(
                f"Sequence cache {cache_path} does not match requested chunk_length/stride "
                f"({cached_chunk_length}, {cached_stride}) != ({chunk_length}, {stride})"
            )
        LOGGER.info(
            event_message(
                "DATA",
                "SEQ_CACHE_LOAD",
                path=cache_path,
                windows=int(window_transition_indices.shape[0]),
                chunk_length=chunk_length,
                stride=stride,
            )
        )
        return SequenceWindowCache(
            window_transition_indices=window_transition_indices,
            path=cache_path,
            cache_hit=True,
        )

    window_transition_indices = build_sequence_window_indices(
        indexed,
        chunk_length=chunk_length,
        stride=stride,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        window_transition_indices=window_transition_indices,
        chunk_length=np.asarray(chunk_length, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
    )
    LOGGER.info(
        event_message(
            "DATA",
            "SEQ_CACHE_SAVE",
            path=cache_path,
            windows=int(window_transition_indices.shape[0]),
            chunk_length=chunk_length,
            stride=stride,
        )
    )
    return SequenceWindowCache(
        window_transition_indices=window_transition_indices,
        path=cache_path,
        cache_hit=False,
    )


class WorldModelSequenceDataset(Dataset):
    def __init__(
        self,
        indexed: IndexedDataset,
        *,
        chunk_length: int,
        stride: int = 1,
        include_metadata: bool = True,
        cache_size: int = 2,
        window_transition_indices: np.ndarray | None = None,
    ) -> None:
        self.indexed = indexed
        self.include_metadata = include_metadata
        self.cache = ShardArrayCache(max_items=cache_size)
        if window_transition_indices is None:
            window_transition_indices = build_sequence_window_indices(
                indexed,
                chunk_length=chunk_length,
                stride=stride,
            )
        if window_transition_indices.ndim != 2:
            raise ValueError("window_transition_indices must be a rank-2 array")
        self.window_transition_indices = np.asarray(window_transition_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.window_transition_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        transition_indices = self.window_transition_indices[index]
        refs = [self.indexed.transitions[int(transition_index)] for transition_index in transition_indices]
        obs_seq = []
        action_seq = []
        next_obs_seq = []
        reward_seq = []
        done_seq = []
        terminated_seq = []
        truncated_seq = []
        for ref in refs:
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
            "mask": torch.ones((len(refs),), dtype=torch.bool),
        }
        if self.include_metadata:
            payload["metadata"] = {
                "episode_key": refs[0].episode_key,
                "source_dataset": refs[0].source_name,
                "route_signatures": [ref.route_signature for ref in refs],
                "scene_signatures": [ref.scene_signature for ref in refs],
                "step_in_episode": [ref.step_in_episode for ref in refs],
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
    cache_sequence_indices: bool = True,
    sequence_cache_dir: str | None = None,
) -> tuple[Dataset, Dataset, IndexedDataset]:
    indexed = build_index(dataset_roots)
    train_dataset, val_dataset = build_sequence_datasets_from_indexed(
        indexed,
        cfg=cfg,
        include_metadata=include_metadata,
        cache_sequence_indices=cache_sequence_indices,
        sequence_cache_dir=sequence_cache_dir,
    )
    return train_dataset, val_dataset, indexed


def build_sequence_datasets_from_indexed(
    indexed: IndexedDataset,
    *,
    cfg: WorldModelSequenceConfig,
    include_metadata: bool = True,
    cache_sequence_indices: bool = True,
    sequence_cache_dir: str | None = None,
) -> tuple[Dataset, Dataset]:
    sequence_window_cache = load_or_build_sequence_window_cache(
        indexed,
        chunk_length=cfg.chunk_length,
        stride=cfg.stride,
        enabled=cache_sequence_indices,
        cache_dir=sequence_cache_dir,
    )
    dataset = WorldModelSequenceDataset(
        indexed,
        chunk_length=cfg.chunk_length,
        stride=cfg.stride,
        include_metadata=include_metadata,
        window_transition_indices=sequence_window_cache.window_transition_indices,
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
        event_message(
            "DATA",
            "DATALOADER_PREP",
            batch_size=cfg.batch_size,
            chunk_length=cfg.chunk_length,
            stride=cfg.stride,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
        )
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
        cache_sequence_indices=cfg.cache_sequence_indices,
        sequence_cache_dir=cfg.sequence_cache_dir,
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
        event_message(
            "DATA",
            "DATALOADER_READY",
            train_windows=len(train_dataset),
            val_windows=len(val_dataset),
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            obs_shape=obs_shape,
        )
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
