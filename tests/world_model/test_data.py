from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.conftest import FakeVectorEnv

from src.world_model.collector import collect_dataset
from src.world_model.contracts import WorldModelSequenceConfig
from src.world_model.data import (
    WorldModelSequenceDataset,
    WorldModelTransitionDataset,
    build_dataloader,
    build_index,
    build_sequence_window_indices,
    build_sequence_datasets,
    build_transition_datasets,
    load_or_build_sequence_window_cache,
)
from src.world_model.validate import validate_datasets


def _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, *, total_transitions: int = 6):
    import src.world_model.collector as collector_mod

    env = FakeVectorEnv(num_envs=1, obs_shape=(3, 8, 8), terminate_every=3)
    monkeypatch.setattr(collector_mod, "make_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        collector_mod,
        "build_train_protocol_sampler",
        lambda _cfg: SimpleNamespace(
            protocol=SimpleNamespace(protocol_id="random_nav_train"),
            initial_options=lambda num_envs: {
                "reset_mask": np.ones((num_envs,), dtype=bool),
                "protocol_id": "random_nav_train",
            },
            next_options=lambda reset_mask, mean_return=None, curriculum_state=None: {
                "reset_mask": reset_mask,
                "protocol_id": "random_nav_train",
            },
            initial_reset_seeds=lambda num_envs: [1000 + i for i in range(num_envs)],
            next_reset_seeds=lambda reset_mask: [2000 + i for i in range(len(reset_mask))],
        ),
    )
    monkeypatch.setattr(
        collector_mod,
        "_current_env_metadata",
        lambda _envs, env_index: {
            "route_signature": f"route-{env_index}",
            "scene_signature": f"scene-{env_index}",
            "straight_fraction": 0.6,
            "left_turn_fraction": 0.2,
            "right_turn_fraction": 0.2,
        },
    )

    cfg = tiny_cfg
    cfg.num_envs = 1
    output_dir = tmp_path / "dataset"
    collect_dataset(
        cfg,
        total_transitions=total_transitions,
        steps_per_shard=2,
        output_dir=str(output_dir),
        policy="random",
        show_progress=False,
    )
    return output_dir


@pytest.mark.integration
def test_transition_dataset_builds_index_and_samples(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)

    indexed = build_index([str(output_dir)])
    dataset = WorldModelTransitionDataset(indexed, include_metadata=False)

    assert len(indexed.transitions) == 6
    assert len(indexed.episodes) == 2
    sample = dataset[0]
    assert tuple(sample["obs"].shape) == (3, 8, 8)
    assert sample["action"].dtype == torch.int64
    assert tuple(sample["next_obs"].shape) == (3, 8, 8)


@pytest.mark.integration
def test_sequence_dataset_exposes_fixed_windows(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)

    indexed = build_index([str(output_dir)])
    dataset = WorldModelSequenceDataset(indexed, chunk_length=2, stride=1, include_metadata=True)

    assert len(dataset) == 4
    sample = dataset[0]
    assert tuple(sample["obs"].shape) == (2, 3, 8, 8)
    assert tuple(sample["action"].shape) == (2,)
    assert tuple(sample["mask"].shape) == (2,)
    assert sample["metadata"]["step_in_episode"] == [0, 1]


@pytest.mark.integration
def test_sequence_window_cache_is_saved_and_reused(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)
    indexed = build_index([str(output_dir)])

    first_cache = load_or_build_sequence_window_cache(
        indexed,
        chunk_length=2,
        stride=1,
        enabled=True,
    )

    assert first_cache.cache_hit is False
    assert first_cache.path is not None
    assert first_cache.path.exists()
    assert tuple(first_cache.window_transition_indices.shape) == (4, 2)

    second_cache = load_or_build_sequence_window_cache(
        indexed,
        chunk_length=2,
        stride=1,
        enabled=True,
    )

    assert second_cache.cache_hit is True
    assert second_cache.path == first_cache.path
    np.testing.assert_array_equal(
        first_cache.window_transition_indices,
        second_cache.window_transition_indices,
    )


@pytest.mark.integration
def test_sequence_dataset_can_use_cached_window_indices(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)
    indexed = build_index([str(output_dir)])
    window_transition_indices = build_sequence_window_indices(indexed, chunk_length=2, stride=1)

    dataset = WorldModelSequenceDataset(
        indexed,
        chunk_length=2,
        stride=1,
        include_metadata=True,
        window_transition_indices=window_transition_indices,
    )

    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["metadata"]["step_in_episode"] == [0, 1]


@pytest.mark.integration
def test_dataset_builders_support_train_val_and_dataloader(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)

    train_dataset, val_dataset, _indexed = build_transition_datasets(
        [str(output_dir)],
        include_metadata=False,
        val_ratio=0.25,
    )
    assert len(train_dataset) + len(val_dataset) == 6

    loader = build_dataloader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert tuple(batch["obs"].shape) == (2, 3, 8, 8)
    assert tuple(batch["action"].shape) == (2,)

    train_seq, val_seq, _ = build_sequence_datasets(
        [str(output_dir)],
        cfg=WorldModelSequenceConfig(chunk_length=2, stride=1, val_ratio=0.25),
        include_metadata=False,
        sequence_cache_dir=str(tmp_path / "wm-cache"),
    )
    assert len(train_seq) + len(val_seq) == 4
    assert list((tmp_path / "wm-cache").glob("sequence_windows_chunk2_stride1_*.npz"))


@pytest.mark.integration
def test_validate_datasets_reports_training_readiness(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)

    report = validate_datasets(
        [str(output_dir)],
        cfg=WorldModelSequenceConfig(chunk_length=1, stride=1, expected_num_actions=5),
        chunk_lengths=(1, 2),
    )

    assert report.total_transitions == 6
    assert report.total_episodes == 2
    assert report.valid_windows[1] == 6
    assert report.valid_windows[2] == 4
    assert report.max_episode_length == 3
    assert report.unique_routes_episode == 1
    assert report.route_uniqueness_rate_episode == pytest.approx(1.0 / 2.0)
    assert report.unique_routes_transition == 1
    assert report.action_histogram


@pytest.mark.unit
def test_build_dataloader_enables_cuda_friendly_options():
    dataset = [torch.tensor([1.0]), torch.tensor([2.0])]

    loader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        device="cuda",
    )

    assert loader.pin_memory is True
    assert loader.persistent_workers is True


@pytest.mark.unit
def test_build_dataloader_respects_explicit_loader_tuning():
    dataset = [torch.tensor([1.0]), torch.tensor([2.0])]

    loader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=4,
        device="cuda",
    )

    assert loader.pin_memory is False
    assert loader.persistent_workers is False
    assert loader.prefetch_factor == 4


@pytest.mark.integration
def test_build_index_updates_progress(monkeypatch, tiny_cfg, tmp_path):
    output_dir = _collect_sample_dataset(monkeypatch, tiny_cfg, tmp_path, total_transitions=6)

    updates: list[dict[str, object]] = []

    class FakeProgress:
        def reset(self, task_id, **kwargs):
            updates.append({"kind": "reset", "task_id": task_id, **kwargs})

        def update(self, task_id, **kwargs):
            updates.append({"kind": "update", "task_id": task_id, **kwargs})

    indexed = build_index([str(output_dir)], progress=FakeProgress(), task_id=7)

    assert len(indexed.transitions) == 6
    assert updates[0]["kind"] == "reset"
    assert updates[0]["total"] == 3
    shard_advances = [entry for entry in updates if entry["kind"] == "update" and entry.get("advance") == 1]
    assert len(shard_advances) == 3
