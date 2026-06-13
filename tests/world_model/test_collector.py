from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.world_model.collector import collect_dataset
from src.world_model.inspect import inspect_dataset_path
from src.world_model.summary import summarize_dataset


@pytest.mark.integration
def test_collect_dataset_writes_shards(monkeypatch, tiny_cfg, discrete_env, tmp_path):
    import src.world_model.collector as collector_mod

    monkeypatch.setattr(collector_mod, "make_env", lambda *_args, **_kwargs: discrete_env)
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
            "straight_fraction": 0.5,
            "left_turn_fraction": 0.25,
            "right_turn_fraction": 0.25,
        },
    )

    cfg = tiny_cfg
    cfg.num_envs = discrete_env.num_envs

    summary = collect_dataset(
        cfg,
        total_transitions=5,
        steps_per_shard=3,
        output_dir=str(tmp_path / "dataset"),
        policy="random",
    )

    assert summary.total_transitions == 5
    assert summary.shard_count == 2
    assert (tmp_path / "dataset" / "summary.json").exists()

    shard_0 = np.load(tmp_path / "dataset" / "shard_000000.npz")
    shard_1 = np.load(tmp_path / "dataset" / "shard_000001.npz")

    assert shard_0["obs"].shape[0] == 3
    assert shard_1["obs"].shape[0] == 2
    assert shard_0["actions"].shape[1] == 1
    assert shard_0["protocol_id"][0] == "random_nav_train"
    assert int(shard_0["reset_seed"][0]) == 1000
    assert shard_0["route_signature"][0] == "route-0"
    assert shard_0["scene_signature"][0] == "scene-0"
    assert float(shard_0["straight_fraction"][0]) == pytest.approx(0.5)


@pytest.mark.integration
def test_inspect_dataset_path_reads_summary_and_shard(monkeypatch, tiny_cfg, discrete_env, tmp_path):
    import src.world_model.collector as collector_mod

    monkeypatch.setattr(collector_mod, "make_env", lambda *_args, **_kwargs: discrete_env)
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
            "straight_fraction": 0.5,
            "left_turn_fraction": 0.25,
            "right_turn_fraction": 0.25,
        },
    )

    cfg = tiny_cfg
    cfg.num_envs = discrete_env.num_envs
    output_dir = tmp_path / "dataset"
    collect_dataset(
        cfg,
        total_transitions=4,
        steps_per_shard=2,
        output_dir=str(output_dir),
        policy="random",
    )

    payload = inspect_dataset_path(str(output_dir), shard_index=1)

    assert payload["summary"]["shard_count"] == 2
    assert payload["arrays"]["obs"]["shape"][0] == 2
    assert payload["arrays"]["actions"]["dtype"] in {"int64", "int32"}
    assert payload["arrays"]["route_signature"]["shape"][0] == 2
    assert payload["arrays"]["straight_fraction"]["dtype"] == "float32"


@pytest.mark.integration
def test_summarize_dataset_aggregates_quality_metrics(monkeypatch, tiny_cfg, discrete_env, tmp_path):
    import src.world_model.collector as collector_mod

    monkeypatch.setattr(collector_mod, "make_env", lambda *_args, **_kwargs: discrete_env)
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
            "straight_fraction": 0.5 + 0.1 * env_index,
            "left_turn_fraction": 0.25,
            "right_turn_fraction": 0.25 - 0.1 * env_index,
        },
    )

    cfg = tiny_cfg
    cfg.num_envs = discrete_env.num_envs
    output_dir = tmp_path / "dataset"
    collect_dataset(
        cfg,
        total_transitions=4,
        steps_per_shard=2,
        output_dir=str(output_dir),
        policy="random",
        show_progress=False,
    )

    payload = summarize_dataset(str(output_dir))

    assert payload["total_transitions"] == 4
    assert payload["shard_count"] == 2
    assert payload["unique_routes"] == 2
    assert payload["unique_scenes"] == 2
    assert payload["done_rate"] == pytest.approx(1.0)
    assert payload["mean_reward"] == pytest.approx(1.0)
    assert payload["mean_straight_fraction"] == pytest.approx(0.55)
    assert payload["action_histogram"]
