from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml

from src.world_model.policies import (
    resolve_ppo_checkpoint,
    validate_ppo_checkpoint_compatibility,
)


def _write_run_artifacts(run_dir: Path, payload: dict) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "ppo_final.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return checkpoint_path


@pytest.mark.unit
def test_resolve_ppo_checkpoint_uses_latest_pointer(tiny_cfg, tmp_workdir):
    run_dir = tmp_workdir / "runs" / tiny_cfg.study_id / f"exp_{tiny_cfg.exp_id}" / "trial_manual" / f"seed_{tiny_cfg.seed}"
    checkpoint_path = _write_run_artifacts(
        run_dir,
        {"args": {"study_id": tiny_cfg.study_id, "exp_id": tiny_cfg.exp_id, "algorithm": tiny_cfg.algorithm, "env": tiny_cfg.env.to_dict()}},
    )
    latest_pointer = tmp_workdir / "runs" / tiny_cfg.study_id / f"exp_{tiny_cfg.exp_id}" / "LATEST_RUN.json"
    latest_pointer.write_text(
        json.dumps({"run_dir": str(run_dir)}),
        encoding="utf-8",
    )

    resolved_checkpoint, resolved_run_dir = resolve_ppo_checkpoint(tiny_cfg)

    assert resolved_checkpoint == str(checkpoint_path)
    assert resolved_run_dir == str(run_dir)


@pytest.mark.unit
def test_validate_ppo_checkpoint_compatibility_accepts_matching_config(tiny_cfg, tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_path = _write_run_artifacts(
        run_dir,
        {"args": {"study_id": tiny_cfg.study_id, "exp_id": tiny_cfg.exp_id, "algorithm": tiny_cfg.algorithm, "env": tiny_cfg.env.to_dict()}},
    )
    matching_env = type(
        "MatchingEnv",
        (),
        {
            "single_observation_space": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(24, 96, 96),
                dtype=np.float32,
            ),
            "single_action_space": gym.spaces.Discrete(5),
        },
    )()

    payload = validate_ppo_checkpoint_compatibility(tiny_cfg, matching_env, str(checkpoint_path))

    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["source_run_dir"] == str(run_dir)


@pytest.mark.unit
def test_validate_ppo_checkpoint_compatibility_rejects_mismatch(tiny_cfg, discrete_env, tmp_path):
    run_dir = tmp_path / "run"
    saved_env = tiny_cfg.env.to_dict()
    saved_env["obs_mode"] = "bev_rgb"
    checkpoint_path = _write_run_artifacts(
        run_dir,
        {"args": {"study_id": tiny_cfg.study_id, "exp_id": tiny_cfg.exp_id, "algorithm": tiny_cfg.algorithm, "env": saved_env}},
    )

    with pytest.raises(ValueError, match="PPO checkpoint is incompatible"):
        validate_ppo_checkpoint_compatibility(tiny_cfg, discrete_env, str(checkpoint_path))
