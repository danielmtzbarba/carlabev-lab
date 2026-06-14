from __future__ import annotations

import pytest

from src.utils import storage_paths
from src.utils.run_paths import RunPaths
from src.world_model.run_paths import WorldModelRunPaths


@pytest.mark.unit
def test_storage_paths_use_explicit_artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CARLABEV_ARTIFACT_ROOT", str(tmp_path / "horse"))

    assert storage_paths.runs_root() == tmp_path / "horse" / "runs"
    assert storage_paths.results_root() == tmp_path / "horse" / "results"
    assert storage_paths.datasets_root() == tmp_path / "horse" / "datasets"
    assert storage_paths.resolve_artifact_path("runs/demo").as_posix().endswith("/horse/runs/demo")


@pytest.mark.unit
def test_run_paths_follow_explicit_runs_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CARLABEV_RUNS_ROOT", str(tmp_path / "runs-root"))

    ppo_paths = RunPaths(study_id="PPO_NAVIGATION", exp_id=26, trial_number=1, seed=7)
    world_model_paths = WorldModelRunPaths("wm-demo")

    assert ppo_paths.run_dir == tmp_path / "runs-root" / "PPO_NAVIGATION" / "exp_26" / "trial_1" / "seed_7"
    assert world_model_paths.run_dir == tmp_path / "runs-root" / "world_model" / "wm-demo"
