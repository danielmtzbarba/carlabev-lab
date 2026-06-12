import pytest

from src.utils.run_paths import RunPaths, build_run_id, build_run_label


@pytest.mark.unit
def test_build_run_identifiers_are_deterministic():
    assert build_run_label("PPO_NAVIGATION", 26) == "PPO_NAVIGATION_e26"
    assert build_run_id("PPO_NAVIGATION", 26, trial_number=3, seed=999) == "PPO_NAVIGATION_e26_t3_s999"


@pytest.mark.unit
def test_run_paths_generate_expected_tree(tmp_workdir):
    paths = RunPaths(study_id="PPO_NAVIGATION", exp_id=26, trial_number=4, seed=555)
    paths.ensure_dirs()

    assert paths.run_dir.exists()
    assert paths.checkpoints_dir.exists()
    assert paths.eval_final_dir.exists()
    assert paths.latest_pointer_path.as_posix() == "runs/PPO_NAVIGATION/exp_26/LATEST_RUN.json"
