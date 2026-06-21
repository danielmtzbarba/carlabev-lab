import json
import warnings

import pytest
import yaml

from CarlaBEV.config import RunConfig as CarlaBEVRunConfig

from src.config.base_config import ArgsCarlaBEV, EnvConfig, to_carlabev_run_config
from src.config.experiment_loader import apply_experiment_config, save_run_config


@pytest.mark.unit
def test_env_config_legacy_aliases_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = EnvConfig(obs_space="bev", masked=False, action_space="continuous", reward_type="shaping")

    assert env.obs_mode == "bev_rgb"
    assert env.action_mode == "continuous"
    assert env.reward_mode == "shaping"
    assert any("obs_space" in str(item.message) for item in caught)


@pytest.mark.unit
def test_apply_experiment_config_maps_study_fields():
    args = apply_experiment_config(ArgsCarlaBEV(), 30, study_id="PPO_NAVIGATION")

    assert args.algorithm == "cnn-ppo"
    assert args.train_protocol_id == "traffic_route_curriculum_train"
    assert args.eval_protocol_ids == ["navigation_eval"]
    assert args.env.ego_anchor_y_frac == 0.75
    assert args.env.reward_mode == "carl"
    assert args.env.input_type == "masks"
    assert args.env.scene_library_enabled is True
    assert args.env.scene_library_read_only is True
    assert args.env.scene_library_path.endswith("ppo_navigation.db")
    assert args.run_id.startswith("PPO_NAVIGATION_e30")


@pytest.mark.unit
def test_to_carlabev_run_config_returns_public_contract():
    args = apply_experiment_config(ArgsCarlaBEV(), 1, study_id="PPO_NAVIGATION")

    run_cfg = to_carlabev_run_config(args)

    assert isinstance(run_cfg, CarlaBEVRunConfig)
    assert run_cfg.env.obs_mode == "bev_rgb"
    assert run_cfg.env.action_mode == "discrete"
    assert run_cfg.num_envs == args.num_envs


@pytest.mark.unit
def test_save_run_config_persists_study_and_latest_pointer(tmp_workdir):
    args = apply_experiment_config(ArgsCarlaBEV(), 1, study_id="PPO_NAVIGATION")

    save_run_config(args)

    config_path = tmp_workdir / args.run_dir / "config.yaml"
    latest_path = tmp_workdir / "runs" / args.study_id / f"exp_{args.exp_id}" / "LATEST_RUN.json"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    latest = json.loads(latest_path.read_text(encoding="utf-8"))

    assert payload["study"]["study_id"] == args.study_id
    assert payload["experiment"]["exp_id"] == args.exp_id
    assert payload["compatibility"]["legacy_experiment_aliases"]["action_space"] == "discrete"
    assert latest["run_id"] == args.run_id
