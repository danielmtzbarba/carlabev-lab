import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import yaml

from CarlaBEV.config import RunConfig as CarlaBEVRunConfig

from src.config.base_config import ArgsCarlaBEV, EnvConfig, to_carlabev_run_config
from src.config.experiment_loader import apply_experiment_config, save_run_config
from src.config.reset_protocol import ResetProtocolSampler, build_train_protocol_sampler
from src.config.studies.models import ExperimentSpec
from src.config.studies.registry import get_train_protocol


class ExperimentConfigTests(unittest.TestCase):
    def test_experiment_spec_accepts_legacy_aliases_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = ExperimentSpec(
                action_space="continuous",
                traffic="on",
                input_type="masks",
                reward_type="carl",
                curriculum="route_only",
                fov_mask="on",
                train_protocol_id="train",
                eval_protocol_ids=["eval"],
            )

        self.assertEqual(spec.action_mode, "continuous")
        self.assertEqual(spec.reward_mode, "carl")
        self.assertTrue(any("action_space" in str(item.message) for item in caught))
        self.assertTrue(any("reward_type" in str(item.message) for item in caught))

    def test_env_config_accepts_legacy_aliases_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env = EnvConfig(obs_space="bev", masked=False, action_space="continuous", reward_type="shaping")

        self.assertEqual(env.obs_mode, "bev_rgb")
        self.assertEqual(env.action_mode, "continuous")
        self.assertEqual(env.reward_mode, "shaping")
        self.assertTrue(any("obs_space" in str(item.message) for item in caught))
        self.assertTrue(any("action_space" in str(item.message) for item in caught))
        self.assertTrue(any("reward_type" in str(item.message) for item in caught))

    def test_apply_experiment_config_uses_canonical_env_fields(self):
        args = apply_experiment_config(ArgsCarlaBEV(), 1, study_id="PPO_NAVIGATION")

        self.assertEqual(args.env.action_mode, "discrete")
        self.assertEqual(args.env.input_type, "rgb")
        self.assertEqual(args.env.reward_mode, "shaping")
        self.assertEqual(args.env.obs_mode, "bev_rgb")
        self.assertIn("_act-discrete", args.exp_name)
        self.assertIn("_rwd-shaping", args.exp_name)

    def test_to_carlabev_run_config_returns_public_run_config(self):
        env = EnvConfig()
        env.action_mode = "continuous"
        env.input_type = "rgb"
        env.reward_mode = "shaping"
        args = ArgsCarlaBEV(env=env, num_envs=3, capture_video=False)

        run_cfg = to_carlabev_run_config(args)

        self.assertIsInstance(run_cfg, CarlaBEVRunConfig)
        self.assertEqual(run_cfg.env.action_mode, "continuous")
        self.assertEqual(run_cfg.env.obs_mode, "bev_rgb")
        self.assertEqual(run_cfg.env.reward_mode, "shaping")
        self.assertEqual(run_cfg.env.map_name, "Town01")
        self.assertEqual(run_cfg.num_envs, 3)

    def test_save_run_config_persists_canonical_fields(self):
        args = apply_experiment_config(ArgsCarlaBEV(), 1, study_id="PPO_NAVIGATION")
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os
                os.chdir(tmpdir)
                save_run_config(args)
                config_path = Path("runs") / args.exp_name / "config.yaml"
                payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            finally:
                os.chdir(cwd)

        self.assertEqual(payload["args"]["env"]["obs_mode"], "bev_rgb")
        self.assertEqual(payload["args"]["env"]["action_mode"], "discrete")
        self.assertEqual(payload["args"]["env"]["reward_mode"], "shaping")
        self.assertNotIn("action_space", payload["args"]["env"])
        self.assertIn("legacy_env_aliases", payload["compatibility"])
        self.assertEqual(payload["compatibility"]["legacy_experiment_aliases"]["action_space"], "discrete")


class ResetProtocolTests(unittest.TestCase):
    def test_random_navigation_sampler_uses_public_reset_builder(self):
        args = ArgsCarlaBEV(train_protocol_id="random_nav_train")
        sampler = build_train_protocol_sampler(args)

        options = sampler.initial_options(2)

        self.assertEqual(options["scene"], "rdm")
        self.assertIn("route_dist_range", options)
        self.assertEqual(options["protocol_id"], "random_nav_train")
        self.assertEqual(options["protocol_mode"], "random_navigation")
        self.assertTrue(np.array_equal(options["reset_mask"], np.array([True, True])))

    def test_scenario_catalog_sampler_uses_public_reset_builder(self):
        args = ArgsCarlaBEV(study_id="EDGE_CASE_SCENARIOS")
        protocol = get_train_protocol("EDGE_CASE_SCENARIOS", "jaywalk_only_train")
        sampler = ResetProtocolSampler(args, protocol)

        options = sampler.next_options(np.array([True, False], dtype=bool))

        self.assertIn("config_file", options)
        self.assertEqual(options["protocol_id"], "jaywalk_only_train")
        self.assertEqual(options["protocol_mode"], "scenario_catalog")
        self.assertTrue(np.array_equal(options["reset_mask"], np.array([True, False])))


if __name__ == "__main__":
    unittest.main()
