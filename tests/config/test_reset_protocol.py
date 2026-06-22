import numpy as np
import pytest

from src.config.base_config import ArgsCarlaBEV
from src.config.reset_protocol import ResetProtocolSampler, build_train_protocol_sampler
from src.config.studies.registry import get_train_protocol


@pytest.mark.envdep
def test_random_navigation_sampler_tags_options_with_protocol_metadata():
    args = ArgsCarlaBEV(train_protocol_id="traffic_fixed_train")
    sampler = build_train_protocol_sampler(args)

    options = sampler.initial_options(2)

    assert options["protocol_id"] == "traffic_fixed_train"
    assert options["protocol_mode"] == "random_navigation"
    assert np.array_equal(options["reset_mask"], np.array([True, True]))


@pytest.mark.envdep
def test_scenario_catalog_cycle_sampler_advances_entries():
    args = ArgsCarlaBEV(study_id="EDGE_CASE_SCENARIOS")
    protocol = get_train_protocol("EDGE_CASE_SCENARIOS", "all_edge_cases_train").model_copy(
        update={"sample_strategy": "cycle"}
    )
    sampler = ResetProtocolSampler(args, protocol)

    first = sampler.next_options(np.array([True], dtype=bool))
    second = sampler.next_options(np.array([True], dtype=bool))

    assert first["protocol_mode"] == "scenario_catalog"
    assert second["protocol_mode"] == "scenario_catalog"
    assert first["config_file"] != second["config_file"]


@pytest.mark.envdep
def test_benchmark_backed_medium_protocols_share_reset_seed_namespace():
    args_a = ArgsCarlaBEV(study_id="PPO_NAVIGATION_DIFFICULTY", seed=7)
    args_b = ArgsCarlaBEV(study_id="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR", seed=7)
    protocol_a = get_train_protocol("PPO_NAVIGATION_DIFFICULTY", "medium_train")
    protocol_b = get_train_protocol("PPO_NAVIGATION_MEDIUM_FOV_ANCHOR", "medium_train")

    sampler_a = ResetProtocolSampler(args_a, protocol_a)
    sampler_b = ResetProtocolSampler(args_b, protocol_b)

    assert sampler_a.initial_reset_seeds(3) == sampler_b.initial_reset_seeds(3)
