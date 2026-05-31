from __future__ import annotations

from copy import deepcopy
import hashlib
import random

import numpy as np
from CarlaBEV.config import (
    AuthoredSceneReset,
    RandomNavigationReset,
    ScenarioConfigReset,
    ScenarioPresetReset,
    build_authored_scene_options,
    build_random_navigation_options,
    build_scenario_config_options,
    build_scenario_preset_options,
)

from src.config.studies.models import (
    ProtocolSpec,
    RandomNavigationProtocol,
    ScenarioCatalogProtocol,
)
from src.config.studies.registry import get_eval_protocol, get_train_protocol


class ResetProtocolSampler:
    def __init__(self, cfg, protocol: ProtocolSpec):
        self.cfg = cfg
        self.protocol = protocol
        self._cycle_index = 0
        self._rng = random.Random(self._protocol_seed())

    def _protocol_seed(self) -> int:
        token = f"{self.cfg.seed}:{self.cfg.study_id}:{self.protocol.protocol_id}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def initial_options(self, num_envs: int) -> dict:
        return self.next_options(np.full((num_envs,), True, dtype=bool))

    def next_options(self, reset_mask, mean_return: float | None = None, curriculum_state=None) -> dict:
        if self.protocol.mode == "random_navigation":
            return self._random_navigation_options(
                reset_mask=reset_mask,
                mean_return=mean_return,
                curriculum_state=curriculum_state,
            )
        if self.protocol.mode == "scenario_catalog":
            return self._scenario_catalog_options(reset_mask=reset_mask)
        raise ValueError(f"Unsupported protocol mode: {self.protocol.mode}")

    def _random_navigation_options(self, reset_mask, mean_return=None, curriculum_state=None) -> dict:
        protocol: RandomNavigationProtocol = self.protocol
        num_vehicles = protocol.initial_num_vehicles
        route_dist_range = list(protocol.initial_route_dist_range)

        if (
            protocol.use_curriculum
            and curriculum_state is not None
            and mean_return is not None
        ):
            num_vehicles = curriculum_state.vehicle_schedule(mean_return)
            route_dist_range = list(curriculum_state.route_schedule(mean_return))

        options = build_random_navigation_options(
            RandomNavigationReset(
                num_vehicles=int(num_vehicles),
                route_dist_range=tuple(route_dist_range),
            ),
            reset_mask=reset_mask,
        )
        options["protocol_id"] = protocol.protocol_id
        options["protocol_mode"] = protocol.mode
        return options

    def _scenario_catalog_options(self, reset_mask) -> dict:
        protocol: ScenarioCatalogProtocol = self.protocol
        if protocol.sample_strategy == "cycle":
            entry = protocol.entries[self._cycle_index % len(protocol.entries)]
            self._cycle_index += 1
        else:
            entry = self._rng.choice(protocol.entries)

        variation_seed = None
        if protocol.variation_enabled:
            if protocol.variation_seed_mode == "random_per_reset":
                variation_seed = self._rng.randint(
                    protocol.variation_seed_min,
                    protocol.variation_seed_max,
                )
            elif protocol.variation_seed_mode == "fixed" and protocol.variation_seed is not None:
                variation_seed = protocol.variation_seed

        if entry.config_file is not None:
            options = build_authored_scene_options(
                AuthoredSceneReset(
                    config_file=entry.config_file,
                    variation_enabled=protocol.variation_enabled,
                    variation_seed=variation_seed,
                ),
                reset_mask=reset_mask,
            )
        elif entry.scenario_preset_id is not None:
            options = build_scenario_preset_options(
                ScenarioPresetReset(
                    preset_id=entry.scenario_preset_id,
                    overrides=deepcopy(entry.parameters),
                ),
                reset_mask=reset_mask,
            )
            if entry.level is not None:
                options["level"] = entry.level
        else:
            options = build_scenario_config_options(
                ScenarioConfigReset(
                    scenario_id=entry.scene or "rdm",
                    level=entry.level or 1,
                    parameters=deepcopy(entry.parameters),
                ),
                reset_mask=reset_mask,
            )
        options["protocol_id"] = protocol.protocol_id
        options["protocol_mode"] = protocol.mode
        return options


def build_train_protocol_sampler(cfg) -> ResetProtocolSampler:
    protocol = get_train_protocol(cfg.study_id, cfg.train_protocol_id)
    return ResetProtocolSampler(cfg, protocol)


def build_eval_protocol_samplers(cfg, protocol_ids: list[str] | None = None) -> dict[str, ResetProtocolSampler]:
    protocol_ids = protocol_ids or list(cfg.eval_protocol_ids)
    return {
        protocol_id: ResetProtocolSampler(cfg, get_eval_protocol(cfg.study_id, protocol_id))
        for protocol_id in protocol_ids
    }
