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
        self._per_env_reset_counts: list[int] = []

    def _protocol_seed(self) -> int:
        token = f"{self.cfg.seed}:{self.cfg.study_id}:{self.protocol.protocol_id}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _derive_seed(self, *parts: object) -> int:
        token = ":".join(
            [str(self.cfg.seed), self.cfg.study_id, self.protocol.protocol_id, *(str(part) for part in parts)]
        )
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31 - 1)

    def _ensure_env_state(self, num_envs: int) -> None:
        if len(self._per_env_reset_counts) != num_envs:
            self._per_env_reset_counts = [0 for _ in range(num_envs)]

    def _seed_mode(self) -> str:
        return getattr(self.protocol, "reset_seed_mode", "hashed_episode")

    def _seed_for_env(self, env_index: int, reset_count: int) -> int:
        seed_mode = self._seed_mode()
        if seed_mode == "fixed":
            return self._derive_seed("fixed", env_index)
        if seed_mode == "incremental":
            return self._derive_seed("incremental", env_index, 0) + reset_count
        return self._derive_seed("hashed_episode", env_index, reset_count)

    def initial_reset_seeds(self, num_envs: int) -> list[int]:
        reset_mask = np.full((num_envs,), True, dtype=bool)
        return self.next_reset_seeds(reset_mask)

    def next_reset_seeds(self, reset_mask) -> list[int]:
        reset_mask = np.asarray(reset_mask, dtype=bool)
        num_envs = len(reset_mask)
        self._ensure_env_state(num_envs)
        seeds = [
            self._seed_for_env(env_index, self._per_env_reset_counts[env_index])
            for env_index in range(num_envs)
        ]
        for env_index, should_reset in enumerate(reset_mask):
            if should_reset:
                self._per_env_reset_counts[env_index] += 1
        return seeds

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
        reset_kwargs = protocol.backbone.model_dump(exclude_none=True)

        if (
            protocol.use_curriculum
            and curriculum_state is not None
            and mean_return is not None
        ):
            if protocol.curriculum_axis in {"near_ego_traffic", "both"}:
                num_vehicles = int(curriculum_state.vehicle_schedule(mean_return))
                reset_kwargs["num_vehicles"] = num_vehicles
                reset_kwargs["num_vehicles_near_ego"] = num_vehicles
            if protocol.curriculum_axis in {"route_distance", "both"}:
                reset_kwargs["route_dist_range"] = tuple(
                    curriculum_state.route_schedule(mean_return)
                )

        options = build_random_navigation_options(
            RandomNavigationReset(**reset_kwargs),
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
