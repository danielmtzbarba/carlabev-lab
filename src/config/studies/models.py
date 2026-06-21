from __future__ import annotations

from typing import Any, Annotated, Literal
import warnings

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field, model_validator


ActionMode = Literal["discrete", "continuous"]
InputType = Literal["rgb", "masks"]
SemanticMaskMode = Literal["binary", "2-class", "4-class", "5-class", "6-class", "7-class"]
TemporalFusionMode = Literal["stack", "vehicle_temporal", "vehicle_weighted"]
RewardMode = Literal["shaping", "carl"]
Toggle = Literal["on", "off"]
FovAnchorMode = Literal["center", "lookahead_75"]
RouteExtentMode = Literal["short", "medium", "large"]
TrafficRoleProfile = Literal["lead", "rear", "cross_path", "opposite_lane", "mix"]
CurriculumAxis = Literal["none", "near_ego_traffic", "route_distance", "both"]
TuningStageName = Literal[
    "policy_dynamics",
    "rollout_geometry",
    "loss_regularization",
    "network_capacity",
]
TuningObjectiveMetric = Literal["normalized_score", "mean_return"]
TuningSamplerKind = Literal["tpe"]
TuningPrunerKind = Literal["median", "none"]


def _warn_legacy_alias(legacy: str, canonical: str):
    warnings.warn(
        f"`{legacy}` is deprecated in carlabev-lab study definitions; use `{canonical}` instead.",
        FutureWarning,
        stacklevel=3,
    )


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_mode: ActionMode = Field(
        validation_alias=AliasChoices("action_mode", "action_space")
    )
    action_profile_id: str | None = None
    input_type: InputType
    semantic_mask_ch: SemanticMaskMode | None = None
    temporal_fusion_mode: TemporalFusionMode = "stack"
    reward_mode: RewardMode = Field(
        validation_alias=AliasChoices("reward_mode", "reward_type")
    )
    reward_profile_id: str | None = None
    fov_mask: Toggle
    fov_anchor: FovAnchorMode = "center"
    train_protocol_id: str
    eval_protocol_ids: list[str]
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_aliases(cls, data: Any):
        if not isinstance(data, dict):
            return data
        if "action_space" in data and "action_mode" not in data:
            _warn_legacy_alias("action_space", "action_mode")
        if "reward_type" in data and "reward_mode" not in data:
            _warn_legacy_alias("reward_type", "reward_mode")
        return data

    @model_validator(mode="after")
    def _validate_semantic_mask_mode(self):
        if self.input_type == "masks" and self.semantic_mask_ch is None:
            raise ValueError(
                "`semantic_mask_ch` is required when `input_type='masks'`."
            )
        if self.input_type == "rgb" and self.semantic_mask_ch is not None:
            raise ValueError(
                "`semantic_mask_ch` must be omitted when `input_type='rgb'`."
            )
        if self.input_type == "rgb" and self.temporal_fusion_mode != "stack":
            raise ValueError(
                "`temporal_fusion_mode` must be 'stack' when `input_type='rgb'`."
            )
        if self.temporal_fusion_mode != "stack":
            if self.input_type != "masks":
                raise ValueError(
                    "`temporal_fusion_mode` requires `input_type='masks'`."
                )
            if self.semantic_mask_ch not in {"4-class", "5-class", "6-class", "7-class"}:
                raise ValueError(
                    "`temporal_fusion_mode` requires a semantic mask layout with a vehicle channel "
                    "('4-class', '5-class', '6-class', or '7-class')."
                )
        return self

    @computed_field(return_type=str)
    @property
    def action_space(self) -> str:
        return self.action_mode

    @computed_field(return_type=str)
    @property
    def reward_type(self) -> str:
        return self.reward_mode


class ScenarioEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scene: str | None = None
    scenario_preset_id: str | None = None
    level: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    config_file: str | None = None
    notes: str | None = None


class SceneGenerationBackbone(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route_extent: RouteExtentMode | None = None
    route_dist_range: tuple[int, int] | None = None
    speed_profile: Literal["slow", "medium", "fast"] | None = None
    num_vehicles: int | None = None
    num_vehicles_near_ego: int | None = None
    traffic_role_profile: TrafficRoleProfile | None = None
    guaranteed_candidate_role: TrafficRoleProfile | None = None
    route_profile: Literal["mostly_straight", "single_left", "single_right", "multi_turn", "mixed"] | None = None
    route_profile_mix: dict[str, float] | None = None
    min_turns: int | None = None
    max_turns: int | None = None
    intersection_required: bool | None = None
    max_route_attempts: int | None = None
    ego_route_graph: str = "canonical"

    @model_validator(mode="after")
    def _validate_counts(self):
        if self.num_vehicles is not None and self.num_vehicles < 0:
            raise ValueError("`num_vehicles` must be >= 0.")
        if self.num_vehicles_near_ego is not None and self.num_vehicles_near_ego < 0:
            raise ValueError("`num_vehicles_near_ego` must be >= 0.")
        if (
            self.num_vehicles is not None
            and self.num_vehicles_near_ego is not None
            and self.num_vehicles_near_ego > self.num_vehicles
        ):
            raise ValueError("`num_vehicles_near_ego` must be <= `num_vehicles`.")
        if self.route_dist_range is not None and self.route_dist_range[0] > self.route_dist_range[1]:
            raise ValueError("`route_dist_range` must be ordered as (min, max).")
        return self


class SceneLibraryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    read_only: bool = True
    require_hit: bool = True
    path: str | None = None
    generator_version: str | None = None


class RandomNavigationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: str
    mode: Literal["random_navigation"]
    reset_seed_mode: Literal["fixed", "incremental", "hashed_episode"] = "hashed_episode"
    backbone: SceneGenerationBackbone
    use_curriculum: bool = False
    curriculum_axis: CurriculumAxis = "none"
    scene_library: SceneLibraryPolicy | None = None


class ScenarioCatalogProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: str
    mode: Literal["scenario_catalog"]
    entries: list[ScenarioEntry]
    sample_strategy: Literal["random", "cycle"] = "random"
    variation_enabled: bool = False
    variation_seed_mode: Literal["none", "random_per_reset", "fixed"] = "none"
    variation_seed: int | None = None
    variation_seed_min: int = 0
    variation_seed_max: int = 2_147_483_647


ProtocolSpec = Annotated[
    RandomNavigationProtocol | ScenarioCatalogProtocol,
    Field(discriminator="mode"),
]


class TuningStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: TuningStageName
    description: str
    n_trials: int
    total_timesteps: int
    inherits_from: list[TuningStageName] = Field(default_factory=list)
    selection_top_k: int = 1
    save_model: bool = False
    capture_video: bool = False


class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: str = "cnn-ppo"
    objective_metric: TuningObjectiveMetric = "normalized_score"
    sampler: TuningSamplerKind = "tpe"
    pruner: TuningPrunerKind = "median"
    num_seeds: int = 3
    eval_episodes: int = 30
    eval_final_episodes: int = 100
    stages: dict[TuningStageName, TuningStageConfig]


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    description: str
    optuna_study_name: str
    db_path: str
    default_algorithm: str = "cnn-ppo"
    metadata: dict[str, Any] = Field(default_factory=dict)
    tuning: TuningConfig | None = None
    train_protocols: dict[str, ProtocolSpec]
    eval_protocols: dict[str, ProtocolSpec]
    experiments: dict[int, ExperimentSpec]
