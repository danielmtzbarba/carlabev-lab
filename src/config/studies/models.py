from __future__ import annotations

from typing import Any, Annotated, Literal
import warnings

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field, model_validator


ActionMode = Literal["discrete", "continuous"]
TrafficMode = Literal["on", "off"]
InputType = Literal["rgb", "masks"]
SemanticMaskMode = Literal["binary", "2-class", "4-class", "5-class", "6-class", "7-class"]
TemporalFusionMode = Literal["stack", "vehicle_temporal", "vehicle_weighted"]
RewardMode = Literal["shaping", "carl"]
CurriculumMode = Literal["off", "vehicles_only", "route_only", "both"]
Toggle = Literal["on", "off"]
FovAnchorMode = Literal["center", "lookahead_75"]


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
    traffic: TrafficMode
    input_type: InputType
    semantic_mask_ch: SemanticMaskMode | None = None
    temporal_fusion_mode: TemporalFusionMode = "stack"
    reward_mode: RewardMode = Field(
        validation_alias=AliasChoices("reward_mode", "reward_type")
    )
    curriculum: CurriculumMode
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


class RandomNavigationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: str
    mode: Literal["random_navigation"]
    initial_num_vehicles: int = 0
    initial_route_dist_range: tuple[int, int] = (50, 150)
    eval_num_vehicles: int = 25
    eval_route_dist_range: tuple[int, int] = (250, 500)
    use_curriculum: bool = True


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


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    description: str
    optuna_study_name: str
    db_path: str
    default_algorithm: str = "cnn-ppo"
    metadata: dict[str, Any] = Field(default_factory=dict)
    train_protocols: dict[str, ProtocolSpec]
    eval_protocols: dict[str, ProtocolSpec]
    experiments: dict[int, ExperimentSpec]
