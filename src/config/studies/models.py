from __future__ import annotations

from typing import Any, Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field


ActionMode = Literal["discrete", "continuous"]
TrafficMode = Literal["on", "off"]
InputType = Literal["rgb", "masks"]
RewardMode = Literal["shaping", "carl"]
CurriculumMode = Literal["off", "vehicles_only", "route_only", "both"]
Toggle = Literal["on", "off"]


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_mode: ActionMode = Field(
        validation_alias=AliasChoices("action_mode", "action_space")
    )
    traffic: TrafficMode
    input_type: InputType
    reward_mode: RewardMode = Field(
        validation_alias=AliasChoices("reward_mode", "reward_type")
    )
    curriculum: CurriculumMode
    fov_mask: Toggle
    train_protocol_id: str
    eval_protocol_ids: list[str]
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

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
