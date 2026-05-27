from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ActionSpace = Literal["discrete", "continuous"]
TrafficMode = Literal["on", "off"]
InputType = Literal["rgb", "masks"]
RewardType = Literal["shaping", "carl"]
CurriculumMode = Literal["off", "vehicles_only", "route_only", "both"]
Toggle = Literal["on", "off"]


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_space: ActionSpace
    traffic: TrafficMode
    input_type: InputType
    reward_type: RewardType
    curriculum: CurriculumMode
    fov_mask: Toggle
    scene: str | None = None
    scenario_preset_id: str | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    description: str
    optuna_study_name: str
    db_path: str
    default_algorithm: str = "cnn-ppo"
    metadata: dict[str, Any] = Field(default_factory=dict)
    experiments: dict[int, ExperimentSpec]
