from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetShardSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_index: int
    transitions: int
    path: str


class DatasetSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    total_transitions: int
    shard_count: int
    shards: list[DatasetShardSummaryModel]
    policy: str
    split: str
    study_id: str
    exp_id: int
    seed: int
    checkpoint_path: str | None = None
    source_run_dir: str | None = None


class DatasetRootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    source_name: str | None = None

    @field_validator("path")
    @classmethod
    def _path_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("dataset path must not be blank")
        return stripped


class WorldModelSequenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_length: int = 1
    stride: int = 1
    val_ratio: float = 0.1
    split_strategy: Literal["by_shard", "by_episode"] = "by_shard"
    expected_num_actions: int | None = 9
    action_space: Literal["discrete"] = "discrete"

    @field_validator("chunk_length", "stride")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("val_ratio")
    @classmethod
    def _valid_ratio(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("val_ratio must be in [0.0, 1.0)")
        return value


class DatasetValidationReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_roots: list[str]
    source_names: list[str]
    total_datasets: int
    total_shards: int
    total_transitions: int
    total_episodes: int
    valid_windows: dict[int, int]
    action_histogram: dict[str, int]
    mean_episode_length: float
    p50_episode_length: float
    p95_episode_length: float
    max_episode_length: int
    unique_routes_episode: int
    unique_scenes_episode: int
    unique_routes_transition: int
    unique_scenes_transition: int
    mean_done_rate: float
    mean_terminated_rate: float
    mean_truncated_rate: float
    route_uniqueness_rate_episode: float
    scene_uniqueness_rate_episode: float
    route_uniqueness_rate_transition: float
    scene_uniqueness_rate_transition: float
    dataset_transition_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)
