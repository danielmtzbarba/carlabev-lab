from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetShardSummary:
    shard_index: int
    transitions: int
    path: str


@dataclass(frozen=True)
class DatasetCollectionSummary:
    output_dir: str
    total_transitions: int
    shard_count: int
    shards: list[DatasetShardSummary]
    policy: str
    split: str
    study_id: str
    exp_id: int
    seed: int
    checkpoint_path: str | None = None
    source_run_dir: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.json"
