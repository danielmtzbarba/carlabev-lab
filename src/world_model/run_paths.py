from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.storage_paths import runs_root


def build_world_model_run_id(run_name: str) -> str:
    return run_name


@dataclass(frozen=True)
class WorldModelRunPaths:
    run_name: str

    @property
    def experiment_root(self) -> Path:
        return runs_root() / "world_model" / self.run_name

    @property
    def run_dir(self) -> Path:
        return self.experiment_root

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def latest_pointer_path(self) -> Path:
        return self.experiment_root / "LATEST_RUN.json"

    @property
    def run_id(self) -> str:
        return build_world_model_run_id(self.run_name)

    def ensure_dirs(self) -> None:
        for path in (self.run_dir, self.checkpoints_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
