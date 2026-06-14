from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.storage_paths import runs_root


def build_run_label(study_id: str, exp_id: int) -> str:
    return f"{study_id}_e{exp_id}"


def build_run_id(study_id: str, exp_id: int, trial_number: int | None = None, seed: int | None = None) -> str:
    base = build_run_label(study_id, exp_id)
    if trial_number is not None:
        base += f"_t{trial_number}"
    if seed is not None:
        base += f"_s{seed}"
    return base


@dataclass(frozen=True)
class RunPaths:
    study_id: str
    exp_id: int
    trial_number: int | None
    seed: int | None

    @property
    def experiment_root(self) -> Path:
        return runs_root() / self.study_id / f"exp_{self.exp_id}"

    @property
    def trial_token(self) -> str:
        return f"trial_{self.trial_number}" if self.trial_number is not None else "trial_manual"

    @property
    def seed_token(self) -> str:
        return f"seed_{self.seed}" if self.seed is not None else "seed_none"

    @property
    def run_dir(self) -> Path:
        return self.experiment_root / self.trial_token / self.seed_token

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def eval_intermediate_dir(self) -> Path:
        return self.eval_dir / "intermediate"

    @property
    def eval_final_dir(self) -> Path:
        return self.eval_dir / "final"

    @property
    def videos_dir(self) -> Path:
        return self.run_dir / "videos"

    @property
    def train_videos_dir(self) -> Path:
        return self.videos_dir / "train"

    @property
    def eval_videos_dir(self) -> Path:
        return self.videos_dir / "eval"

    @property
    def intermediate_eval_videos_dir(self) -> Path:
        return self.eval_videos_dir / "intermediate"

    @property
    def final_eval_videos_dir(self) -> Path:
        return self.eval_videos_dir / "final"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def latest_pointer_path(self) -> Path:
        return self.experiment_root / "LATEST_RUN.json"

    @property
    def run_label(self) -> str:
        return build_run_label(self.study_id, self.exp_id)

    @property
    def run_id(self) -> str:
        return build_run_id(self.study_id, self.exp_id, self.trial_number, self.seed)

    def ensure_dirs(self) -> None:
        for path in (
            self.run_dir,
            self.checkpoints_dir,
            self.eval_intermediate_dir,
            self.eval_final_dir,
            self.train_videos_dir,
            self.intermediate_eval_videos_dir,
            self.final_eval_videos_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
