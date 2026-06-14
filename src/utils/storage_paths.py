from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def _default_workspace_root() -> Path | None:
    user = os.environ.get("USER")
    if not user:
        return None
    candidate = Path("/data/horse/ws") / f"{user}-carlabev"
    if candidate.exists():
        return candidate
    return None


def artifact_root() -> Path:
    explicit = _env_path("CARLABEV_ARTIFACT_ROOT")
    if explicit is not None:
        return explicit
    workspace = _default_workspace_root()
    if workspace is not None:
        return workspace
    return Path(".")


def runs_root() -> Path:
    explicit = _env_path("CARLABEV_RUNS_ROOT")
    if explicit is not None:
        return explicit
    return artifact_root() / "runs"


def results_root() -> Path:
    explicit = _env_path("CARLABEV_RESULTS_ROOT")
    if explicit is not None:
        return explicit
    return artifact_root() / "results"


def datasets_root() -> Path:
    explicit = _env_path("CARLABEV_DATASETS_ROOT")
    if explicit is not None:
        return explicit
    return artifact_root() / "datasets"


def resolve_artifact_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    parts = candidate.parts
    if not parts:
        return artifact_root()
    if parts[0] == "runs":
        return runs_root().joinpath(*parts[1:])
    if parts[0] == "results":
        return results_root().joinpath(*parts[1:])
    if parts[0] == "datasets":
        return datasets_root().joinpath(*parts[1:])
    return candidate
