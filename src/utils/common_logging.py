from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.progress import Progress
from rich.theme import Theme

_FORMAT = "[%(levelname)s] %(asctime)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False
_THEME = Theme(
    {
        "level_info": "blue",
        "level_debug": "yellow",
        "level_warning": "yellow",
        "level_error": "bold red",
        "level_critical": "bold white on red",
    }
)


class _LevelPrefixHighlighter(RegexHighlighter):
    highlights = [
        r"^\[(?P<level_info>INFO)\]",
        r"^\[(?P<level_debug>DEBUG)\]",
        r"^\[(?P<level_warning>WARNING)\]",
        r"^\[(?P<level_error>ERROR)\]",
        r"^\[(?P<level_critical>CRITICAL)\]",
    ]

def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


def _build_console() -> Console:
    force_terminal = _env_flag("FORCE_COLOR") or _env_flag("CLICOLOR_FORCE") or _env_flag("PY_COLORS")
    no_color = _env_flag("NO_COLOR")
    return Console(
        stderr=False,
        soft_wrap=True,
        theme=_THEME,
        force_terminal=force_terminal,
        no_color=no_color,
    )


_CONSOLE = _build_console()
def configure_logging(*, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        logging.getLogger("carlabev_lab").setLevel(level)
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = RichHandler(
        console=_CONSOLE,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format=_DATEFMT,
        highlighter=_LevelPrefixHighlighter(),
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root_logger.addHandler(handler)

    logger = logging.getLogger("carlabev_lab")
    logger.setLevel(level)
    logger.propagate = True
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    namespace = name if name.startswith("carlabev_lab") else f"carlabev_lab.{name}"
    return logging.getLogger(namespace)


def get_console() -> Console:
    if not _CONFIGURED:
        configure_logging()
    return _CONSOLE


def build_progress(*columns: Any, **kwargs: Any) -> Progress:
    if not _CONFIGURED:
        configure_logging()
    return Progress(
        *columns,
        console=_CONSOLE,
        expand=True,
        redirect_stdout=False,
        redirect_stderr=False,
        **kwargs,
    )


def add_file_handler(path: str | Path, *, level: int = logging.INFO) -> None:
    if not _CONFIGURED:
        configure_logging(level=level)
    logger = logging.getLogger("carlabev_lab")
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved:
            return
    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(file_handler)


def kv_message(message: str, /, **kwargs: Any) -> str:
    if not kwargs:
        return message
    serialized = " ".join(f"{key}={_format_value(value)}" for key, value in kwargs.items())
    return f"{message} | {serialized}"


def event_message(stage: str, phase: str | None = None, /, **kwargs: Any) -> str:
    stage_label = stage.upper()
    phase_label = "-" if phase is None else phase.upper()
    serialized = "-" if not kwargs else " ".join(f"{key}={_format_value(value)}" for key, value in kwargs.items())
    return f"{stage_label} - {phase_label} | {serialized}"


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)
