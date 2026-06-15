from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

_FORMAT = "[%(levelname)s] %(asctime)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


def _stream_supports_color() -> bool:
    if _env_flag("NO_COLOR"):
        return False
    if _env_flag("FORCE_COLOR") or _env_flag("CLICOLOR_FORCE") or _env_flag("PY_COLORS"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _build_console() -> Console:
    return Console(
        stderr=False,
        soft_wrap=True,
        force_terminal=_stream_supports_color(),
        no_color=not _stream_supports_color(),
    )


_CONSOLE = _build_console()


class _ColorFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _LEVEL_COLORS = {
        logging.DEBUG: "\033[33m",
        logging.INFO: "\033[34m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;37;41m",
    }

    def __init__(self, fmt: str, datefmt: str | None = None, *, use_color: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self.use_color:
            return rendered
        color = self._LEVEL_COLORS.get(record.levelno)
        if color is None:
            return rendered
        prefix = f"[{record.levelname}]"
        return rendered.replace(prefix, f"{color}{prefix}{self._RESET}", 1)


def configure_logging(*, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        logging.getLogger("carlabev_lab").setLevel(level)
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_ColorFormatter(_FORMAT, datefmt=_DATEFMT, use_color=_stream_supports_color()))
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
