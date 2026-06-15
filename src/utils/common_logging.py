from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

_FORMAT = "[%(levelname)s] %(asctime)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False
_CONSOLE = Console(stderr=False, soft_wrap=True)


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
        markup=True,
        rich_tracebacks=True,
        log_time_format=_DATEFMT,
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


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)
