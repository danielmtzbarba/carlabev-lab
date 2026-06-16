from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger
from rich.console import Console
from rich.progress import Progress

_CONFIGURED = False
_CONSOLE: Console | None = None
_FILE_SINKS: set[Path] = set()


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
    color_enabled = _stream_supports_color()
    return Console(
        stderr=False,
        soft_wrap=True,
        force_terminal=color_enabled,
        no_color=not color_enabled,
    )


def _shorten_path(path_value: Path) -> str:
    expanded = path_value.expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except OSError:
        resolved = expanded

    cwd = Path.cwd()
    home = Path.home()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(home)}"
    except ValueError:
        return str(resolved)


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return _shorten_path(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple, set)):
        return ",".join(_format_value(item) for item in value)
    if isinstance(value, str):
        if value.startswith("/") or value.startswith("~"):
            return _shorten_path(Path(value))
        return value
    return str(value)


def kv_message(message: str, /, **kwargs: Any) -> str:
    if not kwargs:
        return message
    serialized = " ".join(f"{key}={_format_value(value)}" for key, value in kwargs.items())
    return f"{message} | {serialized}"


def event_message(stage: str, phase: str | None = None, /, **kwargs: Any) -> str:
    del stage
    phase_label = "-" if phase is None else phase.upper()
    serialized = "-" if not kwargs else " ".join(f"{key}={_format_value(value)}" for key, value in kwargs.items())
    return f"{phase_label} | {serialized}"


def _colorize_stage_phase(stage_phase: str, *, color: bool) -> str:
    if not color:
        return stage_phase
    return f"\033[93m{stage_phase}\033[0m"


def _colorize_payload(payload: str, *, color: bool) -> str:
    if not color:
        return payload
    if payload == "-":
        return "\033[2m-\033[0m"
    tokens: list[str] = []
    for token in payload.split(" "):
        if "=" not in token:
            tokens.append(token)
            continue
        key, value = token.split("=", 1)
        tokens.append(f"\033[92m{key}\033[0m=\033[97m{value}\033[0m")
    return " ".join(tokens)


def _escape_loguru_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _render_message(record: dict[str, Any], *, color: bool) -> str:
    level_name = record["level"].name
    prefix = f"[{level_name}]"
    if color:
        prefix = f"\033[94m{prefix}\033[0m"

    timestamp = record["time"].strftime("%H:%M:%S")
    if color:
        timestamp = f"\033[97m{timestamp}\033[0m"
    message = _escape_loguru_braces(record["message"])
    parts = message.split(" | ", 1)
    sep = "\033[90m|\033[0m" if color else "|"
    if len(parts) == 1:
        return f"{prefix} {timestamp} {sep} {message}\n"
    stage_phase, payload = parts
    stage_phase = _colorize_stage_phase(stage_phase, color=color)
    payload = _colorize_payload(payload, color=color)
    return f"{prefix} {timestamp} {sep} {stage_phase} {sep} {payload}\n"


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        message = record.getMessage()
        _loguru_logger.bind(external_logger=record.name).opt(depth=6, exception=record.exc_info).log(level, message)


def _reset_root_logging(level: int) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(_InterceptHandler())
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def configure_logging(*, level: int = logging.INFO) -> None:
    global _CONFIGURED, _CONSOLE
    color_enabled = _stream_supports_color()
    if _CONFIGURED:
        _loguru_logger.remove()
        _FILE_SINKS.clear()
    _loguru_logger.remove()
    _FILE_SINKS.clear()
    _CONSOLE = _build_console()
    _loguru_logger.add(
        sys.stdout,
        level=logging.getLevelName(level),
        format=lambda record: _render_message(record, color=color_enabled),
        colorize=False,
        backtrace=False,
        diagnose=False,
    )
    _reset_root_logging(level)
    _CONFIGURED = True


def get_logger(name: str):
    if not _CONFIGURED:
        configure_logging()
    namespace = name if name.startswith("carlabev_lab") else f"carlabev_lab.{name}"
    return _loguru_logger.bind(logger_name=namespace)


def get_console() -> Console:
    global _CONSOLE
    if not _CONFIGURED or _CONSOLE is None:
        configure_logging()
    assert _CONSOLE is not None
    return _CONSOLE


def build_progress(*columns: Any, **kwargs: Any) -> Progress:
    return Progress(
        *columns,
        console=get_console(),
        expand=True,
        redirect_stdout=False,
        redirect_stderr=False,
        **kwargs,
    )


def add_file_handler(path: str | Path, *, level: int = logging.INFO) -> None:
    if not _CONFIGURED:
        configure_logging(level=level)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved in _FILE_SINKS:
        return
    _loguru_logger.add(
        resolved,
        level=logging.getLevelName(level),
        format=lambda record: _render_message(record, color=False),
        colorize=False,
        backtrace=False,
        diagnose=False,
    )
    _FILE_SINKS.add(resolved)
