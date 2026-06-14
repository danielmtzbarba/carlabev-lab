from __future__ import annotations

import logging
import sys
from pathlib import Path


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def configure_logging(*, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger("carlabev_lab").setLevel(level)
        return

    logger = logging.getLogger("carlabev_lab")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    namespace = name if name.startswith("carlabev_lab") else f"carlabev_lab.{name}"
    return logging.getLogger(namespace)


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
