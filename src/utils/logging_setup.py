"""Logging configuration for the multi-factor ranking system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(cfg: dict[str, Any] | None = None) -> None:
    """Configure the root logger from config."""
    level_str = "INFO"
    log_file: str | None = None

    if cfg and "logging" in cfg:
        level_str = cfg["logging"].get("level", "INFO")
        log_file = cfg["logging"].get("file")

    level = getattr(logging, level_str.upper(), logging.INFO)

    fmt = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
