"""Centralised configuration loader.

Reads YAML config files, merges overrides, and exposes a typed namespace
so the rest of the codebase never touches raw dicts.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration from YAML with optional overrides.

    Parameters
    ----------
    path:
        Path to a YAML config file.  Falls back to ``config/default_config.yaml``.
    overrides:
        Dict of values to merge on top of the loaded file.
    """
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s", path)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
        logger.debug("Applied config overrides: %s", list(overrides.keys()))

    return cfg


def get_factor_weights(cfg: dict[str, Any]) -> dict[str, float]:
    """Flatten the nested factor→sub-factor weights into a single dict.

    Returns mapping like ``{"return_1m": 0.15, "dist_ma50": 0.10, …}``.
    """
    weights: dict[str, float] = {}
    for _factor_group, meta in cfg.get("factors", {}).items():
        if not meta.get("enabled", False):
            continue
        for sub_factor, w in meta.get("weights", {}).items():
            weights[sub_factor] = w
    return weights


def get_factor_directions(cfg: dict[str, Any]) -> dict[str, str]:
    """Map each sub-factor name to its directionality string."""
    directions: dict[str, str] = {}
    for _factor_group, meta in cfg.get("factors", {}).items():
        if not meta.get("enabled", False):
            continue
        direction = meta.get("direction", "higher_is_better")
        for sub_factor in meta.get("weights", {}).keys():
            directions[sub_factor] = direction
    return directions
