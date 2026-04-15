"""Factor registry — discover, store, and orchestrate factor computation."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.factors.base import BaseFactor
from src.factors.composite import CompositeScorer
from src.factors.fundamental import FundamentalFactor
from src.factors.liquidity import LiquidityFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.momentum import MomentumFactor
from src.factors.sentiment import SentimentFactor
from src.factors.trend import TrendFactor
from src.factors.volatility import VolatilityFactor
from src.utils.config import get_factor_directions, get_factor_weights

logger = logging.getLogger(__name__)

_FACTOR_GROUP_CLS: dict[str, type[BaseFactor]] = {
    "momentum": MomentumFactor,
    "trend": TrendFactor,
    "mean_reversion": MeanReversionFactor,
    "liquidity": LiquidityFactor,
    "volatility": VolatilityFactor,
    "sentiment": SentimentFactor,
}


class FactorRegistry:
    """Maps factor group names to :class:`BaseFactor` instances and
    orchestrates batch computation.

    Parameters
    ----------
    factors:
        Optional pre-populated mapping.  Use :meth:`register` to add
        factors incrementally, or :meth:`build_default_registry` for a
        config-driven setup.
    """

    def __init__(self, factors: dict[str, BaseFactor] | None = None) -> None:
        self._factors: dict[str, BaseFactor] = dict(factors) if factors else {}

    def register(self, name: str, factor_instance: BaseFactor) -> None:
        """Register a factor under *name*, replacing any existing entry."""
        if name in self._factors:
            logger.warning("Overwriting existing factor '%s'", name)
        self._factors[name] = factor_instance
        logger.info("Registered factor '%s' (%s)", name, type(factor_instance).__name__)

    def get(self, name: str) -> BaseFactor:
        """Retrieve a registered factor by *name*.

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' is not registered")
        return self._factors[name]

    @property
    def names(self) -> list[str]:
        """Return sorted list of registered factor names."""
        return sorted(self._factors)

    def compute_all(
        self,
        prices: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> pd.DataFrame:
        """Compute every registered factor and merge results.

        Parameters
        ----------
        prices:
            Long-format OHLCV price DataFrame.
        cfg:
            Full configuration dict (used to derive weights / directions
            for the composite scorer).

        Returns
        -------
        DataFrame
            A single DataFrame keyed on ``(date, ticker)`` containing all
            individual factor columns plus ``composite_score`` and
            ``composite_rank``.
        """
        logger.info("Computing all factors: %s", self.names)

        merged = prices[["date", "ticker"]].drop_duplicates()

        for name in self.names:
            factor = self._factors[name]
            logger.info("Running factor group '%s'", name)
            result = factor.compute(prices)

            if not factor.validate(result):
                logger.error("Validation failed for '%s' — skipping", name)
                continue

            new_cols = [c for c in result.columns if c not in {"date", "ticker"}]
            merged = merged.merge(
                result[["date", "ticker"] + new_cols],
                on=["date", "ticker"],
                how="left",
            )
            logger.info("Merged %d column(s) from '%s'", len(new_cols), name)

        weights = get_factor_weights(cfg)
        directions = get_factor_directions(cfg)

        if weights:
            scorer = CompositeScorer(
                factor_weights=weights,
                factor_directions=directions,
            )
            merged = scorer.compute_composite(merged)
        else:
            logger.warning("No factor weights found — skipping composite scoring")

        logger.info(
            "Factor computation complete — %d rows × %d columns",
            len(merged),
            len(merged.columns),
        )
        return merged

    @classmethod
    def build_default_registry(
        cls,
        cfg: dict[str, Any],
        fundamentals_df: pd.DataFrame | None = None,
    ) -> FactorRegistry:
        """Create and populate a registry based on the YAML config.

        Only factor groups with ``enabled: true`` in *cfg* are registered.

        Parameters
        ----------
        cfg:
            Full configuration dict (as returned by
            :func:`src.utils.config.load_config`).
        fundamentals_df:
            Optional fundamentals DataFrame required when the
            ``fundamental`` factor group is enabled.

        Returns
        -------
        FactorRegistry
            A ready-to-use registry with all enabled factors registered.
        """
        registry = cls()
        factor_cfg = cfg.get("factors", {})

        for group_name, meta in factor_cfg.items():
            if not meta.get("enabled", False):
                logger.debug("Factor group '%s' is disabled — skipping", group_name)
                continue

            if group_name == "fundamental":
                if fundamentals_df is None:
                    logger.warning(
                        "Fundamental factor enabled but no fundamentals_df "
                        "provided — skipping"
                    )
                    continue
                instance: BaseFactor = FundamentalFactor(fundamentals_df)
            elif group_name in _FACTOR_GROUP_CLS:
                instance = _FACTOR_GROUP_CLS[group_name]()
            else:
                logger.warning(
                    "Unknown factor group '%s' in config — skipping", group_name
                )
                continue

            registry.register(group_name, instance)

        logger.info(
            "Default registry built with %d factor group(s): %s",
            len(registry._factors),
            registry.names,
        )
        return registry
