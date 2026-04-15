"""Momentum factor — trailing return lookbacks."""

from __future__ import annotations

import logging

import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)

_LOOKBACKS: dict[str, int] = {
    "return_1m": 21,
    "return_3m": 63,
    "return_6m": 126,
}


class MomentumFactor(BaseFactor):
    """Compute trailing percentage returns over 1-month, 3-month, and
    6-month windows using ``adj_close``.

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, return_1m, return_3m, return_6m``.
    """

    name: str = "momentum"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing momentum factors")
        df = prices[["date", "ticker", "adj_close"]].copy()
        df = df.sort_values(["ticker", "date"])

        for col_name, lookback in _LOOKBACKS.items():
            df[col_name] = df.groupby("ticker")["adj_close"].transform(
                lambda s, lb=lookback: s / s.shift(lb) - 1
            )
            logger.debug("Computed %s (lookback=%d)", col_name, lookback)

        result = df[["date", "ticker"] + list(_LOOKBACKS.keys())]
        logger.info("Momentum factors complete — %d rows", len(result))
        return result
