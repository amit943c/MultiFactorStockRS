"""Fundamental factor — PE ratio and EBITDA margin from external data."""

from __future__ import annotations

import logging

import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)

_FUNDAMENTAL_COLS: list[str] = ["pe_ratio", "ebitda_margin"]


class FundamentalFactor(BaseFactor):
    """Merge fundamental data onto the price panel.

    Parameters
    ----------
    fundamentals_df:
        DataFrame with at least ``date, ticker, pe_ratio, ebitda_margin``.
        This is typically sourced from a separate data layer (e.g. SEC
        filings, vendor API) and passed in at construction time.

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, pe_ratio, ebitda_margin``.
    """

    name: str = "fundamental"

    def __init__(self, fundamentals_df: pd.DataFrame) -> None:
        self._fundamentals = fundamentals_df

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing fundamental factors via merge")

        required = {"date", "ticker"} | set(_FUNDAMENTAL_COLS)
        missing = required - set(self._fundamentals.columns)
        if missing:
            raise ValueError(
                f"fundamentals_df is missing columns: {missing}"
            )

        spine = prices[["date", "ticker"]].copy()

        # Forward-fill fundamentals so each trading date inherits the most
        # recent reported value.
        fund = self._fundamentals[["date", "ticker"] + _FUNDAMENTAL_COLS].copy()
        fund = fund.sort_values(["ticker", "date"])

        merged = spine.merge(fund, on=["date", "ticker"], how="left")
        merged = merged.sort_values(["ticker", "date"])

        for col in _FUNDAMENTAL_COLS:
            merged[col] = merged.groupby("ticker")[col].transform(
                lambda s: s.ffill()
            )

        result = merged[["date", "ticker"] + _FUNDAMENTAL_COLS]
        logger.info(
            "Fundamental factors complete — %d rows, %d non-null pe_ratio",
            len(result),
            result["pe_ratio"].notna().sum(),
        )
        return result
