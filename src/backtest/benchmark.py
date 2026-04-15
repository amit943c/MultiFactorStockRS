"""Benchmark tracker — compute an equity curve for a reference index (e.g. SPY)."""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkTracker:
    """Build a benchmark equity curve from the prices panel or yfinance.

    Parameters
    ----------
    ticker:
        Benchmark ticker symbol (default ``"SPY"``).
    """

    def __init__(self, ticker: str = "SPY") -> None:
        self._ticker = ticker
        logger.info("BenchmarkTracker initialised with ticker=%s", ticker)

    @property
    def ticker(self) -> str:
        return self._ticker

    def compute(
        self,
        prices_df: pd.DataFrame,
        start_date: datetime.date | str,
        end_date: datetime.date | str,
    ) -> pd.Series:
        """Return an equity curve (starting at 1.0) for the benchmark.

        Parameters
        ----------
        prices_df:
            The full prices panel with ``date, ticker, adj_close``.
        start_date, end_date:
            Date range (inclusive) for the curve.

        Returns
        -------
        pd.Series
            Date-indexed equity curve starting at 1.0.
        """
        start_dt = pd.Timestamp(start_date).date()
        end_dt = pd.Timestamp(end_date).date()

        bench = self._extract_from_prices(prices_df, start_dt, end_dt)
        if bench is None:
            bench = self._fetch_via_yfinance(start_dt, end_dt)
        if bench is None or bench.empty:
            logger.error(
                "Could not obtain benchmark data for %s", self._ticker
            )
            return pd.Series(dtype=float)

        equity = self._prices_to_equity(bench)
        logger.info(
            "Benchmark %s equity curve: %d days, final=%.4f",
            self._ticker,
            len(equity),
            equity.iloc[-1] if len(equity) else 0.0,
        )
        return equity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_from_prices(
        self,
        prices_df: pd.DataFrame,
        start_dt: datetime.date,
        end_dt: datetime.date,
    ) -> pd.Series | None:
        """Try to pull the benchmark from the existing prices DataFrame."""
        df = prices_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        bench = df.loc[df["ticker"] == self._ticker].copy()
        if bench.empty:
            logger.debug(
                "Ticker %s not found in prices_df — will try yfinance",
                self._ticker,
            )
            return None

        bench = bench.loc[
            (bench["date"] >= start_dt) & (bench["date"] <= end_dt)
        ]
        bench = bench.sort_values("date").drop_duplicates(subset="date", keep="last")
        return bench.set_index("date")["adj_close"]

    def _fetch_via_yfinance(
        self,
        start_dt: datetime.date,
        end_dt: datetime.date,
    ) -> pd.Series | None:
        """Fallback: download benchmark data using yfinance."""
        try:
            import yfinance as yf  # noqa: WPS433
        except ImportError:
            logger.warning(
                "yfinance not installed — cannot fetch benchmark %s",
                self._ticker,
            )
            return None

        logger.info("Fetching benchmark %s via yfinance", self._ticker)
        try:
            data = yf.download(
                self._ticker,
                start=str(start_dt),
                end=str(end_dt + datetime.timedelta(days=1)),
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            logger.exception("yfinance download failed for %s", self._ticker)
            return None

        if data.empty:
            return None

        close = data["Close"].squeeze()
        close.index = pd.to_datetime(close.index).date
        close.index.name = "date"
        return close

    @staticmethod
    def _prices_to_equity(prices: pd.Series) -> pd.Series:
        """Convert a price series to a cumulative equity curve starting at 1.0."""
        equity = prices / prices.iloc[0]
        equity.name = "benchmark_equity"
        return equity
