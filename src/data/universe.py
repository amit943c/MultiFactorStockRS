"""Ticker-universe management.

Handles loading symbols from CSV files, hard-coded watchlists, or
config-driven dispatch, and filtering tickers by available price history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class UniverseManager:
    """Load and filter the investable ticker universe."""

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_csv(path: str | Path) -> list[str]:
        """Read tickers from a CSV file containing a ``ticker`` column.

        Parameters
        ----------
        path:
            Filesystem path to the CSV.  The file must contain a column
            named ``ticker`` (case-insensitive header matching is applied).

        Returns
        -------
        list[str]
            Sorted, deduplicated list of ticker symbols.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Universe CSV not found: {path}")

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        if "ticker" not in df.columns:
            raise ValueError(f"CSV {path} does not contain a 'ticker' column. Found: {list(df.columns)}")

        tickers = sorted(df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist())
        logger.info("Loaded %d ticker(s) from %s", len(tickers), path)
        return tickers

    @staticmethod
    def load_from_watchlist(tickers: list[str]) -> list[str]:
        """Normalise and deduplicate a hand-supplied ticker list.

        Parameters
        ----------
        tickers:
            Raw list of ticker strings.

        Returns
        -------
        list[str]
            Sorted, deduplicated, upper-cased symbols.
        """
        cleaned = sorted({t.strip().upper() for t in tickers if t.strip()})
        logger.info("Watchlist contains %d ticker(s)", len(cleaned))
        return cleaned

    # ------------------------------------------------------------------
    # Config-driven dispatch
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, cfg: dict[str, Any]) -> list[str]:
        """Load the ticker universe based on ``cfg["universe"]``.

        Supported ``cfg["universe"]["source"]`` values:

        * ``"csv"``       — reads ``cfg["universe"]["path"]``
        * ``"watchlist"`` — reads ``cfg["universe"]["tickers"]``

        Parameters
        ----------
        cfg:
            Full application config dict (must contain a ``universe`` key).

        Returns
        -------
        list[str]
            Ticker symbols ready for downstream consumption.

        Raises
        ------
        ValueError
            If the source type is not recognised.
        """
        uni_cfg: dict[str, Any] = cfg.get("universe", {})
        source = uni_cfg.get("source", "watchlist")
        logger.info("Loading universe with source=%s", source)

        if source == "csv":
            csv_path = uni_cfg.get("path") or uni_cfg.get("csv_path")
            if not csv_path:
                raise ValueError("Universe source is 'csv' but no 'path' or 'csv_path' configured")
            return cls.load_from_csv(csv_path)
        if source == "watchlist":
            wl = uni_cfg.get("watchlist") or uni_cfg.get("tickers") or []
            return cls.load_from_watchlist(wl)

        raise ValueError(f"Unknown universe source: {source!r}. Expected 'csv' or 'watchlist'.")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_history(
        tickers: list[str],
        prices_df: pd.DataFrame,
        min_days: int,
    ) -> list[str]:
        """Keep only tickers that have at least *min_days* of price data.

        Parameters
        ----------
        tickers:
            Candidate ticker list.
        prices_df:
            Long-format price DataFrame with a ``ticker`` column and one
            row per trading day.
        min_days:
            Minimum number of distinct dates required.

        Returns
        -------
        list[str]
            Filtered, sorted list of tickers meeting the threshold.
        """
        if prices_df.empty:
            logger.warning("Empty prices DataFrame — no tickers pass history filter")
            return []

        counts = prices_df.groupby("ticker")["date"].nunique()
        eligible = counts[counts >= min_days].index.tolist()
        kept = sorted(t for t in tickers if t in eligible)
        dropped = len(tickers) - len(kept)

        if dropped:
            logger.info(
                "History filter (>=%d days): kept %d, dropped %d ticker(s)",
                min_days,
                len(kept),
                dropped,
            )
        return kept
