"""Yahoo Finance data source backed by the *yfinance* library.

Uses :func:`yf.download` for efficient batched price retrieval, and
per-ticker ``Ticker.info`` for fundamental snapshots.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.interfaces import DataSource

logger = logging.getLogger(__name__)

_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]


class YahooFinanceSource(DataSource):
    """Concrete :class:`DataSource` that pulls data from Yahoo Finance."""

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        """Batch-download OHLCV data and return a long-format DataFrame.

        Parameters
        ----------
        tickers:
            List of Yahoo Finance ticker symbols.
        start, end:
            Date range (inclusive of *start*, exclusive of *end* per yfinance
            convention).

        Returns
        -------
        pd.DataFrame
            Columns: ``date, ticker, open, high, low, close, volume, adj_close``.
        """
        logger.info(
            "Fetching prices for %d ticker(s) from %s to %s",
            len(tickers),
            start,
            end,
        )

        raw: pd.DataFrame = yf.download(
            tickers=tickers,
            start=str(start),
            end=str(end),
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )

        if raw.empty:
            logger.warning("yf.download returned an empty DataFrame")
            return pd.DataFrame(columns=_PRICE_COLUMNS)

        frames: list[pd.DataFrame] = []

        if len(tickers) == 1:
            sub = raw
            if isinstance(raw.columns, pd.MultiIndex):
                ticker_sym = tickers[0]
                for level in range(raw.columns.nlevels):
                    if ticker_sym in raw.columns.get_level_values(level):
                        try:
                            sub = raw.xs(ticker_sym, axis=1, level=level)
                        except KeyError:
                            pass
                        break
            frames.append(self._reshape_single(sub, tickers[0]))
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                available_tickers = raw.columns.get_level_values(1).unique() if raw.columns.nlevels > 1 else []
                for ticker in tickers:
                    try:
                        if ticker in available_tickers:
                            sub = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                        else:
                            sub = raw[ticker].dropna(how="all")
                        if sub.empty:
                            logger.warning("No price data for %s — skipping", ticker)
                            continue
                        frames.append(self._reshape_single(sub, ticker))
                    except (KeyError, TypeError):
                        logger.warning("Ticker %s not found in download result — skipping", ticker)
            else:
                for ticker in tickers:
                    try:
                        sub = raw[ticker].dropna(how="all")
                        if sub.empty:
                            logger.warning("No price data for %s — skipping", ticker)
                            continue
                        frames.append(self._reshape_single(sub, ticker))
                    except KeyError:
                        logger.warning("Ticker %s not found in download result — skipping", ticker)

        if not frames:
            return pd.DataFrame(columns=_PRICE_COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        logger.info("Fetched %d price rows for %d ticker(s)", len(result), result["ticker"].nunique())
        return result

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Retrieve PE ratio, EBITDA margin, market cap, and sector.

        Fetches ``Ticker.info`` one-by-one because yfinance does not expose
        a batched fundamentals API.  Failures are logged and the ticker is
        included with ``NaN`` values.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker, pe_ratio, ebitda_margin, market_cap, sector``.
        """
        logger.info("Fetching fundamentals for %d ticker(s)", len(tickers))
        records: list[dict[str, Any]] = []

        for ticker in tickers:
            try:
                info: dict[str, Any] = yf.Ticker(ticker).info or {}
                records.append(self._extract_fundamentals(ticker, info))
            except Exception:
                logger.exception("Failed to fetch fundamentals for %s", ticker)
                records.append(
                    {
                        "ticker": ticker,
                        "pe_ratio": np.nan,
                        "ebitda_margin": np.nan,
                        "market_cap": np.nan,
                        "sector": np.nan,
                    }
                )

        df = pd.DataFrame(records, columns=["ticker", "pe_ratio", "ebitda_margin", "market_cap", "sector"])
        logger.info("Fundamentals retrieved for %d ticker(s)", len(df))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalise a single-ticker OHLCV frame into the canonical schema."""
        out = df.copy()

        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [
                str(c[0]).strip() if isinstance(c, tuple) else str(c).strip()
                for c in out.columns
            ]

        out.index.name = "date"
        out = out.reset_index()

        col_map: dict[str, str] = {}
        for col in out.columns:
            lower = str(col).lower().strip().replace(" ", "_")
            if lower in {"adj_close", "adj close", "adjclose", "adj_close"}:
                col_map[col] = "adj_close"
            elif lower in {"open", "high", "low", "close", "volume", "date", "price"}:
                target = "date" if lower == "date" else lower
                if lower == "price":
                    target = "date"
                col_map[col] = target
        out = out.rename(columns=col_map)

        if "adj_close" not in out.columns:
            out["adj_close"] = out.get("close", np.nan)

        for needed in ["open", "high", "low", "close", "volume"]:
            if needed not in out.columns:
                out[needed] = np.nan

        out["ticker"] = ticker
        return out[_PRICE_COLUMNS]

    @staticmethod
    def _extract_fundamentals(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
        """Pull the required fundamental fields from a yfinance info dict."""
        ebitda = info.get("ebitda")
        revenue = info.get("totalRevenue")
        ebitda_margin = (ebitda / revenue) if (ebitda and revenue) else np.nan

        return {
            "ticker": ticker,
            "pe_ratio": info.get("trailingPE", np.nan),
            "ebitda_margin": ebitda_margin,
            "market_cap": info.get("marketCap", np.nan),
            "sector": info.get("sector", np.nan),
        }
