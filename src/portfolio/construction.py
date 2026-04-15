"""Portfolio construction — select holdings, assign weights, enforce constraints."""

from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Build equal-weighted, capped portfolios from a ranked stock universe.

    Parameters
    ----------
    cfg:
        Full application config dict (see ``config/default_config.yaml``).
        Relevant keys live under ``portfolio`` and ``rebalance``.
    """

    CASH_TICKER = "_CASH"

    def __init__(self, cfg: dict[str, Any]) -> None:
        port_cfg = cfg.get("portfolio", {})
        self._top_n: int = int(port_cfg.get("top_n", 30))
        self._max_weight: float = float(port_cfg.get("max_position_weight", 0.10))
        self._allow_cash: bool = bool(port_cfg.get("allow_cash", True))
        self._equal_weight: bool = bool(port_cfg.get("equal_weight", True))

        reb_cfg = cfg.get("rebalance", {})
        self._frequency: str = reb_cfg.get("frequency", "weekly")
        self._day_of_week: int = int(reb_cfg.get("day_of_week", 4))

        logger.info(
            "PortfolioConstructor: top_n=%d  max_weight=%.2f  freq=%s",
            self._top_n,
            self._max_weight,
            self._frequency,
        )

    # ------------------------------------------------------------------
    # Holding selection & weighting
    # ------------------------------------------------------------------

    def select_holdings(
        self,
        ranked_df: pd.DataFrame,
        rebalance_date: datetime.date,
    ) -> pd.DataFrame:
        """Select top-ranked stocks and assign capped equal weights.

        Parameters
        ----------
        ranked_df:
            DataFrame with columns ``date, ticker, composite_score,
            composite_rank`` (rank 1 = best).
        rebalance_date:
            The target rebalance date.  Only rows whose ``date`` matches this
            value are considered.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker, weight, composite_score, composite_rank``.
        """
        snapshot = ranked_df.loc[
            ranked_df["date"] == pd.Timestamp(rebalance_date)
        ].copy()

        if snapshot.empty:
            logger.warning("No ranked data for %s — returning cash-only portfolio", rebalance_date)
            return self._cash_only_frame()

        snapshot = snapshot.sort_values("composite_rank", ascending=True)
        selected = snapshot.head(self._top_n).copy()

        n_stocks = len(selected)
        n_cash = max(0, self._top_n - n_stocks) if self._allow_cash else 0

        if n_cash > 0:
            logger.info(
                "Only %d stocks available on %s; allocating %d cash slot(s)",
                n_stocks,
                rebalance_date,
                n_cash,
            )

        total_slots = n_stocks + n_cash
        if self._equal_weight:
            raw_weight = 1.0 / total_slots if total_slots > 0 else 0.0
            selected["weight"] = raw_weight
        else:
            selected["weight"] = 1.0 / n_stocks if n_stocks > 0 else 0.0

        selected = self._enforce_cap(selected)

        holdings = selected[["ticker", "weight", "composite_score", "composite_rank"]].copy()

        cash_weight = 1.0 - holdings["weight"].sum()
        if cash_weight > 1e-9:
            cash_row = pd.DataFrame(
                {
                    "ticker": [self.CASH_TICKER],
                    "weight": [cash_weight],
                    "composite_score": [np.nan],
                    "composite_rank": [np.nan],
                }
            )
            holdings = pd.concat([holdings, cash_row], ignore_index=True)

        logger.debug(
            "Holdings on %s: %d stocks + cash=%.4f",
            rebalance_date,
            n_stocks,
            cash_weight,
        )
        return holdings

    # ------------------------------------------------------------------
    # Rebalance-date generation
    # ------------------------------------------------------------------

    def generate_rebalance_dates(
        self,
        start: str,
        end: str,
        prices_df: pd.DataFrame,
    ) -> list[datetime.date]:
        """Return valid rebalance dates within ``[start, end]``.

        Parameters
        ----------
        start, end:
            ISO-format date strings (e.g. ``"2020-01-01"``).
        prices_df:
            Prices DataFrame with a ``date`` column used to verify trading days.

        Returns
        -------
        list[datetime.date]
            Sorted list of actual trading days matching the rebalance schedule.
        """
        trading_days = pd.to_datetime(prices_df["date"]).dt.date.unique()
        trading_set = set(trading_days)
        trading_idx = pd.DatetimeIndex(sorted(trading_days))

        start_dt = pd.Timestamp(start).date()
        end_dt = pd.Timestamp(end).date()

        mask = (trading_idx >= pd.Timestamp(start_dt)) & (trading_idx <= pd.Timestamp(end_dt))
        valid_idx = trading_idx[mask]

        if self._frequency == "monthly":
            dates = self._monthly_last_trading_days(valid_idx)
        else:
            dates = self._weekly_dates(valid_idx, self._day_of_week, trading_set)

        dates = sorted(d for d in dates if d in trading_set)
        logger.info(
            "Generated %d rebalance dates (%s) from %s to %s",
            len(dates),
            self._frequency,
            start,
            end,
        )
        return dates

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    @staticmethod
    def compute_turnover(
        prev_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> float:
        """One-way turnover: sum of absolute weight changes / 2."""
        all_tickers = set(prev_weights) | set(new_weights)
        total_abs = sum(
            abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in all_tickers
        )
        return total_abs / 2.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enforce_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iteratively cap weights at ``_max_weight`` and redistribute excess."""
        weights = df["weight"].values.copy()
        for _ in range(20):
            over_mask = weights > self._max_weight + 1e-9
            if not over_mask.any():
                break
            excess = (weights[over_mask] - self._max_weight).sum()
            weights[over_mask] = self._max_weight
            under_mask = ~over_mask
            n_under = under_mask.sum()
            if n_under == 0:
                break
            weights[under_mask] += excess / n_under
        df = df.copy()
        df["weight"] = weights
        return df

    def _cash_only_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": [self.CASH_TICKER],
                "weight": [1.0],
                "composite_score": [np.nan],
                "composite_rank": [np.nan],
            }
        )

    @staticmethod
    def _monthly_last_trading_days(
        trading_idx: pd.DatetimeIndex,
    ) -> list[datetime.date]:
        """Return the last trading day of each calendar month."""
        s = pd.Series(trading_idx, index=trading_idx)
        last_days = s.resample("ME").last().dropna()
        return [d.date() for d in last_days]

    @staticmethod
    def _weekly_dates(
        trading_idx: pd.DatetimeIndex,
        day_of_week: int,
        trading_set: set[datetime.date],
    ) -> list[datetime.date]:
        """Return trading days matching the target weekday.

        If the target weekday is not a trading day (e.g. holiday), fall back
        to the nearest prior trading day within the same week.
        """
        candidates: list[datetime.date] = []
        for ts in trading_idx:
            dt = ts.date()
            if dt.weekday() == day_of_week:
                candidates.append(dt)

        seen_weeks: set[tuple[int, int]] = set()
        for ts in trading_idx:
            dt = ts.date()
            iso = dt.isocalendar()
            week_key = (iso[0], iso[1])
            if week_key in seen_weeks:
                continue
            if dt.weekday() == day_of_week and dt in trading_set:
                seen_weeks.add(week_key)
                continue
            if dt.weekday() <= day_of_week:
                continue
            target = dt - datetime.timedelta(days=dt.weekday() - day_of_week)
            if target in trading_set:
                if week_key not in seen_weeks:
                    candidates.append(target)
                    seen_weeks.add(week_key)
            else:
                for offset in range(1, day_of_week + 1):
                    fallback = target - datetime.timedelta(days=offset)
                    if fallback in trading_set:
                        if week_key not in seen_weeks:
                            candidates.append(fallback)
                            seen_weeks.add(week_key)
                        break

        return sorted(set(candidates))
