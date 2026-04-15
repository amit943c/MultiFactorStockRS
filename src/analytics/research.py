"""Portfolio research extensions — sensitivity, drawdown episodes, calendar, regimes."""

from __future__ import annotations

import copy
import datetime
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceAnalyzer
from src.portfolio.construction import PortfolioConstructor
from src.portfolio.rebalance import RebalanceEngine

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class DrawdownEpisode:
    """A single drawdown episode."""

    start: datetime.date
    trough: datetime.date
    end: datetime.date | None  # None if not yet recovered
    depth: float  # max drawdown (negative)
    duration_days: int
    recovery_days: int | None


class DrawdownAnalyzer:
    """Identify and characterize drawdown episodes from an equity curve."""

    @staticmethod
    def find_episodes(
        equity_curve: pd.Series,
        threshold: float = -0.05,
    ) -> list[DrawdownEpisode]:
        """Find all drawdown episodes exceeding *threshold*.

        An episode starts when the drawdown from peak drops below *threshold*
        and ends when equity recovers to a new high (or remains open if still
        underwater at the end of the series).
        """
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak

        dates = [
            d.date() if isinstance(d, pd.Timestamp) else d for d in dd.index
        ]

        episodes: list[DrawdownEpisode] = []
        in_episode = False
        ep_start: datetime.date | None = None
        ep_trough_date: datetime.date | None = None
        ep_trough_val: float = 0.0

        for i, (date, drawdown) in enumerate(zip(dates, dd.values)):
            if not in_episode:
                if drawdown < threshold:
                    in_episode = True
                    ep_start = date
                    ep_trough_date = date
                    ep_trough_val = drawdown
            else:
                if drawdown < ep_trough_val:
                    ep_trough_val = drawdown
                    ep_trough_date = date

                if drawdown >= 0.0:
                    duration = (date - ep_start).days
                    recovery = (date - ep_trough_date).days
                    episodes.append(
                        DrawdownEpisode(
                            start=ep_start,
                            trough=ep_trough_date,
                            end=date,
                            depth=ep_trough_val,
                            duration_days=duration,
                            recovery_days=recovery,
                        )
                    )
                    in_episode = False

        if in_episode and ep_start is not None:
            last_date = dates[-1]
            duration = (last_date - ep_start).days
            episodes.append(
                DrawdownEpisode(
                    start=ep_start,
                    trough=ep_trough_date,
                    end=None,
                    depth=ep_trough_val,
                    duration_days=duration,
                    recovery_days=None,
                )
            )

        logger.info(
            "Found %d drawdown episodes exceeding %.1f%% threshold",
            len(episodes),
            threshold * 100,
        )
        return episodes

    @staticmethod
    def episodes_to_dataframe(episodes: list[DrawdownEpisode]) -> pd.DataFrame:
        """Convert episodes to a summary DataFrame."""
        rows = [
            {
                "start": ep.start,
                "trough": ep.trough,
                "end": ep.end,
                "depth_pct": ep.depth * 100,
                "duration_days": ep.duration_days,
                "recovery_days": ep.recovery_days,
            }
            for ep in episodes
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def worst_episodes(
        equity_curve: pd.Series,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Return the *top_n* worst drawdown episodes (by depth)."""
        episodes = DrawdownAnalyzer.find_episodes(equity_curve, threshold=-0.01)
        episodes.sort(key=lambda ep: ep.depth)
        worst = episodes[:top_n]
        return DrawdownAnalyzer.episodes_to_dataframe(worst)


class CalendarAnalyzer:
    """Calendar-based return analysis."""

    @staticmethod
    def monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
        """Pivot monthly returns into a *year x month* DataFrame (values in %).

        Compounds daily returns within each calendar month.
        """
        idx = pd.DatetimeIndex(daily_returns.index)
        dr = daily_returns.copy()
        dr.index = idx

        monthly = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        table = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values * 100,
            }
        )
        pivot = table.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            datetime.date(2000, m, 1).strftime("%b") for m in pivot.columns
        ]
        return pivot

    @staticmethod
    def yearly_returns(daily_returns: pd.Series) -> pd.Series:
        """Compound daily returns into yearly returns."""
        idx = pd.DatetimeIndex(daily_returns.index)
        dr = daily_returns.copy()
        dr.index = idx

        yearly = dr.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        yearly.index = yearly.index.year
        yearly.index.name = "year"
        yearly.name = "annual_return"
        return yearly

    @staticmethod
    def best_worst_periods(
        daily_returns: pd.Series,
        n: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Return the *n* best/worst months and days.

        Returns
        -------
        dict with keys ``best_months``, ``worst_months``, ``best_days``,
        ``worst_days``, each a DataFrame with date and return columns.
        """
        idx = pd.DatetimeIndex(daily_returns.index)
        dr = daily_returns.copy()
        dr.index = idx

        monthly = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        def _to_df(series: pd.Series) -> pd.DataFrame:
            df = series.reset_index()
            df.columns = ["date", "return"]
            return df

        best_months = _to_df(monthly.nlargest(n))
        worst_months = _to_df(monthly.nsmallest(n))
        best_days = _to_df(dr.nlargest(n))
        worst_days = _to_df(dr.nsmallest(n))

        return {
            "best_months": best_months,
            "worst_months": worst_months,
            "best_days": best_days,
            "worst_days": worst_days,
        }

    @staticmethod
    def monthly_stats(daily_returns: pd.Series) -> pd.DataFrame:
        """For each calendar month (Jan–Dec), compute aggregate statistics.

        Columns: ``avg_return``, ``win_rate``, ``best_year``, ``worst_year``.
        """
        idx = pd.DatetimeIndex(daily_returns.index)
        dr = daily_returns.copy()
        dr.index = idx

        monthly = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        months = monthly.index.month
        years = monthly.index.year

        records: list[dict[str, Any]] = []
        for m in range(1, 13):
            mask = months == m
            subset = monthly[mask]
            if subset.empty:
                continue
            subset_years = years[mask]
            records.append(
                {
                    "month": datetime.date(2000, m, 1).strftime("%b"),
                    "avg_return": float(subset.mean()),
                    "win_rate": float((subset > 0).mean()),
                    "best_year": int(subset_years[subset.argmax()]),
                    "worst_year": int(subset_years[subset.argmin()]),
                }
            )

        return pd.DataFrame(records).set_index("month")


class SensitivityAnalyzer:
    """Run parameter sweeps and compare performance."""

    @staticmethod
    def _run_pipeline(
        factor_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        cfg: dict,
    ) -> dict[str, float]:
        """Execute the full portfolio→rebalance→backtest→stats pipeline."""
        constructor = PortfolioConstructor(cfg)
        engine = RebalanceEngine(constructor)
        reb_history = engine.run(factor_df, prices_df, cfg)

        bt = BacktestEngine(cfg)
        result = bt.run(reb_history, prices_df)

        stats = PerformanceAnalyzer.compute_stats(
            result.equity_curve,
            result.daily_returns,
            turnover=result.turnover,
        )
        return stats

    @staticmethod
    def sweep_top_n(
        factor_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        cfg: dict,
        top_n_values: list[int] | None = None,
    ) -> pd.DataFrame:
        """For each *top_n* value, run the full pipeline and collect stats.

        Returns a DataFrame with columns:
        ``top_n, cagr, sharpe, max_dd, turnover``.
        """
        if top_n_values is None:
            top_n_values = [5, 10, 15, 20, 30, 50]

        rows: list[dict[str, float]] = []
        for n in top_n_values:
            logger.info("Sensitivity sweep: top_n=%d", n)
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.setdefault("portfolio", {})["top_n"] = n

            try:
                stats = SensitivityAnalyzer._run_pipeline(
                    factor_df, prices_df, cfg_copy
                )
                rows.append(
                    {
                        "top_n": n,
                        "cagr": stats.get("cagr", np.nan),
                        "sharpe": stats.get("sharpe_ratio", np.nan),
                        "max_dd": stats.get("max_drawdown", np.nan),
                        "turnover": stats.get("avg_turnover", np.nan),
                    }
                )
            except Exception:
                logger.exception("Sweep failed for top_n=%d", n)
                rows.append(
                    {
                        "top_n": n,
                        "cagr": np.nan,
                        "sharpe": np.nan,
                        "max_dd": np.nan,
                        "turnover": np.nan,
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def sweep_rebalance_freq(
        factor_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        cfg: dict,
        frequencies: list[str] | None = None,
    ) -> pd.DataFrame:
        """For each rebalance frequency, run the pipeline and compare.

        Returns a DataFrame with columns:
        ``frequency, cagr, sharpe, max_dd, turnover``.
        """
        if frequencies is None:
            frequencies = ["weekly", "monthly"]

        rows: list[dict[str, Any]] = []
        for freq in frequencies:
            logger.info("Sensitivity sweep: frequency=%s", freq)
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.setdefault("rebalance", {})["frequency"] = freq

            try:
                stats = SensitivityAnalyzer._run_pipeline(
                    factor_df, prices_df, cfg_copy
                )
                rows.append(
                    {
                        "frequency": freq,
                        "cagr": stats.get("cagr", np.nan),
                        "sharpe": stats.get("sharpe_ratio", np.nan),
                        "max_dd": stats.get("max_drawdown", np.nan),
                        "turnover": stats.get("avg_turnover", np.nan),
                    }
                )
            except Exception:
                logger.exception("Sweep failed for frequency=%s", freq)
                rows.append(
                    {
                        "frequency": freq,
                        "cagr": np.nan,
                        "sharpe": np.nan,
                        "max_dd": np.nan,
                        "turnover": np.nan,
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def sweep_transaction_costs(
        factor_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        cfg: dict,
        cost_values: list[int] | None = None,
    ) -> pd.DataFrame:
        """For each transaction cost (bps), run the pipeline and compare.

        Returns a DataFrame with columns:
        ``cost_bps, cagr, sharpe, max_dd, total_return``.
        """
        if cost_values is None:
            cost_values = [0, 5, 10, 20, 30, 50]

        rows: list[dict[str, float]] = []
        for bps in cost_values:
            logger.info("Sensitivity sweep: transaction_cost_bps=%d", bps)
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.setdefault("portfolio", {})["transaction_cost_bps"] = bps

            try:
                stats = SensitivityAnalyzer._run_pipeline(
                    factor_df, prices_df, cfg_copy
                )
                rows.append(
                    {
                        "cost_bps": bps,
                        "cagr": stats.get("cagr", np.nan),
                        "sharpe": stats.get("sharpe_ratio", np.nan),
                        "max_dd": stats.get("max_drawdown", np.nan),
                        "total_return": stats.get("total_return", np.nan),
                    }
                )
            except Exception:
                logger.exception(
                    "Sweep failed for transaction_cost_bps=%d", bps
                )
                rows.append(
                    {
                        "cost_bps": bps,
                        "cagr": np.nan,
                        "sharpe": np.nan,
                        "max_dd": np.nan,
                        "total_return": np.nan,
                    }
                )

        return pd.DataFrame(rows)


class RegimeAnalyzer:
    """Simple market regime classification based on trend and volatility."""

    @staticmethod
    def classify_regimes(
        benchmark_prices: pd.Series,
        vol_window: int = 60,
        trend_window: int = 200,
    ) -> pd.Series:
        """Classify each date into a market regime.

        Regimes
        -------
        - ``bull_low_vol``:  price above MA(*trend_window*), vol below median
        - ``bull_high_vol``: price above MA(*trend_window*), vol above median
        - ``bear_low_vol``:  price below MA(*trend_window*), vol below median
        - ``bear_high_vol``: price below MA(*trend_window*), vol above median

        Parameters
        ----------
        benchmark_prices:
            Date-indexed price series (e.g. SPY adj_close).
        vol_window:
            Rolling window for realised volatility (trading days).
        trend_window:
            Simple moving average window for trend detection.

        Returns
        -------
        pd.Series
            Date-indexed regime labels.
        """
        prices = benchmark_prices.copy()
        prices.index = pd.DatetimeIndex(prices.index)

        ma = prices.rolling(window=trend_window, min_periods=trend_window).mean()
        daily_ret = prices.pct_change()
        vol = daily_ret.rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(
            _TRADING_DAYS_PER_YEAR
        )

        valid = ma.notna() & vol.notna()
        prices = prices[valid]
        ma = ma[valid]
        vol = vol[valid]

        vol_median = vol.median()
        is_bull = prices > ma
        is_low_vol = vol <= vol_median

        conditions = [
            is_bull & is_low_vol,
            is_bull & ~is_low_vol,
            ~is_bull & is_low_vol,
            ~is_bull & ~is_low_vol,
        ]
        labels = ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]

        regime = pd.Series(
            np.select(conditions, labels, default="unknown"),
            index=prices.index,
            name="regime",
        )

        for label in labels:
            pct = (regime == label).mean() * 100
            logger.info("Regime '%s': %.1f%% of observations", label, pct)

        return regime

    @staticmethod
    def regime_performance(
        daily_returns: pd.Series,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """Compute performance statistics broken down by regime.

        For each regime: annualised return, volatility, Sharpe, max drawdown,
        and percentage of time spent in that regime.
        """
        dr = daily_returns.copy()
        dr.index = pd.DatetimeIndex(dr.index)
        regimes = regimes.copy()
        regimes.index = pd.DatetimeIndex(regimes.index)

        common = dr.index.intersection(regimes.index)
        dr = dr.reindex(common)
        regimes = regimes.reindex(common)

        total_days = len(common)
        records: list[dict[str, Any]] = []

        for label in sorted(regimes.unique()):
            mask = regimes == label
            subset = dr[mask]
            n = len(subset)
            if n < 2:
                continue

            ann_ret = float(subset.mean() * _TRADING_DAYS_PER_YEAR)
            ann_vol = float(subset.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
            sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

            eq = (1 + subset).cumprod()
            peak = eq.cummax()
            dd = (eq - peak) / peak
            max_dd = float(dd.min())

            records.append(
                {
                    "regime": label,
                    "ann_return": ann_ret,
                    "ann_volatility": ann_vol,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "pct_of_time": n / total_days * 100,
                }
            )

        return pd.DataFrame(records).set_index("regime")

    @staticmethod
    def regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
        """Compute transition probabilities between regimes.

        Entry *(i, j)* is the probability of moving from regime *i* to
        regime *j* on the next trading day.
        """
        labels = sorted(regimes.unique())
        vals = regimes.values

        counts: dict[str, dict[str, int]] = {
            a: {b: 0 for b in labels} for a in labels
        }

        for i in range(len(vals) - 1):
            counts[vals[i]][vals[i + 1]] += 1

        matrix = pd.DataFrame(counts).T.reindex(index=labels, columns=labels)

        row_sums = matrix.sum(axis=1)
        transition = matrix.div(row_sums, axis=0).fillna(0.0)
        transition.index.name = "from"
        transition.columns.name = "to"

        return transition
