"""Factor-level analytics — information coefficient, quantile returns, and autocorrelation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation, robust to varying scipy return types."""
    result = spearmanr(a, b)
    stat = np.asarray(result.statistic)
    if stat.ndim == 0:
        return float(stat)
    if stat.ndim == 2:
        return float(stat[0, 1])
    return float(stat.flat[0])


class FactorAnalytics:
    """Stateless toolkit for evaluating single-factor predictive power."""

    @staticmethod
    def factor_ic(
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str,
    ) -> pd.Series:
        """Cross-sectional rank IC (Spearman) between a factor and forward returns.

        Returns a Series indexed by date with the rank correlation for each
        cross-section.
        """

        def _rank_corr(group: pd.DataFrame) -> float:
            clean = group[[factor_col, forward_return_col]].dropna()
            if len(clean) < 5:
                return np.nan
            return _spearman_rho(clean[factor_col].values, clean[forward_return_col].values)

        try:
            ic = factor_df.groupby("date").apply(_rank_corr, include_groups=False)
        except TypeError:
            ic = factor_df.groupby("date").apply(_rank_corr)
        ic.name = f"IC_{factor_col}"
        return ic

    @staticmethod
    def factor_ic_summary(
        factor_df: pd.DataFrame,
        factor_cols: list[str],
        forward_return_col: str = "return_1m",
    ) -> pd.DataFrame:
        """Summary table of IC statistics for multiple factors.

        Returns a DataFrame with rows per factor and columns:
        ``mean_ic``, ``ic_std``, ``ic_ir``, ``hit_rate``, ``n_periods``.
        """
        rows: list[dict] = []
        for col in factor_cols:
            ic = FactorAnalytics.factor_ic(factor_df, col, forward_return_col)
            ic_clean = ic.dropna()
            mean_ic = float(ic_clean.mean()) if len(ic_clean) else 0.0
            ic_std = float(ic_clean.std()) if len(ic_clean) > 1 else 0.0
            ic_ir = mean_ic / ic_std if ic_std > 1e-12 else 0.0
            hit_rate = float((ic_clean > 0).mean()) if len(ic_clean) else 0.0
            rows.append({
                "factor": col,
                "mean_ic": mean_ic,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "hit_rate": hit_rate,
                "n_periods": len(ic_clean),
            })
        return pd.DataFrame(rows).set_index("factor")

    @staticmethod
    def factor_autocorrelation(
        factor_df: pd.DataFrame,
        factor_col: str,
        lag: int = 1,
    ) -> pd.Series:
        """Rank autocorrelation of factor scores across consecutive dates.

        For each date pair (t, t-lag), compute the Spearman correlation of the
        factor's cross-sectional ranks. High autocorrelation implies the factor
        is stable and turnover-friendly.
        """
        dates = sorted(factor_df["date"].unique())
        records: list[dict] = []
        for i in range(lag, len(dates)):
            d_curr = dates[i]
            d_prev = dates[i - lag]
            curr = factor_df.loc[factor_df["date"] == d_curr, ["ticker", factor_col]].set_index("ticker")
            prev = factor_df.loc[factor_df["date"] == d_prev, ["ticker", factor_col]].set_index("ticker")
            common = curr.index.intersection(prev.index)
            if len(common) < 5:
                continue
            rho = _spearman_rho(
                curr.loc[common, factor_col].values,
                prev.loc[common, factor_col].values,
            )
            records.append({"date": d_curr, "autocorr": rho})
        out = pd.DataFrame(records).set_index("date")["autocorr"]
        out.name = f"autocorr_{factor_col}_lag{lag}"
        return out

    @staticmethod
    def factor_quantile_returns(
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str = "return_1m",
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Average forward return by quantile bucket of the factor.

        Returns a DataFrame indexed by quantile (1 = lowest factor value,
        *n_quantiles* = highest) with columns ``mean_return``, ``count``.
        """

        def _assign_quantile(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group["quantile"] = pd.qcut(
                group[factor_col], q=n_quantiles, labels=False, duplicates="drop",
            ) + 1
            return group

        tagged = factor_df.dropna(subset=[factor_col, forward_return_col]).copy()
        tagged = tagged.groupby("date", group_keys=False).apply(_assign_quantile)

        summary = (
            tagged.groupby("quantile")[forward_return_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "mean_return"})
        )
        summary.index.name = "quantile"
        return summary

    # ------------------------------------------------------------------
    # Extended analytics
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_ic(
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str,
        window: int = 12,
    ) -> pd.Series:
        """Rolling-window IC (Spearman) over the last *window* cross-sections.

        Returns a date-indexed Series of rolling IC values.  Each value is the
        average rank-IC across the preceding *window* dates (inclusive).
        """
        ic = FactorAnalytics.factor_ic(factor_df, factor_col, forward_return_col)
        rolling = ic.rolling(window=window, min_periods=window).mean()
        rolling.name = f"rolling_IC_{factor_col}_w{window}"
        return rolling.dropna()

    @staticmethod
    def quantile_cumulative_returns(
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str = "return_1m",
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Cumulative return series for each factor-quantile bucket.

        For every date, stocks are assigned to quantiles by *factor_col*.  The
        mean forward return per quantile per date is then cumulated over time.
        Returns a DataFrame with date index and columns ``Q1`` … ``Q{n}``.
        """

        def _assign_quantile(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group["quantile"] = pd.qcut(
                group[factor_col], q=n_quantiles, labels=False, duplicates="drop",
            ) + 1
            return group

        tagged = factor_df.dropna(subset=[factor_col, forward_return_col]).copy()
        tagged = tagged.groupby("date", group_keys=False).apply(_assign_quantile)

        mean_ret = (
            tagged.groupby(["date", "quantile"])[forward_return_col]
            .mean()
            .unstack("quantile")
        )
        mean_ret.columns = [f"Q{int(c)}" for c in mean_ret.columns]
        cum_ret = (1 + mean_ret).cumprod() - 1
        return cum_ret

    @staticmethod
    def long_short_spread(
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str = "return_1m",
        n_quantiles: int = 5,
    ) -> pd.Series:
        """Cumulative long-short spread (top quantile minus bottom quantile).

        Returns a date-indexed Series of cumulative spread returns.
        """

        def _assign_quantile(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group["quantile"] = pd.qcut(
                group[factor_col], q=n_quantiles, labels=False, duplicates="drop",
            ) + 1
            return group

        tagged = factor_df.dropna(subset=[factor_col, forward_return_col]).copy()
        tagged = tagged.groupby("date", group_keys=False).apply(_assign_quantile)

        mean_ret = (
            tagged.groupby(["date", "quantile"])[forward_return_col]
            .mean()
            .unstack("quantile")
        )
        spread = mean_ret[n_quantiles] - mean_ret[1]
        cum_spread = (1 + spread).cumprod() - 1
        cum_spread.name = f"LS_spread_{factor_col}"
        return cum_spread

    @staticmethod
    def score_persistence(
        factor_df: pd.DataFrame,
        factor_col: str = "composite_score",
    ) -> pd.Series:
        """Week-over-week rank correlation of *factor_col*.

        For each consecutive date pair, computes Spearman correlation of the
        cross-sectional scores.  High values indicate stable, low-turnover
        rankings.  Returns a date-indexed Series.
        """
        dates = sorted(factor_df["date"].unique())
        records: list[dict] = []
        for i in range(1, len(dates)):
            d_curr, d_prev = dates[i], dates[i - 1]
            curr = (
                factor_df.loc[factor_df["date"] == d_curr, ["ticker", factor_col]]
                .set_index("ticker")
                .dropna()
            )
            prev = (
                factor_df.loc[factor_df["date"] == d_prev, ["ticker", factor_col]]
                .set_index("ticker")
                .dropna()
            )
            common = curr.index.intersection(prev.index)
            if len(common) < 5:
                continue
            rho = _spearman_rho(
                curr.loc[common, factor_col].values,
                prev.loc[common, factor_col].values,
            )
            records.append({"date": d_curr, "persistence": rho})
        out = pd.DataFrame(records).set_index("date")["persistence"]
        out.name = f"persistence_{factor_col}"
        return out

    @staticmethod
    def holdings_overlap(holdings_history: dict[str, set[str]]) -> pd.Series:
        """Jaccard similarity of holdings between consecutive rebalance dates.

        Parameters
        ----------
        holdings_history:
            Mapping of date (or date-string) to set of ticker symbols held on
            that date.

        Returns a date-indexed Series of overlap ratios (0 = no overlap,
        1 = identical).
        """
        sorted_dates = sorted(holdings_history.keys())
        records: list[dict] = []
        for i in range(1, len(sorted_dates)):
            d_curr, d_prev = sorted_dates[i], sorted_dates[i - 1]
            curr_set = set(holdings_history[d_curr])
            prev_set = set(holdings_history[d_prev])
            union = curr_set | prev_set
            if not union:
                continue
            jaccard = len(curr_set & prev_set) / len(union)
            records.append({"date": d_curr, "overlap": jaccard})
        out = pd.DataFrame(records).set_index("date")["overlap"]
        out.name = "holdings_overlap"
        return out

    @staticmethod
    def ranking_dispersion(
        factor_df: pd.DataFrame,
        score_col: str = "composite_score",
    ) -> pd.Series:
        """Cross-sectional standard deviation of *score_col* per date.

        Higher dispersion indicates greater differentiation among stocks.
        Returns a date-indexed Series.
        """
        dispersion = factor_df.groupby("date")[score_col].std()
        dispersion.name = f"dispersion_{score_col}"
        return dispersion

    @staticmethod
    def factor_decay(
        factor_df: pd.DataFrame,
        factor_col: str,
        max_lag: int = 10,
    ) -> pd.DataFrame:
        """IC at increasing forward horizons (1 … *max_lag*).

        Shows how the factor's predictive power decays as the forecast
        horizon lengthens.  Returns a DataFrame with ``lag`` as index and
        ``ic`` as column.
        """
        dates = sorted(factor_df["date"].unique())
        results: list[dict] = []
        for lag in range(1, max_lag + 1):
            ic_values: list[float] = []
            for i in range(len(dates) - lag):
                d_now = dates[i]
                d_fwd = dates[i + lag]
                now = (
                    factor_df.loc[factor_df["date"] == d_now, ["ticker", factor_col]]
                    .set_index("ticker")
                )
                fwd_ret = (
                    factor_df.loc[factor_df["date"] == d_fwd, ["ticker", factor_col]]
                    .set_index("ticker")
                )
                common = now.index.intersection(fwd_ret.index)
                if len(common) < 5:
                    continue
                rho = _spearman_rho(
                    now.loc[common, factor_col].values,
                    fwd_ret.loc[common, factor_col].values,
                )
                ic_values.append(rho)
            mean_ic = float(np.nanmean(ic_values)) if ic_values else np.nan
            results.append({"lag": lag, "ic": mean_ic})
        return pd.DataFrame(results).set_index("lag")

    @staticmethod
    def top_holdings_stability(
        factor_df: pd.DataFrame,
        top_n: int = 20,
        score_col: str = "composite_score",
    ) -> pd.Series:
        """Fraction of the top-*top_n* stocks that remain in the top-*top_n*
        between consecutive dates.

        Returns a date-indexed Series of retention rates (0 to 1).
        """
        dates = sorted(factor_df["date"].unique())
        records: list[dict] = []
        prev_top: set[str] | None = None
        for d in dates:
            snap = factor_df.loc[factor_df["date"] == d].nlargest(top_n, score_col)
            curr_top = set(snap["ticker"])
            if prev_top is not None and curr_top:
                retention = len(curr_top & prev_top) / max(len(curr_top), 1)
                records.append({"date": d, "retention": retention})
            prev_top = curr_top
        out = pd.DataFrame(records).set_index("date")["retention"]
        out.name = f"top{top_n}_retention"
        return out
