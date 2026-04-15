"""Portfolio-level analytics — concentration, effective N, sector breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioAnalytics:
    """Stateless helpers for analysing a single-date holdings snapshot."""

    @staticmethod
    def concentration_ratio(holdings: pd.DataFrame, top_k: int = 5) -> float:
        """Sum of the *top_k* largest weights in the portfolio.

        Parameters
        ----------
        holdings:
            DataFrame with at least a ``weight`` column.
        top_k:
            Number of top positions to sum.

        Returns
        -------
        float
            Concentration ratio in [0, 1].
        """
        weights = holdings["weight"].sort_values(ascending=False)
        return float(weights.iloc[:top_k].sum())

    @staticmethod
    def effective_n(holdings: pd.DataFrame) -> float:
        """Herfindahl-based effective number of positions: ``1 / Σ wᵢ²``.

        A portfolio of *N* equal-weight positions returns *N*.
        """
        w = holdings["weight"].values.astype(float)
        hhi = float(np.sum(w ** 2))
        return 1.0 / hhi if hhi > 1e-12 else 0.0

    @staticmethod
    def sector_breakdown(
        holdings: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate portfolio weight by sector.

        Parameters
        ----------
        holdings:
            Must contain ``ticker`` and ``weight``.
        fundamentals:
            Must contain ``ticker`` and ``sector``.

        Returns
        -------
        pd.DataFrame
            Columns: ``sector``, ``weight``, ``n_stocks``, ``pct``.
        """
        sector_map = dict(zip(fundamentals["ticker"], fundamentals["sector"]))
        h = holdings.copy()
        h["sector"] = h["ticker"].map(sector_map).fillna("Unknown")

        agg = h.groupby("sector")["weight"].agg(["sum", "count"]).reset_index()
        agg.columns = ["sector", "weight", "n_stocks"]
        total = agg["weight"].sum()
        agg["pct"] = agg["weight"] / total if total > 0 else 0.0
        return agg.sort_values("weight", ascending=False).reset_index(drop=True)

    @staticmethod
    def weight_distribution_stats(holdings: pd.DataFrame) -> dict:
        """Descriptive statistics of the weight distribution.

        Returns
        -------
        dict
            Keys: ``min``, ``max``, ``mean``, ``median``, ``std``,
            ``n_positions``, ``effective_n``, ``top5_concentration``.
        """
        w = holdings["weight"]
        hhi = float(np.sum(w.values.astype(float) ** 2))
        return {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "median": float(w.median()),
            "std": float(w.std()),
            "n_positions": int(len(w)),
            "effective_n": 1.0 / hhi if hhi > 1e-12 else 0.0,
            "top5_concentration": float(w.nlargest(5).sum()),
        }
