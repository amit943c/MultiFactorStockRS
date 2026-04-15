"""Factor neutralization and alternative weighting schemes."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.helpers import winsorize, cross_sectional_zscore

logger = logging.getLogger(__name__)


class FactorNeutralizer:
    """Sector-neutral and market-cap-neutral factor transformations."""

    @staticmethod
    def sector_neutral_zscore(
        factor_df: pd.DataFrame,
        factor_col: str,
        sector_col: str = "sector",
    ) -> pd.Series:
        """Z-score factor values within each sector on each date.

        For each (date, sector) group, compute z-score of factor_col.
        This removes sector bias so that the factor ranks stocks
        within their sector rather than across sectors.
        """
        date_col = "date" if "date" in factor_df.columns else factor_df.index.names[0]
        group_keys = [date_col, sector_col] if date_col in factor_df.columns else [factor_df.index.get_level_values(0), sector_col]

        if date_col in factor_df.columns:
            grouped = factor_df.groupby([date_col, sector_col])[factor_col]
        else:
            grouped = factor_df.groupby([factor_df.index.get_level_values(0), sector_col])[factor_col]

        result = grouped.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        logger.debug("Sector-neutral z-score computed for '%s'", factor_col)
        return result

    @staticmethod
    def neutralize_all(
        factor_df: pd.DataFrame,
        factor_cols: list[str],
        sector_col: str = "sector",
    ) -> pd.DataFrame:
        """Apply sector-neutral z-scoring to all factor columns.
        Returns df with new columns named {col}_neutral.
        """
        df = factor_df.copy()
        for col in factor_cols:
            if col not in df.columns:
                logger.warning("Factor column '%s' not found — skipping neutralization", col)
                continue
            df[f"{col}_neutral"] = FactorNeutralizer.sector_neutral_zscore(df, col, sector_col)
        logger.info("Neutralized %d factor columns by sector", len(factor_cols))
        return df


class FactorWeighter:
    """Alternative factor weighting schemes beyond static config weights."""

    @staticmethod
    def equal_weights(factor_cols: list[str]) -> dict[str, float]:
        """Equal weight across all factor columns."""
        n = len(factor_cols)
        if n == 0:
            return {}
        w = 1.0 / n
        return {col: w for col in factor_cols}

    @staticmethod
    def ic_weighted(
        factor_df: pd.DataFrame,
        factor_cols: list[str],
        forward_return_col: str = "return_1m",
        lookback_periods: int = 52,
    ) -> dict[str, float]:
        """Weight factors proportional to their recent information coefficient.

        Compute rolling IC for each factor over the last lookback_periods,
        take the absolute mean IC, and normalize to sum to 1.
        Factors with higher IC get more weight.
        """
        date_col = "date" if "date" in factor_df.columns else None
        if date_col:
            dates = factor_df[date_col].unique()
        else:
            dates = factor_df.index.get_level_values(0).unique()

        recent_dates = sorted(dates)[-lookback_periods:]

        if date_col:
            recent = factor_df[factor_df[date_col].isin(recent_dates)]
        else:
            recent = factor_df[factor_df.index.get_level_values(0).isin(recent_dates)]

        abs_ics: dict[str, float] = {}
        for col in factor_cols:
            if col not in recent.columns or forward_return_col not in recent.columns:
                abs_ics[col] = 0.0
                continue

            if date_col:
                ic_by_date = recent.groupby(date_col).apply(
                    lambda g: g[col].corr(g[forward_return_col]), include_groups=False
                )
            else:
                ic_by_date = recent.groupby(level=0).apply(
                    lambda g: g[col].corr(g[forward_return_col]), include_groups=False
                )

            abs_ics[col] = float(np.abs(ic_by_date).mean()) if not ic_by_date.empty else 0.0

        total = sum(abs_ics.values())
        if total < 1e-12:
            logger.warning("All ICs near zero — falling back to equal weights")
            return FactorWeighter.equal_weights(factor_cols)

        weights = {col: v / total for col, v in abs_ics.items()}
        logger.info("IC-weighted factors: %s", {k: round(v, 4) for k, v in weights.items()})
        return weights

    @staticmethod
    def inverse_correlation_weights(
        factor_df: pd.DataFrame,
        factor_cols: list[str],
    ) -> dict[str, float]:
        """Weight factors inversely proportional to their average pairwise correlation.

        Factors that are less correlated with others get more weight,
        providing better diversification of signal sources.

        Algorithm:
        1. Compute pairwise correlation matrix of factors
        2. For each factor, compute average absolute correlation with all others
        3. Weight = 1 / avg_corr, then normalize
        """
        if len(factor_cols) < 2:
            return FactorWeighter.equal_weights(factor_cols)

        available = [c for c in factor_cols if c in factor_df.columns]
        if len(available) < 2:
            return FactorWeighter.equal_weights(factor_cols)

        corr_matrix = factor_df[available].corr().abs()

        inv_weights: dict[str, float] = {}
        for col in available:
            others = [c for c in available if c != col]
            avg_corr = corr_matrix.loc[col, others].mean()
            inv_weights[col] = 1.0 / (avg_corr + 1e-9)

        total = sum(inv_weights.values())
        weights = {col: v / total for col, v in inv_weights.items()}

        for col in factor_cols:
            if col not in weights:
                weights[col] = 0.0

        logger.info(
            "Inverse-correlation weights: %s",
            {k: round(v, 4) for k, v in weights.items()},
        )
        return weights

    @staticmethod
    def apply_weights(
        scheme: str,
        factor_df: pd.DataFrame,
        factor_cols: list[str],
        static_weights: dict[str, float] | None = None,
        forward_return_col: str = "return_1m",
    ) -> dict[str, float]:
        """Dispatch to the appropriate weighting scheme.

        scheme: "static" | "equal" | "ic_weighted" | "inverse_correlation"
        """
        if scheme == "static":
            if static_weights is None:
                raise ValueError("static_weights must be provided when scheme='static'")
            return static_weights
        elif scheme == "equal":
            return FactorWeighter.equal_weights(factor_cols)
        elif scheme == "ic_weighted":
            return FactorWeighter.ic_weighted(
                factor_df, factor_cols, forward_return_col=forward_return_col
            )
        elif scheme == "inverse_correlation":
            return FactorWeighter.inverse_correlation_weights(factor_df, factor_cols)
        else:
            raise ValueError(
                f"Unknown weighting scheme '{scheme}'. "
                "Choose from: static, equal, ic_weighted, inverse_correlation"
            )
