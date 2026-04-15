"""Composite scoring — combine individual factor z-scores into a single rank."""

from __future__ import annotations

import logging

import pandas as pd

from src.utils.helpers import cross_sectional_zscore, winsorize

logger = logging.getLogger(__name__)


class CompositeScorer:
    """Weight and combine normalised factor columns into a composite score.

    Parameters
    ----------
    factor_weights:
        Mapping of factor column name to its weight, e.g.
        ``{"return_1m": 0.15, "dist_ma50": 0.10, ...}``.
    factor_directions:
        Mapping of factor column name to its directionality, one of
        ``"higher_is_better"`` or ``"lower_is_better"``.  Factors with
        ``"lower_is_better"`` have their z-score sign flipped so that a
        *higher* composite score is always preferable.
    """

    def __init__(
        self,
        factor_weights: dict[str, float],
        factor_directions: dict[str, str],
    ) -> None:
        self._weights = factor_weights
        self._directions = factor_directions
        logger.info(
            "CompositeScorer initialised with %d factor(s)", len(factor_weights)
        )

    def compute_composite(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Build ``composite_score`` and ``composite_rank`` columns.

        Parameters
        ----------
        factor_df:
            Wide DataFrame containing ``date, ticker`` plus every factor
            column referenced in *factor_weights*.

        Returns
        -------
        DataFrame
            The input *factor_df* augmented with ``composite_score`` and
            ``composite_rank`` columns.
        """
        df = factor_df.copy()

        weighted_sum = pd.Series(0.0, index=df.index)
        active_weight = 0.0

        for col, weight in self._weights.items():
            if col not in df.columns:
                logger.warning("Factor column '%s' not in DataFrame — skipping", col)
                continue

            df[col] = winsorize(df[col])

            z = cross_sectional_zscore(df, col)

            direction = self._directions.get(col, "higher_is_better")
            if direction == "lower_is_better":
                z = -z

            weighted_sum += z * weight
            active_weight += weight
            logger.debug(
                "Factor %-20s  weight=%.3f  dir=%s", col, weight, direction
            )

        if active_weight > 0:
            weighted_sum /= active_weight

        df["composite_score"] = weighted_sum

        df["composite_rank"] = df.groupby("date")["composite_score"].rank(
            ascending=False, method="min"
        )

        logger.info(
            "Composite scoring complete — %d rows, active weight sum=%.3f",
            len(df),
            active_weight,
        )
        return df
