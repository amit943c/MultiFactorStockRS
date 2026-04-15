"""Chart factory — every method produces a publication-quality Plotly figure."""

from __future__ import annotations

import calendar as cal
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from src.utils.helpers import drawdown_series, rolling_sharpe
from src.visualization.themes import (
    PALETTE_ACCENT,
    PALETTE_MAIN,
    get_theme,
)


def _apply_theme(fig: go.Figure, theme: str) -> go.Figure:
    """Apply a named theme to the figure layout, preserving axis type settings."""
    theme_dict = get_theme(theme)

    # Preserve axis type/tickvals/ticktext that charts set explicitly
    for axis_key in ("xaxis", "yaxis"):
        if axis_key in theme_dict:
            current = fig.layout[axis_key].to_plotly_json()
            preserve_keys = ("type", "tickvals", "ticktext", "categoryorder",
                             "categoryarray", "autorange", "tickangle")
            saved = {k: current[k] for k in preserve_keys if k in current}
            fig.update_layout(**{axis_key: theme_dict[axis_key]})
            if saved:
                fig.update_layout(**{axis_key: saved})
            theme_dict = {k: v for k, v in theme_dict.items() if k != axis_key}

    fig.update_layout(**theme_dict)
    return fig


class ChartFactory:
    """Static / class-method collection for every chart the platform needs."""

    # ------------------------------------------------------------------
    # 1. Equity curve
    # ------------------------------------------------------------------
    @staticmethod
    def equity_curve(
        equity: pd.Series,
        benchmark: pd.Series,
        title: str = "Cumulative Returns",
        theme: str = "light",
    ) -> go.Figure:
        # Align benchmark to the portfolio's start date so both begin at 1.0
        port_start = equity.index[0]
        bench_aligned = benchmark.loc[benchmark.index >= port_start].copy()
        if len(bench_aligned) > 0:
            bench_aligned = bench_aligned / bench_aligned.iloc[0]
        else:
            bench_aligned = benchmark.copy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Portfolio",
                line={"color": PALETTE_MAIN[0], "width": 2.2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned.values,
                mode="lines",
                name="Benchmark",
                line={"color": PALETTE_MAIN[2], "width": 1.8, "dash": "dot"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 2. Rolling drawdown
    # ------------------------------------------------------------------
    @staticmethod
    def rolling_drawdown(
        equity: pd.Series,
        title: str = "Drawdown from Peak",
        theme: str = "light",
    ) -> go.Figure:
        dd = drawdown_series(equity)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line={"color": PALETTE_ACCENT[0], "width": 1.5},
                fillcolor="rgba(192,57,43,0.25)",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 3. Rolling Sharpe
    # ------------------------------------------------------------------
    @staticmethod
    def rolling_sharpe_chart(
        returns: pd.Series,
        window: int = 63,
        title: str = "Rolling Sharpe Ratio",
        theme: str = "light",
    ) -> go.Figure:
        rs = rolling_sharpe(returns, window=window)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rs.index,
                y=rs.values,
                mode="lines",
                name=f"Sharpe ({window}d)",
                line={"color": PALETTE_MAIN[1], "width": 2},
            )
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="#7F8C8D", line_width=1,
            annotation_text="0",
            annotation_position="bottom right",
        )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio (ann.)",
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 4. Monthly returns heatmap
    # ------------------------------------------------------------------
    @staticmethod
    def monthly_returns_heatmap(
        returns: pd.Series,
        title: str = "Monthly Returns (%)",
        theme: str = "light",
    ) -> go.Figure:
        idx = pd.DatetimeIndex(returns.index)
        monthly = returns.copy()
        monthly.index = idx
        monthly = monthly.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame(
            {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
        )
        pivot = monthly_df.pivot_table(index="year", columns="month", values="ret")
        pivot.columns = [cal.month_abbr[m] for m in pivot.columns]

        z = pivot.values * 100
        text = [
            [f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
            for row in z
        ]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index],
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmid=0,
                colorbar={"title": "Return %", "ticksuffix": "%"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis={"title": "Month", "type": "category"},
            yaxis={"title": "Year", "type": "category", "autorange": "reversed"},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 5. Factor correlation heatmap
    # ------------------------------------------------------------------
    @staticmethod
    def factor_correlation_heatmap(
        factor_df: pd.DataFrame,
        factor_cols: list[str],
        title: str = "Factor Correlation Matrix",
        theme: str = "light",
    ) -> go.Figure:
        corr = factor_df[factor_cols].corr()
        z = corr.values
        labels = [c.replace("_", " ").title() for c in factor_cols]
        text = [[f"{v:.2f}" for v in row] for row in z]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar={"title": "Corr"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis={"type": "category", "tickangle": -45},
            yaxis={"type": "category", "autorange": "reversed"},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 6. Factor score distribution
    # ------------------------------------------------------------------
    @staticmethod
    def factor_score_distribution(
        factor_df: pd.DataFrame,
        date,
        top_n: int = 50,
        title: Optional[str] = None,
        theme: str = "light",
    ) -> go.Figure:
        date = pd.Timestamp(date)
        mask = factor_df["date"] == date
        subset = factor_df.loc[mask].nlargest(top_n, "composite_score")
        title = title or f"Composite Score Distribution — {date}"

        fig = go.Figure(
            go.Histogram(
                x=subset["composite_score"],
                nbinsx=20,
                marker_color=PALETTE_MAIN[0],
                marker_line={"color": "#FFFFFF", "width": 0.8},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Composite Score",
            yaxis_title="Count",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 7. Factor vs return scatter
    # ------------------------------------------------------------------
    @staticmethod
    def factor_vs_return_scatter(
        factor_df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        title: Optional[str] = None,
        theme: str = "light",
    ) -> go.Figure:
        clean = factor_df[[factor_col, return_col]].dropna()
        x = clean[factor_col].values
        y = clean[return_col].values
        _stat = np.asarray(sp_stats.spearmanr(x, y).statistic)
        rho = float(_stat) if _stat.ndim == 0 else float(_stat[0, 1])

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        title = title or f"{factor_col} vs {return_col}"
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker={"color": PALETTE_MAIN[0], "size": 3, "opacity": 0.45},
                name="Observations",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Trend (ρ={rho:.3f})",
                line={"color": PALETTE_ACCENT[0], "width": 2},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=factor_col,
            yaxis_title=return_col,
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 8. Sector exposure (stacked area)
    # ------------------------------------------------------------------
    @staticmethod
    def sector_exposure(
        holdings_history: dict,
        fundamentals_df: pd.DataFrame,
        title: str = "Sector Exposure Over Time",
        theme: str = "light",
    ) -> go.Figure:
        sector_map = dict(zip(fundamentals_df["ticker"], fundamentals_df["sector"]))
        records: list[dict] = []
        for dt, hdf in sorted(holdings_history.items()):
            for _, row in hdf.iterrows():
                records.append({
                    "date": dt,
                    "sector": sector_map.get(row["ticker"], "Unknown"),
                    "weight": row["weight"],
                })
        agg = (
            pd.DataFrame(records)
            .groupby(["date", "sector"])["weight"]
            .sum()
            .unstack(fill_value=0)
        )
        fig = go.Figure()
        for i, sector in enumerate(agg.columns):
            fig.add_trace(
                go.Scatter(
                    x=agg.index,
                    y=agg[sector],
                    mode="lines",
                    stackgroup="one",
                    name=sector,
                    line={"width": 0.5, "color": PALETTE_MAIN[i % len(PALETTE_MAIN)]},
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Weight",
            yaxis_tickformat=".0%",
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 9. Turnover over time
    # ------------------------------------------------------------------
    @staticmethod
    def turnover_over_time(
        turnover: pd.Series,
        title: str = "Portfolio Turnover at Rebalance",
        theme: str = "light",
    ) -> go.Figure:
        fig = go.Figure(
            go.Bar(
                x=turnover.index,
                y=turnover.values,
                marker_color=PALETTE_MAIN[3],
                marker_line={"color": PALETTE_MAIN[0], "width": 0.6},
            )
        )
        fig.add_hline(
            y=turnover.mean(),
            line_dash="dash",
            line_color=PALETTE_ACCENT[2],
            line_width=1.5,
            annotation_text=f"avg {turnover.mean():.1%}",
            annotation_position="top right",
        )
        fig.update_layout(
            title=title,
            xaxis_title="Rebalance Date",
            yaxis_title="One-Way Turnover",
            yaxis_tickformat=".0%",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 10. Top-ranked table
    # ------------------------------------------------------------------
    @staticmethod
    def top_ranked_table(
        ranked_df: pd.DataFrame,
        date,
        top_n: int = 20,
        title: Optional[str] = None,
        theme: str = "light",
    ) -> go.Figure:
        date = pd.Timestamp(date)
        mask = ranked_df["date"] == date
        subset = ranked_df.loc[mask].nlargest(top_n, "composite_score")

        display_cols = [
            c
            for c in [
                "ticker",
                "composite_score",
                "composite_rank",
                "return_1m",
                "return_3m",
                "rsi_14",
                "realized_vol_60d",
            ]
            if c in subset.columns
        ]
        values = [subset[c].tolist() for c in display_cols]

        fmt_map = {
            "composite_score": ".2f",
            "composite_rank": ".0f",
            "return_1m": ".2%",
            "return_3m": ".2%",
            "rsi_14": ".1f",
            "realized_vol_60d": ".2%",
        }
        formatted_values: list[list] = []
        for col, vals in zip(display_cols, values):
            if col in fmt_map:
                formatted_values.append([f"{v:{fmt_map[col]}}" if pd.notna(v) else "" for v in vals])
            else:
                formatted_values.append(vals)

        th = get_theme(theme)
        header_bg = PALETTE_MAIN[0]
        header_font = "#FFFFFF"
        cell_bg = th.get("plot_bgcolor", "#FAFBFC")
        cell_font = th["font"]["color"]

        title = title or f"Top {top_n} Ranked Stocks — {date}"
        fig = go.Figure(
            go.Table(
                header={
                    "values": [f"<b>{c}</b>" for c in display_cols],
                    "fill_color": header_bg,
                    "font": {"color": header_font, "size": 12},
                    "align": "center",
                    "line_color": "#FFFFFF",
                },
                cells={
                    "values": formatted_values,
                    "fill_color": cell_bg,
                    "font": {"color": cell_font, "size": 11},
                    "align": ["left"] + ["right"] * (len(display_cols) - 1),
                    "height": 26,
                },
            )
        )
        fig.update_layout(title=title, margin={"l": 20, "r": 20, "t": 50, "b": 10})
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 11. Portfolio composition (stacked bar)
    # ------------------------------------------------------------------
    @staticmethod
    def portfolio_composition(
        holdings_history: dict,
        max_tickers: int = 15,
        title: str = "Portfolio Composition at Rebalance",
        theme: str = "light",
    ) -> go.Figure:
        all_tickers: set[str] = set()
        for hdf in holdings_history.values():
            all_tickers.update(hdf["ticker"].tolist())

        avg_weights: dict[str, float] = {}
        for t in all_tickers:
            ws = []
            for hdf in holdings_history.values():
                row = hdf.loc[hdf["ticker"] == t, "weight"]
                ws.append(float(row.iloc[0]) if len(row) else 0.0)
            avg_weights[t] = np.mean(ws)

        top = sorted(avg_weights, key=avg_weights.get, reverse=True)[:max_tickers]  # type: ignore[arg-type]

        dates = sorted(holdings_history.keys())
        fig = go.Figure()
        for i, ticker in enumerate(top):
            weights_over_time = []
            for dt in dates:
                hdf = holdings_history[dt]
                row = hdf.loc[hdf["ticker"] == ticker, "weight"]
                weights_over_time.append(float(row.iloc[0]) if len(row) else 0.0)
            fig.add_trace(
                go.Bar(
                    x=[str(d) for d in dates],
                    y=weights_over_time,
                    name=ticker,
                    marker_color=PALETTE_MAIN[i % len(PALETTE_MAIN)],
                )
            )
        fig.update_layout(
            barmode="stack",
            title=title,
            xaxis_title="Rebalance Date",
            yaxis_title="Weight",
            yaxis_tickformat=".0%",
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 12. Factor heatmap (tickers x factors, coloured by z-score)
    # ------------------------------------------------------------------
    @staticmethod
    def factor_heatmap(
        factor_df: pd.DataFrame,
        date,
        tickers: list[str],
        factor_cols: list[str],
        title: Optional[str] = None,
        theme: str = "light",
    ) -> go.Figure:
        date = pd.Timestamp(date)
        mask = (factor_df["date"] == date) & (factor_df["ticker"].isin(tickers))
        subset = factor_df.loc[mask].set_index("ticker").reindex(tickers)

        raw = subset[factor_cols].astype(float)
        zscores = raw.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9))

        z = zscores.values
        labels = [c.replace("_", " ").title() for c in factor_cols]
        text = [[f"{v:.2f}" for v in row] for row in np.nan_to_num(z)]

        title = title or f"Factor Z-Scores — {date}"
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=tickers,
                text=text,
                texttemplate="%{text}",
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "Z-score"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis={"type": "category", "tickangle": -45},
            yaxis={"type": "category", "autorange": "reversed"},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 13. Rolling Information Coefficient
    # ------------------------------------------------------------------
    @staticmethod
    def rolling_ic_chart(
        ic_series: pd.Series,
        factor_name: str = "",
        theme: str = "light",
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ic_series.index,
                y=ic_series.values,
                mode="lines",
                name="IC",
                line={"color": PALETTE_MAIN[0], "width": 2},
            )
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="#7F8C8D", line_width=1,
        )
        fig.add_hrect(
            y0=-0.05, y1=0.05,
            fillcolor="rgba(127,140,141,0.12)",
            line_width=0,
            annotation_text="±0.05",
            annotation_position="top left",
        )
        label = f" — {factor_name}" if factor_name else ""
        fig.update_layout(
            title={"text": f"Rolling Information Coefficient{label}", "font": {"size": 18}},
            xaxis_title="Date",
            yaxis_title="IC",
            font={"size": 12},
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 14. Quantile returns
    # ------------------------------------------------------------------
    @staticmethod
    def quantile_returns_chart(
        quantile_df: pd.DataFrame,
        title: str = "Cumulative Returns by Score Quantile",
        theme: str = "light",
    ) -> go.Figure:
        diverging = [
            PALETTE_ACCENT[0],   # Q1 red
            "#E67E22",           # Q2 orange
            "#7F8C8D",           # Q3 neutral
            PALETTE_MAIN[1],     # Q4 teal
            PALETTE_ACCENT[1],   # Q5 green
        ]
        fig = go.Figure()
        for i, col in enumerate(quantile_df.columns):
            fig.add_trace(
                go.Scatter(
                    x=quantile_df.index,
                    y=quantile_df[col].values,
                    mode="lines",
                    name=col,
                    line={"color": diverging[i % len(diverging)], "width": 2},
                )
            )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            font={"size": 12},
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 15. Long-short spread
    # ------------------------------------------------------------------
    @staticmethod
    def long_short_spread_chart(
        spread: pd.Series,
        title: str = "Top − Bottom Quantile Spread",
        theme: str = "light",
    ) -> go.Figure:
        cum = (1 + spread).cumprod() - 1
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                fill="tozeroy",
                name="Spread",
                line={"color": PALETTE_MAIN[0], "width": 2},
                fillcolor="rgba(39,174,96,0.18)",
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Date",
            yaxis_title="Cumulative Spread Return",
            yaxis_tickformat=".1%",
            font={"size": 12},
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 16. Holdings overlap (Jaccard)
    # ------------------------------------------------------------------
    @staticmethod
    def holdings_overlap_chart(
        overlap: pd.Series,
        title: str = "Holdings Overlap (Jaccard)",
        theme: str = "light",
    ) -> go.Figure:
        fig = go.Figure(
            go.Bar(
                x=overlap.index,
                y=overlap.values,
                marker_color=PALETTE_MAIN[3],
                marker_line={"color": PALETTE_MAIN[0], "width": 0.6},
            )
        )
        fig.add_hline(
            y=overlap.mean(),
            line_dash="dash",
            line_color=PALETTE_ACCENT[2],
            line_width=1.5,
            annotation_text=f"mean {overlap.mean():.2f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Rebalance Date",
            yaxis_title="Jaccard Overlap",
            font={"size": 12},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 17. Composite score dispersion
    # ------------------------------------------------------------------
    @staticmethod
    def score_dispersion_chart(
        dispersion: pd.Series,
        title: str = "Composite Score Dispersion",
        theme: str = "light",
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dispersion.index,
                y=dispersion.values,
                mode="lines",
                name="Cross-Sectional Std Dev",
                line={"color": PALETTE_MAIN[6], "width": 2},
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Date",
            yaxis_title="Score Std Dev",
            font={"size": 12},
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 18. Sensitivity heatmap
    # ------------------------------------------------------------------
    @staticmethod
    def sensitivity_heatmap(
        sweep_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Sensitivity Analysis",
        theme: str = "light",
    ) -> go.Figure:
        pivot = sweep_df.pivot_table(
            index=y_col, columns=x_col, values="value", aggfunc="mean",
        )
        z = pivot.values
        text = [[f"{v:.3f}" for v in row] for row in np.nan_to_num(z)]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                colorbar={"title": "Metric"},
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis={"title": x_col, "type": "category"},
            yaxis={"title": y_col, "type": "category"},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 19. Annual returns bar
    # ------------------------------------------------------------------
    @staticmethod
    def yearly_returns_bar(
        yearly: pd.Series,
        title: str = "Annual Returns",
        theme: str = "light",
    ) -> go.Figure:
        colors = [
            PALETTE_ACCENT[1] if v >= 0 else PALETTE_ACCENT[0]
            for v in yearly.values
        ]
        fig = go.Figure(
            go.Bar(
                x=[str(y) for y in yearly.index],
                y=yearly.values,
                marker_color=colors,
                marker_line={"color": "#FFFFFF", "width": 0.6},
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Year",
            yaxis_title="Return",
            yaxis_tickformat=".0%",
            font={"size": 12},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 20. Performance by market regime
    # ------------------------------------------------------------------
    @staticmethod
    def regime_performance_chart(
        regime_stats: pd.DataFrame,
        title: str = "Performance by Market Regime",
        theme: str = "light",
    ) -> go.Figure:
        regimes = regime_stats.index.tolist()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=regime_stats["ann_return"],
                name="Ann. Return",
                marker_color=PALETTE_MAIN[0],
                text=[f"{v:.1%}" for v in regime_stats["ann_return"]],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=regime_stats["sharpe"],
                name="Sharpe",
                marker_color=PALETTE_MAIN[1],
                text=[f"{v:.2f}" for v in regime_stats["sharpe"]],
                textposition="outside",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Regime",
            yaxis={"title": "Ann. Return", "tickformat": ".0%", "side": "left"},
            yaxis2={
                "title": "Sharpe Ratio",
                "overlaying": "y",
                "side": "right",
            },
            barmode="group",
            font={"size": 12},
            legend={"x": 0.01, "y": 0.99},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 21. Drawdown episodes table
    # ------------------------------------------------------------------
    @staticmethod
    def drawdown_episodes_table(
        episodes_df: pd.DataFrame,
        title: str = "Drawdown Episodes",
        theme: str = "light",
    ) -> go.Figure:
        th = get_theme(theme)
        header_bg = PALETTE_MAIN[0]
        header_font = "#FFFFFF"
        cell_bg = th.get("plot_bgcolor", "#FAFBFC")
        cell_font = th["font"]["color"]

        cols = episodes_df.columns.tolist()
        formatted_values: list[list] = []
        for col in cols:
            vals = episodes_df[col].tolist()
            if col.lower() in ("max_dd", "max_drawdown", "drawdown"):
                formatted_values.append(
                    [f"{v:.2%}" if isinstance(v, (int, float)) else str(v) for v in vals]
                )
            elif col.lower() in ("duration", "days", "length"):
                formatted_values.append([str(v) for v in vals])
            elif col.lower() in ("start", "end", "trough", "peak", "recovery"):
                formatted_values.append(
                    [v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v) for v in vals]
                )
            else:
                formatted_values.append([str(v) for v in vals])

        fig = go.Figure(
            go.Table(
                header={
                    "values": [f"<b>{c}</b>" for c in cols],
                    "fill_color": header_bg,
                    "font": {"color": header_font, "size": 12},
                    "align": "center",
                    "line_color": "#FFFFFF",
                },
                cells={
                    "values": formatted_values,
                    "fill_color": cell_bg,
                    "font": {"color": cell_font, "size": 11},
                    "align": "center",
                    "height": 26,
                },
            )
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            margin={"l": 20, "r": 20, "t": 50, "b": 10},
        )
        return _apply_theme(fig, theme)

    # ------------------------------------------------------------------
    # 22. Before / after transaction costs
    # ------------------------------------------------------------------
    @staticmethod
    def before_after_costs_chart(
        equity_gross: pd.Series,
        equity_net: pd.Series,
        title: str = "Impact of Transaction Costs",
        theme: str = "light",
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_gross.index,
                y=equity_gross.values,
                mode="lines",
                name="Gross",
                line={"color": PALETTE_MAIN[2], "width": 2, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=equity_net.index,
                y=equity_net.values,
                mode="lines",
                name="Net",
                line={"color": PALETTE_MAIN[0], "width": 2.2},
            )
        )
        drag = equity_gross.iloc[-1] - equity_net.iloc[-1]
        drag_pct = drag / equity_gross.iloc[-1] * 100
        fig.add_annotation(
            x=equity_net.index[-1],
            y=equity_net.iloc[-1],
            text=f"Cost drag: {drag_pct:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=-60,
            ay=-30,
            font={"size": 12, "color": PALETTE_ACCENT[0]},
        )
        fig.update_layout(
            title={"text": title, "font": {"size": 18}},
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            font={"size": 12},
            hovermode="x unified",
        )
        return _apply_theme(fig, theme)
