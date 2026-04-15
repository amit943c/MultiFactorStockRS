"""Markdown report generation from backtest results."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Build human-readable markdown reports from backtest outputs."""

    @staticmethod
    def generate_summary_table(stats: dict[str, float]) -> pd.DataFrame:
        """Format raw stats into a two-column presentation DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``Metric``, ``Value``.
        """
        _pct = lambda v: f"{v * 100:+.2f}%"
        _pct_u = lambda v: f"{v * 100:.2f}%"
        _ratio = lambda v: f"{v:.3f}"

        label_map: dict[str, tuple[str, Any]] = {
            "total_return": ("Total Return", _pct),
            "cagr": ("CAGR", _pct),
            "annualized_volatility": ("Annualized Volatility", _pct_u),
            "sharpe_ratio": ("Sharpe Ratio", _ratio),
            "sortino_ratio": ("Sortino Ratio", _ratio),
            "max_drawdown": ("Max Drawdown", _pct),
            "calmar_ratio": ("Calmar Ratio", _ratio),
            "win_rate_daily": ("Daily Win Rate", _pct_u),
            "win_rate_weekly": ("Weekly Win Rate", _pct_u),
            "avg_turnover": ("Avg Turnover (one-way)", _pct_u),
            "excess_return_annualized": ("Excess Return (ann.)", _pct),
            "information_ratio": ("Information Ratio", _ratio),
        }

        rows: list[dict[str, str]] = []
        for key, (label, fmt) in label_map.items():
            if key in stats:
                rows.append({"Metric": label, "Value": fmt(stats[key])})
        return pd.DataFrame(rows)

    @classmethod
    def generate_markdown(
        cls,
        stats: dict[str, float],
        backtest_result: Any,
        cfg: dict,
        output_path: str,
    ) -> None:
        """Write a comprehensive markdown report to *output_path*.

        Sections
        --------
        1. Performance summary table
        2. Key highlights
        3. Configuration used
        4. Last rebalance holdings
        5. Best / worst periods
        6. Assumptions & caveats
        """
        lines: list[str] = []
        _w = lines.append

        _w(f"# Multi-Factor Strategy — Backtest Report")
        _w(f"*Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}*\n")

        # --- 1. Performance summary ---
        _w("## Performance Summary\n")
        summary = cls.generate_summary_table(stats)
        _w(_df_to_md_table(summary))
        _w("")

        # --- 2. Key highlights ---
        _w("## Key Highlights\n")
        cagr = stats.get("cagr", 0)
        sharpe = stats.get("sharpe_ratio", 0)
        mdd = stats.get("max_drawdown", 0)
        _w(f"- **CAGR**: {cagr * 100:+.2f}%")
        _w(f"- **Sharpe Ratio**: {sharpe:.3f}")
        _w(f"- **Max Drawdown**: {mdd * 100:+.2f}%")
        if "sortino_ratio" in stats:
            _w(f"- **Sortino Ratio**: {stats['sortino_ratio']:.3f}")
        if "calmar_ratio" in stats:
            _w(f"- **Calmar Ratio**: {stats['calmar_ratio']:.3f}")
        _w("")

        # --- 3. Configuration ---
        _w("## Configuration\n")
        _w("```yaml")
        _write_dict_yaml(cfg, lines, indent=0)
        _w("```\n")

        # --- 4. Last rebalance holdings ---
        _w("## Holdings at Last Rebalance\n")
        if backtest_result.rebalance_dates:
            last_date = backtest_result.rebalance_dates[-1]
            _w(f"*Date: {last_date}*\n")
            holdings = backtest_result.holdings_history.get(last_date)
            if holdings is not None and len(holdings) > 0:
                top = holdings.nlargest(20, "weight")
                display = top[["ticker", "weight"]].copy()
                display["weight"] = display["weight"].map(lambda w: f"{w:.2%}")
                _w(_df_to_md_table(display))
            else:
                _w("*No holdings data available.*")
        else:
            _w("*No rebalance dates recorded.*")
        _w("")

        # --- 5. Best / worst performing periods ---
        _w("## Top / Bottom Performing Periods\n")
        dr = backtest_result.daily_returns
        if len(dr) > 0:
            idx = pd.DatetimeIndex(dr.index)
            monthly = dr.copy()
            monthly.index = idx
            monthly = monthly.resample("ME").apply(lambda x: (1 + x).prod() - 1)

            if len(monthly) >= 3:
                best = monthly.nlargest(3)
                worst = monthly.nsmallest(3)
                _w("**Best months:**\n")
                _w("| Month | Return |")
                _w("|-------|--------|")
                for d, v in best.items():
                    _w(f"| {d:%Y-%m} | {v * 100:+.2f}% |")
                _w("")
                _w("**Worst months:**\n")
                _w("| Month | Return |")
                _w("|-------|--------|")
                for d, v in worst.items():
                    _w(f"| {d:%Y-%m} | {v * 100:+.2f}% |")
            else:
                _w("*Insufficient data for monthly breakdown.*")
        _w("")

        # --- 6. Assumptions & caveats ---
        _w("## Assumptions & Caveats\n")
        _w("- Returns are computed on a **close-to-close** basis using adjusted prices.")
        _w("- Transaction costs and slippage are modelled as fixed basis-point charges per unit of turnover.")
        _w("- No market-impact model is applied; results may overstate achievable returns for large portfolios.")
        _w("- Short selling is **not** modelled; the strategy is long-only.")
        _w("- Survivorship bias may be present depending on the underlying data source.")
        _w("- Factor scores are computed from point-in-time data but look-ahead bias in fundamental data cannot be fully excluded.")
        _w("- Past performance is not indicative of future results.\n")

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report written → %s", dest)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _df_to_md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame into a GitHub-flavoured markdown table."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def _write_dict_yaml(d: dict, lines: list[str], indent: int = 0) -> None:
    """Recursively render a dict as YAML-like text (for the report)."""
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            _write_dict_yaml(v, lines, indent + 1)
        else:
            lines.append(f"{prefix}{k}: {v}")
