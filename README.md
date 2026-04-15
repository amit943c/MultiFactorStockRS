# Multi-Factor Stock Ranking System

A quantitative equity research platform that ranks ~100 U.S. large-cap stocks using a composite factor model, builds concentrated portfolios, and evaluates them through backtesting with full transparency on assumptions, costs, and biases.

The default configuration runs a momentum + trend model on a 100-stock S&P 500 subset from 2020–2025. Monthly rebalancing, top-10 holdings, equal-weight. Out of the box it produces a **+299% total return (26.5% CAGR)** with a **1.07 Sharpe** and **-27% max drawdown** against SPY — though the usual caveats apply (see [Assumptions and Limitations](#assumptions-and-limitations)).

---

## Why this exists

Most open-source factor backtests fall into one of two categories: a 50-line notebook that hides every assumption, or an over-engineered framework where the actual research gets lost. This project tries to sit between the two.

The goal was to build something close to what a small systematic equity team would use internally — proper separation between signals and execution, configurable everything via YAML, automated lookahead bias checks, and research diagnostics that go beyond a single equity curve. It's meant to be readable, extensible, and honest about what it can and can't tell you.

It also needed to produce visuals clean enough for a blog post or a portfolio walkthrough, without requiring a separate charting step.

---

## Key results

From the default backtest (Jan 2020 – Dec 2025, 100-stock universe):

| Metric | Value |
|--------|-------|
| Total Return | +299.5% |
| CAGR | 26.5% |
| Sharpe Ratio | 1.07 |
| Sortino Ratio | 1.04 |
| Max Drawdown | -27.4% |
| Calmar Ratio | 0.97 |
| Avg Monthly Turnover | 47.8% |

The model concentrates into 10 names with the strongest 3-month and 6-month momentum, filtered by trend strength (distance from 50/200-day MAs) and a liquidity floor. Mean reversion and volatility factors are available but disabled by default — they tend to drag in trending markets.

These numbers are in-sample. There's no walk-forward validation yet. Take them as a starting point for research, not a trading signal.

---

## Project structure

```
MultiFactorStockRS/
├── app/                          # Streamlit dashboard
│   ├── dashboard.py              # Main entry point
│   └── views/                    # Page modules (9 pages)
├── config/
│   └── default_config.yaml       # All parameters
├── data/
│   ├── processed/                # Parquet cache (auto-generated)
│   └── sp500_universe.csv        # 100-ticker universe
├── outputs/
│   ├── charts/                   # 16 exported PNGs
│   └── reports/                  # Markdown reports
├── src/
│   ├── data/                     # Yahoo Finance fetcher, caching, universe mgmt
│   ├── factors/                  # Momentum, trend, mean reversion, liquidity,
│   │                             # volatility, fundamental, sentiment (placeholder)
│   ├── portfolio/                # Top-N selection, equal weighting, rebalancing
│   ├── backtest/                 # Mark-to-market engine, benchmark, perf stats
│   ├── analytics/                # Factor IC, quantile returns, regime analysis,
│   │                             # sensitivity sweeps, validation, reporting
│   ├── visualization/            # 22 Plotly chart types, light/dark themes, export
│   └── utils/                    # Config loader, logging, helpers
├── tests/
├── main.py                       # CLI pipeline
└── requirements.txt
```

The pipeline flows left to right: **data → factors → portfolio → backtest → analytics → visualization**. Each stage is independent and testable on its own.

---

## How the factor model works

### Factors

| Group | Signals | Direction | Default weight |
|-------|---------|-----------|----------------|
| Momentum | 1m, 3m, 6m return | Higher = better | 15%, 35%, 20% |
| Trend | Dist from MA-50, MA-200 | Higher = better | 15%, 10% |
| Liquidity | Avg dollar volume | Higher = better | 5% |
| Mean Reversion | RSI-14 | Lower = better | Disabled |
| Volatility | 60d realized vol | Lower = better | Disabled |
| Fundamental | PE, EBITDA margin | Mixed | Disabled |
| Sentiment | Placeholder | — | Disabled |

### Scoring

On each date, for each stock:

1. Compute raw factor values from trailing price data
2. Winsorize at 1st/99th percentile to cap outliers
3. Cross-sectional z-score (optionally within sectors)
4. Flip sign for "lower is better" factors
5. Weighted sum → composite score
6. Rank across all stocks

The top N by composite rank go into the portfolio.

### Weighting options

The default is static weights from the config. The codebase also supports equal-weighting across factors, IC-weighted (tilt toward factors with recent predictive power), and inverse-correlation-weighted (diversify across less correlated signals).

---

## Backtest mechanics

**Rebalancing**: monthly by default (weekly also supported). On each rebalance date, the system re-ranks the universe using only data available up to that date, selects the top N, and equal-weights them.

**Between rebalances**: buy-and-hold drift. Weights evolve with daily returns. No daily reweighting is applied.

**Cost model**: 10 bps transaction cost + 5 bps slippage per unit of turnover, applied at each rebalance. No market impact model — the linear cost assumption breaks down for large positions in illiquid names.

**Benchmark**: SPY total return, aligned to the portfolio's start date.

### Lookahead bias checks

The system runs an automated `LookaheadValidator` suite:

- Factor scores on date T use only prices through T
- Momentum factors correlate with past price changes, not future ones (tested via Pearson correlation)
- Every held position has a valid price on the rebalance date
- No duplicate or negative price rows
- Survivorship coverage reported (how many tickers span the full backtest)

These checks run automatically and the results are surfaced on the Assumptions page of the dashboard.

---

## Research diagnostics

Beyond the backtest P&L, the platform includes tools for evaluating whether the factor signal is real:

**Factor IC**: Spearman rank correlation between composite score and next-period returns. Reported per-date, as a rolling average, and as an IC ratio (mean IC / std IC). A factor with a mean IC of 0.05 and an IR above 0.5 is doing something.

**Quantile analysis**: Stocks are sorted into quintiles by composite score each period. The top quintile should outperform the bottom. The long-short spread (Q1 minus Q5 cumulative return) is the core test.

**Factor decay**: IC computed at forward horizons of 1 to 10 periods. Shows how quickly the signal loses predictive value.

**Score persistence**: Rank autocorrelation between consecutive rebalance dates. High persistence (~0.97 for daily data) means the model isn't thrashing.

**Regime analysis**: Each date is classified as bull/bear (vs SPY 200-day MA) crossed with low/high vol (vs median 60-day realized vol). Strategy performance is broken down by regime.

**Sensitivity**: Automated sweeps across portfolio size (5 to 50 stocks), transaction costs (0 to 50 bps), and rebalance frequency (weekly vs monthly).

---

## Dashboard

The interactive dashboard has 9 pages:

| Page | What's there |
|------|-------------|
| Overview | Headline metrics, equity curve vs SPY, config summary |
| Backtest | Gross/net equity curves, drawdown, rolling Sharpe, monthly heatmap, turnover |
| Factors | Correlation matrix, score distributions, IC summary, z-score heatmaps, scatter plots |
| Holdings | Top-ranked stocks table (adjustable N), weight distribution, composition over time |
| Visuals | Chart gallery with download buttons |
| Research | Rolling IC, quantile returns, long-short spread, persistence, factor decay |
| Sensitivity | Top-N / cost / frequency sweeps with tables and charts |
| Regimes | Market regime timeline, conditional performance, calendar returns, drawdowns |
| Assumptions | Validation results, data quality audit, coverage matrix, methodology notes |

Launch it with:

```bash
streamlit run app/dashboard.py
```

---

## Getting started

### Requirements

- Python 3.11+
- ~2 minutes for the initial data download (100 tickers, 5 years)

### Install

```bash
git clone <repo-url>
cd MultiFactorStockRS
pip install -r requirements.txt
```

### Run the pipeline

```bash
python main.py
```

This downloads prices, computes factors, runs the backtest, generates 16 charts, and writes two markdown reports. Takes about 30 seconds after the first data download.

### CLI flags

```bash
python main.py --skip-fetch          # reuse cached price data
python main.py --theme dark          # dark chart theme
python main.py --dashboard           # auto-launch Streamlit after pipeline
python main.py --config myconfig.yaml
```

### Run tests

```bash
pytest tests/ -v
```

---

## Configuration

Everything lives in `config/default_config.yaml`. Key sections:

```yaml
portfolio:
  top_n: 10                    # number of holdings
  max_position_weight: 0.10    # hard cap per stock
  transaction_cost_bps: 10     # one-way, per unit turnover
  slippage_bps: 5

rebalance:
  frequency: "monthly"         # or "weekly"

factors:
  momentum:
    enabled: true
    weights:
      return_1m: 0.15
      return_3m: 0.35
      return_6m: 0.20
    direction: "higher_is_better"
```

To experiment, just edit the YAML and re-run `python main.py --skip-fetch`.

---

## Assumptions and limitations

Things to keep in mind before reading too much into the numbers:

**Data**: Prices come from Yahoo Finance. Adjusted close prices are used, which means splits and dividends are retroactively adjusted — this is standard but imperfect. The 100-stock universe is defined using current S&P 500 constituents, so there's survivorship bias baked in.

**Execution**: Trades execute at the same close price used to compute signals. A one-day lag would be more realistic. There's no market impact model — the linear cost assumption is fine for small portfolios but breaks down with size.

**Statistical**: This is a single in-sample backtest. The factor weights weren't optimized, but the choice of which factors to enable (and which to disable) is itself a form of selection bias. Walk-forward validation would make the results more credible.

**Fundamentals**: The fundamental factor module exists but is disabled by default because Yahoo Finance fundamentals are a static snapshot, not point-in-time. Using them without proper as-reported dates would introduce lookahead.

**Scope**: Long-only, no leverage, no sector constraints by default. The system doesn't model short selling, borrowing costs, or margin.

---

## Ways to improve this

If you're forking this or building on it, here are directions that would genuinely make it better:

### Reduce overfitting risk

- **Walk-forward validation**: split the sample into expanding training/test windows and only evaluate on out-of-sample periods. This is the single biggest improvement you could make.
- **Purged cross-validation**: for factor weight optimization, use purged k-fold CV to avoid temporal leakage between folds.
- **Multiple testing correction**: if you're sweeping over many factor combinations, apply a Bonferroni or step-down correction to the results.

### Better data

- **Point-in-time fundamentals**: services like Sharadar or Compustat provide as-reported quarterly data with actual filing dates. This eliminates the biggest source of lookahead in fundamental factors.
- **Alternative data**: sentiment from financial news (FinBERT), earnings call transcripts, short interest, or options-implied volatility could add orthogonal signals.
- **Broader universe**: extend to mid-caps (Russell 1000/2000) or international equities. Factor premia tend to be stronger outside large-cap U.S.

### Smarter portfolio construction

- **Risk-based weighting**: instead of equal weight, use inverse-volatility or minimum-variance allocation. This usually improves risk-adjusted returns without needing better signals.
- **Sector constraints**: enforce sector-neutral or sector-bounded portfolios to avoid unintended sector bets.
- **Long-short portfolios**: go long the top quintile and short the bottom. This isolates the factor premium from market beta.
- **Transaction cost optimization**: instead of a hard rebalance, use a no-trade buffer or solve for the portfolio that maximizes expected return net of estimated trading costs.

### More signals

- **Quality factor**: ROE, gross margins, accruals, earnings stability. Quality tends to be persistent and diversifying vs momentum.
- **Value factor**: earnings yield, book-to-price, free cash flow yield. Works best when combined with momentum (value-momentum barbell).
- **Short-term reversal**: 1-week or intraday mean reversion. Orthogonal to medium-term momentum but requires faster execution.
- **Composite signal blending**: use machine learning (ridge regression, gradient boosting, or neural nets) to combine raw factors instead of fixed weights. Train on rolling windows to avoid lookahead.

### Infrastructure

- **Live signal generation**: add a mode that computes today's rankings and outputs a trade list, without running a full backtest.
- **Broker integration**: connect to Interactive Brokers or Alpaca for paper trading or live execution.
- **Scheduling**: run the pipeline on a cron job or Airflow DAG for daily factor updates.
- **Database backend**: replace parquet files with a proper time-series database (TimescaleDB, QuestDB) for faster queries at scale.

### Research depth

- **Factor attribution**: decompose portfolio returns into factor contributions (how much came from momentum vs trend vs stock selection).
- **Crowding analysis**: measure how correlated your top holdings are with popular ETFs or hedge fund filings (13F data).
- **Tail risk analysis**: model the portfolio's exposure to fat-tail events using EVT or historical scenario analysis.
- **Bayesian shrinkage**: apply Black-Litterman or shrinkage estimators to the covariance matrix for more stable portfolio weights.

---

## License

This project is for research and educational purposes. It does not constitute financial advice. Past performance does not predict future returns. Use at your own risk.
