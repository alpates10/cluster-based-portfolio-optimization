# Cluster-Based Portfolio Optimization

A systematic rolling-backtest framework comparing **clustering-based portfolio strategies** against classical optimization methods across four equity universes (S&P 500, BIST 100, NIFTY 50, SSE) over a 17-year out-of-sample period (2008–2025).

This is the codebase for my undergraduate thesis at Galatasaray University.

---

## Overview

The central idea: instead of optimizing directly over a large asset universe, group assets into clusters using similarity measures, then allocate across and within clusters. This project evaluates whether such structured diversification improves risk-adjusted performance over classical methods.

**Clustering algorithms:** K-Means (Euclidean), K-Medoids (DTW distance)  
**Optimization layers:** Equal Weight, Global Minimum Variance, Markowitz (Max-Sharpe)  
**Cluster allocation variants:** fixed-k, adaptive-k (silhouette-optimal per window)  
**Baselines:** Equal Weight (1/N), Mean-Variance, GMV applied to the full universe  
**Evaluation:** Sharpe ratio, annualized return/volatility, max drawdown, Calmar ratio — out-of-sample only

---

## How it works

**Rolling backtest** — the core evaluation loop. At each month, weights are computed using only the past 756 trading days (~3 years) of data, then applied forward for one month. This is repeated across the full out-of-sample period to simulate realistic, no-look-ahead portfolio management.

**Classical optimizers** applied to the full asset universe:
- *Equal Weight (1/N)* — uniform allocation, the hardest-to-beat baseline in practice.
- *Global Minimum Variance (GMV)* — minimizes portfolio variance regardless of expected returns.
- *Mean-Variance (Markowitz)* — maximizes Sharpe ratio by jointly optimizing return and risk.

**Clustering-based strategies** — assets are first grouped by return similarity, then weights are allocated in two stages: *inter* (across clusters) and *intra* (within each cluster). Each stage can independently use a different optimizer, giving six combinations per clustering algorithm:

| Strategy name | Inter-cluster | Intra-cluster |
|---|---|---|
| `kmeans_k*` / `kmedoids_dtw_k*` | Equal Weight | Equal Weight |
| `*_markowitz_inter_ew_intra_k*` | Markowitz | Equal Weight |
| `*_ew_inter_markowitz_intra_k*` | Equal Weight | Markowitz |
| `*_markowitz_inter_markowitz_intra_k*` | Markowitz | Markowitz |

- *K-Means* uses Euclidean distance on raw return vectors — fast and interpretable.
- *K-Medoids (DTW)* uses Dynamic Time Warping, which captures shape similarity between return series regardless of small timing shifts — slower but more robust for financial time series.
- *Fixed-k* runs with a predetermined number of clusters; *adaptive-k* (`*_adaptive_*`) selects the optimal k each window via silhouette score.

**Rolling validation** — instead of fixing k globally, each rebalancing window is split into an estimation period and a held-out validation window. The k that maximizes out-of-sample Sharpe on the validation window is selected. Variants: `3m`, `6m`, `12m` validation window lengths × EW and Markowitz intra-cluster optimizers.

**Permutation test** — a statistical sanity check for clustering-based strategies. At each rebalance date, the real cluster labels are computed but the asset-to-cluster assignments are randomly shuffled (preserving cluster sizes). Running this across multiple random seeds produces a null distribution of returns — if the real strategy doesn't outperform its permuted counterparts, the clustering structure adds no value.

---

## Project Structure

```
scripts/
├── prepare_data.py                   # Step 1: clean raw CSVs → returns_final.csv
├── run_backtest.py                   # Step 2: rolling backtests (classical + clustering)
├── run_rolling_validation_backtest.py # Step 3: rolling-validation k-selection backtests
├── make_performance_report.py        # Step 4: summary tables & figures
├── run_permutation_backtest.py       # Optional: permutation (random-baseline) backtests
├── make_permutation_summary.py       # Optional: aggregate permutation test results
├── make_clustering_plots.py          # Optional: clustering visualisation plots
├── analyze_clustering_k.py          # Optional: silhouette analysis for k selection
├── analyze_universe_coverage.py      # Optional: simulate coverage filtering across parameters
└── run_all.py                        # Convenience: runs the full pipeline end-to-end

src/
├── backtest/                # Rolling backtest engine + permutation backtest logic
├── clustering/              # K-Means & K-Medoids pipelines, DTW distances, silhouette evaluation
├── optimizers/              # Strategy implementations (EW, GMV, MV, cluster variants)
├── analysis/                # Performance report generation + figures
├── metrics/                 # Performance metric calculations
└── data/                   # Data loading utilities

data/
├── raw/                     # Per-ticker CSV files (Date, Close) — one file per asset
│   ├── sp500_stocks/
│   ├── bist100_stocks/
│   ├── nifty50_stocks/
│   └── sse_stocks/
└── processed/
    ├── sp500/               # returns_final.csv + backtests/<strategy>/
    ├── bist100/
    ├── nifty50/
    ├── sse/
    └── dtw_cache/           # SHA-256 hash-keyed DTW matrix cache (.npy)
```

---

## Setup

**Requirements:** Python 3.13

```bash
git clone https://github.com/alpates/cluster-based-portfolio-optimization.git
cd cluster-based-portfolio-optimization
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Apple Silicon note:** tslearn has a float64 incompatibility with MPS; the pipeline automatically falls back to CPU.

---

## Usage

> A full reference of all commands with flags and examples is available in [`commands.html`](commands.html).

To run the full pipeline end-to-end with a single command:

```bash
python scripts/run_all.py --universe sp500
python scripts/run_all.py --universe sp500 --dry-run   # preview commands without executing
python scripts/run_all.py --universe sp500 --skip-permutation  # skip permutation tests
```

### 1. Prepare data

Raw data is not included in this repository. Each universe requires per-ticker CSV files with columns `Date` and `Close`, placed under `data/raw/<universe>_stocks/`. Data was sourced from Yahoo Finance via `yfinance`. Once the files are in place, run:

```bash
python scripts/prepare_data.py --universe sp500
```

**Date range:** The script starts from `--start_date` (default `2005-01-01`) and automatically extends to the last date present in the raw CSV files — no manual end date needed. When new data arrives, just re-run the script.

**Filtering parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--start_date` | `2005-01-01` | Rows before this date are dropped |
| `--coverage_threshold` | `0.95` | Tickers with less than 95% non-null rows (after start date) are excluded |
| `--drop_worst_k` | `10` | The 10 tickers with the highest remaining missing-return rate are dropped |

After filtering, any day that still has a missing return for any ticker is removed (`dropna`), so the actual start of `returns_final.csv` may be later than `--start_date` depending on the universe.

### 2. Run backtests

```bash
# Classical baselines only (EW, Mean-Variance, GMV)
python scripts/run_backtest.py --universe sp500 --mode classical

# Fixed-k K-Means equal-weight strategies (k = 2..10, 15, 20, 30, 50)
python scripts/run_backtest.py --universe sp500 --mode kmeans

# Fixed-k K-Medoids (DTW) equal-weight strategies
python scripts/run_backtest.py --universe sp500 --mode kmedoids

# Adaptive-k equal-weight strategies (silhouette-optimal k per window)
python scripts/run_backtest.py --universe sp500 --mode adaptive

# Cluster + Markowitz variants (K-Means)
python scripts/run_backtest.py --universe sp500 --mode kmeans_markowitz

# Cluster + Markowitz variants (K-Medoids DTW)
python scripts/run_backtest.py --universe sp500 --mode kmedoids_markowitz

# Adaptive-k Markowitz strategies
python scripts/run_backtest.py --universe sp500 --mode adaptive_markowitz

# Everything at once (default)
python scripts/run_backtest.py --universe sp500 --mode all
```

To skip specific strategies during a run:

```bash
python scripts/run_backtest.py --universe sp500 --skip-strategies mean_variance
```

### 3. Run rolling validation backtests

```bash
# Both methods, all validation windows (3m, 6m, 12m), both weightings (EW, MV)
python scripts/run_rolling_validation_backtest.py --universe sp500

# Customise method, weighting, or validation window
python scripts/run_rolling_validation_backtest.py --universe sp500 --method kmeans --weighting ew --val_window 6 12
```

### 4. Generate performance report

```bash
python scripts/make_performance_report.py --universe sp500 --mode all
```

| Mode | Included strategies | Output dir |
|---|---|---|
| `classical` | EW, MV, GMV | `summary/classical/` |
| `clustering` | kmeans\_k\*, kmedoids\_dtw\_k\* | `summary/clustering/` |
| `rolling_validation` | rolling validation strategies | `summary/rolling_validation/` |
| `all` | classical + clustering + rolling\_validation (aligned to 12-month start) | `summary/all/` |
| `all_no_rolling` | classical + clustering, natural start dates | `summary/all_no_rolling/` |

Available universes: `sp500` · `bist100` · `nifty50` · `sse`

Reports are saved to `data/processed/<universe>/backtests/summary/`.

### 6. Run permutation tests (optional)

```bash
python scripts/run_permutation_backtest.py --universe sp500 --method both --n_seeds 5
python scripts/run_permutation_backtest.py --universe sp500 --method kmeans --n_seeds 10 --k_values 3 5 10
```

### 7. Generate permutation summary (optional)

```bash
python scripts/make_permutation_summary.py --universe sp500 --method both
```

### 8. Generate clustering plots (optional)

```bash
python scripts/make_clustering_plots.py --universe sp500
```

### 9. Analyze optimal k (optional)

```bash
python scripts/analyze_clustering_k.py --universe sp500
```

---

## Methodology Notes

- **Estimation window:** 756 trading days (~3 years), monthly rebalancing
- **Out-of-sample period:** 2008–2025 (215 rebalancing periods)
- **K search space:** {2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50}
- **DTW cache:** Distance matrices are cached using SHA-256 hashes of the input data to avoid redundant computation
- **Risk-free rate:** 0% throughout (both optimization and evaluation)
- **No look-ahead bias:** All parameters estimated strictly on past data within each rolling window

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | K-Means clustering |
| `scikit-learn-extra` | K-Medoids (PAM) |
| `tslearn` | DTW distance computation |
| `cvxpy` | Convex optimization (GMV) |
| `pyportfolioopt` | Markowitz / Max-Sharpe optimization |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Visualization |
