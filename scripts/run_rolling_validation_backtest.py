"""
scripts/run_rolling_validation_backtest.py

Run rolling-validation k-selection backtests.

Each combination of (clustering method, weighting scheme, validation window)
is an independent strategy whose results are saved to:

    data/processed/backtests/rolling_validation/<name>/

where <name> follows the pattern:
    kmeans_rolling_val_ew_3m
    kmeans_rolling_val_ew_6m
    kmeans_rolling_val_ew_12m
    kmeans_rolling_val_mv_3m
    kmeans_rolling_val_mv_6m
    kmeans_rolling_val_mv_12m
    kmedoids_dtw_rolling_val_ew_3m
    ...  (12 strategies total)

Per strategy, four CSV files are written:
    <name>_daily_portfolio_returns.csv
    <name>_monthly_portfolio_returns.csv
    <name>_weights.csv
    <name>_selected_k.csv   (columns: selected_k, best_sharpe)

Note: the first val_window rebalances are a warm-up phase; the output series
starts after that phase, so strategies with larger val_window have a later
start date.

Checkpoint / --force
--------------------
By default, if all four output files already exist for a strategy it is
skipped entirely.  Pass --force to overwrite existing results.

Usage examples
--------------
# Run everything (default)
python scripts/run_rolling_validation_backtest.py

# Force re-run all (ignore existing files)
python scripts/run_rolling_validation_backtest.py --force

# Only KMeans strategies with 6-month validation window
python scripts/run_rolling_validation_backtest.py --method kmeans --val_window 6

# KMedoids EW with all validation windows, force re-run
python scripts/run_rolling_validation_backtest.py --method kmedoids --weighting ew --force

# Single combination
python scripts/run_rolling_validation_backtest.py --method kmeans --weighting mv --val_window 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_returns_csv
from src.backtest.rolling_validation_backtest import run_rolling_validation_backtest
from src.paths import get_returns_path, get_backtests_dir, get_dtw_cache_dir, UNIVERSE_CHOICES

BACKTEST_BASE = (
    PROJECT_ROOT / "data" / "processed" / "backtests" / "rolling_validation"
)


# ── Naming helpers ────────────────────────────────────────────────────────────

def _strategy_name(method: str, weighting: str, val_window: int) -> str:
    """Return the canonical strategy name string.

    Parameters
    ----------
    method : str
        Clustering method, either ``"kmeans"`` or ``"kmedoids"``.
    weighting : str
        Portfolio weighting scheme, either ``"ew"`` (equal-weight) or
        ``"mv"`` (mean-variance).
    val_window : int
        Validation window length in months (e.g. ``3``, ``6``, ``12``).

    Returns
    -------
    str
        Strategy name, e.g. ``"kmeans_rolling_val_ew_6m"``.
    """
    prefix = "kmeans" if method == "kmeans" else "kmedoids_dtw"
    return f"{prefix}_rolling_val_{weighting}_{val_window}m"


def _all_files_present(out_dir: Path, name: str) -> bool:
    """Return True if all four output files already exist (checkpoint).

    Parameters
    ----------
    out_dir : Path
        Output directory for this strategy.
    name : str
        Canonical strategy name used to construct expected file names.

    Returns
    -------
    bool
        ``True`` if all four CSVs (daily returns, monthly returns,
        weights, selected_k) are present in ``out_dir``.
    """
    required = [
        f"{name}_daily_portfolio_returns.csv",
        f"{name}_monthly_portfolio_returns.csv",
        f"{name}_weights.csv",
        f"{name}_selected_k.csv",
    ]
    return all((out_dir / fname).exists() for fname in required)


# ── Per-strategy runner ───────────────────────────────────────────────────────

def _run_one(
    returns,
    method: str,
    weighting: str,
    val_window: int,
    backtest_base: Path,
    dtw_cache: str,
    force: bool = False,
) -> None:
    """Run one rolling-validation backtest strategy and save its CSV outputs.

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily return matrix for the universe.
    method : str
        Clustering method, either ``"kmeans"`` or ``"kmedoids"``.
    weighting : str
        Portfolio weighting scheme, either ``"ew"`` or ``"mv"``.
    val_window : int
        Number of past rebalance periods used to score candidate k values.
    backtest_base : Path
        Root directory for rolling-validation strategy outputs.
    dtw_cache : str
        Path to the DTW distance-matrix cache directory.
    force : bool, optional
        If ``True``, overwrite existing output files; otherwise skip
        strategies whose four output CSVs are already present.
    """
    name = _strategy_name(method, weighting, val_window)
    out_dir = backtest_base / name

    if not force and _all_files_present(out_dir, name):
        tqdm.write(f"[SKIP] {name} — all output files present (use --force to overwrite)")
        return

    tqdm.write(f"\n>>> {name}")
    os.makedirs(out_dir, exist_ok=True)

    clustering_method = "kmedoids" if method == "kmedoids" else "kmeans"

    t0 = time.perf_counter()
    try:
        portfolio_returns, weights_history, selected_k_df = run_rolling_validation_backtest(
            returns=returns,
            clustering_method=clustering_method,
            weighting_method=weighting,
            val_window=val_window,
            estimation_window=756,
            k_values=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50],
            random_state=42,
            dtw_n_jobs=-1,
            dtw_cache_dir=dtw_cache,
            show_progress=True,
            progress_desc=name,
        )
    except Exception as exc:
        tqdm.write(f"[ERROR] {name} failed: {exc}")
        traceback.print_exc()
        return

    elapsed = time.perf_counter() - t0
    tqdm.write(f"    Completed in {elapsed:.1f}s")

    monthly_portfolio_returns = (
        (1.0 + portfolio_returns).resample("ME").prod() - 1.0
    )

    portfolio_returns.to_csv(out_dir / f"{name}_daily_portfolio_returns.csv")
    monthly_portfolio_returns.to_csv(out_dir / f"{name}_monthly_portfolio_returns.csv")
    weights_history.to_csv(out_dir / f"{name}_weights.csv")
    selected_k_df.to_csv(out_dir / f"{name}_selected_k.csv")

    tqdm.write(f"    Saved to: {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run requested rolling-validation backtests."""
    parser = argparse.ArgumentParser(
        description="Run rolling-validation k-selection backtests."
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["kmeans", "kmedoids"],
        default=["kmeans", "kmedoids"],
        help="Clustering method(s) to run (default: both).",
    )
    parser.add_argument(
        "--weighting",
        nargs="+",
        choices=["ew", "mv"],
        default=["ew", "mv"],
        help="Portfolio weighting scheme(s) to run (default: both).",
    )
    parser.add_argument(
        "--val_window",
        nargs="+",
        type=int,
        choices=[3, 6, 12],
        default=[3, 6, 12],
        help="Validation window(s) in months (default: 3 6 12).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Overwrite existing output files.  Without this flag strategies "
            "whose four output CSVs already exist are skipped."
        ),
    )
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    args = parser.parse_args()

    backtest_base = get_backtests_dir(args.universe) / "rolling_validation"
    dtw_cache = get_dtw_cache_dir(args.universe)
    returns_path = get_returns_path(args.universe)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Universe     : {args.universe}")
    print(f"Loading returns from: {returns_path}")
    returns = load_returns_csv(returns_path)
    print(f"Returns shape: {returns.shape}")
    if args.force:
        print("Mode: FORCE (existing files will be overwritten)")
    else:
        print("Mode: checkpoint (existing files will be skipped; use --force to overwrite)")

    # Build all (method, weighting, val_window) combinations
    combos = [
        (method, weighting, val_window)
        for method in args.method
        for weighting in args.weighting
        for val_window in sorted(args.val_window)
    ]

    print(f"\nStrategies to run: {len(combos)}")
    for method, weighting, val_window in combos:
        print(f"  {_strategy_name(method, weighting, val_window)}")
    print()

    os.makedirs(backtest_base, exist_ok=True)

    outer_bar = tqdm(
        combos,
        desc="Rolling validation backtests",
        unit="strategy",
        dynamic_ncols=True,
    )

    for method, weighting, val_window in outer_bar:
        name = _strategy_name(method, weighting, val_window)
        outer_bar.set_postfix_str(name)
        _run_one(returns, method, weighting, val_window,
                 backtest_base=backtest_base, dtw_cache=dtw_cache, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
