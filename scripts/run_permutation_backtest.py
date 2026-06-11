"""
scripts/run_permutation_backtest.py

Run rolling backtests for the permutation (random-baseline) ablation study.

For each (method, k, seed) combination the script:
  * Computes real cluster labels at each rebalance date
  * Permutes the asset→cluster assignments (preserving cluster sizes)
  * Runs a full rolling backtest with the permuted strategy
  * Saves 3 CSV files identical in format to the live strategy outputs

Output layout
-------------
data/processed/backtests/permutation_test/
    kmeans_ew_inter_markowitz_intra_k5_seed0000/
        kmeans_ew_inter_markowitz_intra_k5_seed0000_weights.csv
        kmeans_ew_inter_markowitz_intra_k5_seed0000_daily_portfolio_returns.csv
        kmeans_ew_inter_markowitz_intra_k5_seed0000_monthly_portfolio_returns.csv
    kmedoids_dtw_ew_inter_markowitz_intra_k5_seed0000/
        ...

Checkpoint
----------
If all 3 CSVs already exist the combination is skipped (safe to re-run).

CLI
---
    python scripts/run_permutation_backtest.py --method both --n_seeds 5
    python scripts/run_permutation_backtest.py --method kmeans --n_seeds 3
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
import sys

warnings.filterwarnings(
    "ignore",
    message="n_clusters should be larger than 2",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.data.load import load_returns_csv
from src.backtest.rolling import run_rolling_backtest
from src.backtest.permutation_backtest import make_permuted_strategy
from src.paths import get_returns_path, get_backtests_dir, UNIVERSE_CHOICES

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]

PERM_ROOT = PROJECT_ROOT / "data" / "processed" / "backtests" / "permutation_test"

_METHOD_PREFIX = {
    "kmeans":   "kmeans_ew_inter_markowitz_intra",
    "kmedoids": "kmedoids_dtw_ew_inter_markowitz_intra",
}


def _strategy_name(method: str, k: int, seed: int) -> str:
    """Return the canonical permutation strategy name.

    Parameters
    ----------
    method : str
        Clustering method, either ``"kmeans"`` or ``"kmedoids"``.
    k : int
        Number of clusters.
    seed : int
        Random seed used for label permutation.

    Returns
    -------
    str
        Strategy name string, e.g.
        ``"kmeans_ew_inter_markowitz_intra_k5_seed0003"``.
    """
    return f"{_METHOD_PREFIX[method]}_k{k}_seed{seed:04d}"


def _output_dir(method: str, k: int, seed: int, perm_root: Path = PERM_ROOT) -> Path:
    """Return the output directory for one permutation strategy.

    Parameters
    ----------
    method : str
        Clustering method, either ``"kmeans"`` or ``"kmedoids"``.
    k : int
        Number of clusters.
    seed : int
        Random seed used for label permutation.
    perm_root : Path, optional
        Root directory for permutation backtest outputs.

    Returns
    -------
    Path
        Directory path ``<perm_root>/<strategy_name>/``.
    """
    return perm_root / _strategy_name(method, k, seed)


def _already_done(out_dir: Path, name: str) -> bool:
    """Return True if all expected permutation output files already exist.

    Parameters
    ----------
    out_dir : Path
        Output directory for this (method, k, seed) combination.
    name : str
        Canonical strategy name used to construct expected file names.

    Returns
    -------
    bool
        ``True`` if all three CSVs (weights, daily returns, monthly
        returns) are present in ``out_dir``.
    """
    required = [
        f"{name}_weights.csv",
        f"{name}_daily_portfolio_returns.csv",
        f"{name}_monthly_portfolio_returns.csv",
    ]
    return all((out_dir / f).exists() for f in required)


def _run_one(
    returns,
    method: str,
    k: int,
    seed: int,
    perm_root: Path = PERM_ROOT,
) -> None:
    """Run one permutation backtest and save daily, monthly, and weight CSVs.

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily return matrix for the universe.
    method : str
        Clustering method, either ``"kmeans"`` or ``"kmedoids"``.
    k : int
        Number of clusters.
    seed : int
        Random seed for the label-permutation RNG.
    perm_root : Path, optional
        Root directory where per-strategy output folders are created.
    """
    name    = _strategy_name(method, k, seed)
    out_dir = _output_dir(method, k, seed, perm_root)

    if _already_done(out_dir, name):
        tqdm.write(f"[SKIP] {name} — already complete")
        return

    tqdm.write(f"\n>>> {name}")
    os.makedirs(out_dir, exist_ok=True)

    strategy_fn = make_permuted_strategy(
        method=method,
        k=k,
        seed=seed,
        random_state=42,
        dtw_n_jobs=-1,
    )

    t0 = time.perf_counter()
    try:
        portfolio_returns, weights_history, _ = run_rolling_backtest(
            returns=returns,
            strategy_func=strategy_fn,
            estimation_window=756,
            show_progress=True,
            progress_desc=name,
            detailed_progress=False,
            heartbeat_seconds=None,
        )
    except Exception as exc:
        tqdm.write(f"\n[ERROR] {name} failed — skipping. Reason: {exc}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.perf_counter() - t0
    tqdm.write(f"    Completed in {elapsed:.1f}s")

    monthly_returns = (1 + portfolio_returns).resample("ME").prod() - 1

    portfolio_returns.to_csv(out_dir / f"{name}_daily_portfolio_returns.csv")
    monthly_returns.to_csv(out_dir / f"{name}_monthly_portfolio_returns.csv")
    weights_history.to_csv(out_dir / f"{name}_weights.csv")
    tqdm.write(f"    Saved to: {out_dir}")


def main() -> None:
    """Parse CLI arguments and run all requested permutation backtests."""
    parser = argparse.ArgumentParser(description="Run permutation (random-baseline) backtests.")
    parser.add_argument(
        "--method",
        choices=["kmeans", "kmedoids", "both"],
        default="both",
        help="Which clustering method to permute (default: both)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds per (method, k) combination (default: 5)",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=None,
        help="Override K_VALUES (e.g. --k_values 3 5 10). Default: all K_VALUES.",
    )
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    args = parser.parse_args()

    methods = ["kmeans", "kmedoids"] if args.method == "both" else [args.method]
    seeds = list(range(args.n_seeds))

    returns_path = get_returns_path(args.universe)
    # Load returns first so we can clip k_values to the universe size
    returns = load_returns_csv(returns_path)
    n_tickers = returns.shape[1]
    base_k_values = args.k_values if args.k_values else K_VALUES
    k_values = [k for k in base_k_values if k <= n_tickers]

    project_root = Path(__file__).resolve().parents[1]
    perm_root = get_backtests_dir(args.universe) / "permutation_test"
    os.makedirs(perm_root, exist_ok=True)

    print(f"Project root : {project_root}")
    print(f"Universe     : {args.universe}")
    print(f"Methods      : {methods}")
    print(f"K_VALUES     : {k_values}")
    print(f"Seeds        : {seeds}")
    print(f"Total runs   : {len(methods) * len(k_values) * len(seeds)}")

    combos = [
        (method, k, seed)
        for method in methods
        for k in k_values
        for seed in seeds
    ]

    outer_bar = tqdm(combos, desc="Permutation backtests", unit="run", dynamic_ncols=True)
    for method, k, seed in outer_bar:
        outer_bar.set_postfix_str(f"{method} k={k} seed={seed}")
        _run_one(returns, method, k, seed, perm_root=perm_root)


if __name__ == "__main__":
    main()
