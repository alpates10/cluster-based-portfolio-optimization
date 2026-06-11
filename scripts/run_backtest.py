import argparse
import os
import shutil
import time
from pathlib import Path
import sys
from functools import partial

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import get_returns_path, get_backtests_dir, get_dtw_cache_dir, UNIVERSE_CHOICES

from src.data.load import load_returns_csv
from src.optimizers.equal_weight import get_equal_weight
from src.optimizers.mean_variance import get_mean_variance_weights
from src.optimizers.gmv import get_gmv_weights
from src.optimizers.cluster_equal_weight import (
    get_cluster_kmeans_equal_weight,
    get_cluster_kmedoids_dtw_equal_weight,
    get_cluster_kmeans_adaptive_ew,
    get_cluster_kmedoids_dtw_adaptive_ew,
)
from src.optimizers.cluster_markowitz import (
    get_cluster_kmeans_markowitz_inter_ew_intra,
    get_cluster_kmeans_ew_inter_markowitz_intra,
    get_cluster_kmeans_markowitz_inter_markowitz_intra,
    get_cluster_kmedoids_dtw_markowitz_inter_ew_intra,
    get_cluster_kmedoids_dtw_ew_inter_markowitz_intra,
    get_cluster_kmedoids_dtw_markowitz_inter_markowitz_intra,
)
from src.optimizers.cluster_markowitz_adaptive import (
    get_cluster_kmeans_adaptive_markowitz_inter_ew_intra,
    get_cluster_kmeans_adaptive_ew_inter_markowitz_intra,
    get_cluster_kmeans_adaptive_markowitz_inter_markowitz_intra,
    get_cluster_kmedoids_dtw_adaptive_markowitz_inter_ew_intra,
    get_cluster_kmedoids_dtw_adaptive_ew_inter_markowitz_intra,
    get_cluster_kmedoids_dtw_adaptive_markowitz_inter_markowitz_intra,
)
from src.backtest.rolling import run_rolling_backtest

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]

CLASSICAL_NAMES = {"equal_weight", "mean_variance", "gmv"}


# ── Fixed-k Markowitz strategy families ──────────────────────────────────────
_MARKOWITZ_FAMILIES = (
    "kmeans_markowitz_inter_ew_intra",
    "kmeans_ew_inter_markowitz_intra",
    "kmeans_markowitz_inter_markowitz_intra",
    "kmedoids_dtw_markowitz_inter_ew_intra",
    "kmedoids_dtw_ew_inter_markowitz_intra",
    "kmedoids_dtw_markowitz_inter_markowitz_intra",
)

# ── Adaptive strategy names (need selected_k.csv) ─────────────────────────────
_ADAPTIVE_STRATEGY_NAMES = {
    "kmeans_adaptive_ew",
    "kmedoids_dtw_adaptive_ew",
    "kmeans_adaptive_markowitz_inter_ew_intra",
    "kmeans_adaptive_ew_inter_markowitz_intra",
    "kmeans_adaptive_markowitz_inter_markowitz_intra",
    "kmedoids_dtw_adaptive_markowitz_inter_ew_intra",
    "kmedoids_dtw_adaptive_ew_inter_markowitz_intra",
    "kmedoids_dtw_adaptive_markowitz_inter_markowitz_intra",
}


def _strategy_out_dir(base_out_dir: Path, strategy_name: str) -> Path:
    """Map a strategy name to its output directory under the new folder layout.

    Classical strategies          → backtests/<name>/
    KMeans equal-weight (fixed k) → backtests/kmeans_equal_weight/<name>/
    KMedoids equal-weight (fixed k)→ backtests/kmedoids_dtw_equal_weight/<name>/
    Markowitz families (fixed k)  → backtests/<family>/<name>/
    Adaptive strategies           → backtests/<name>/
    """
    if strategy_name.startswith("kmeans_k"):
        return base_out_dir / "kmeans_equal_weight" / strategy_name
    if strategy_name.startswith("kmedoids_dtw_k"):
        return base_out_dir / "kmedoids_dtw_equal_weight" / strategy_name
    # Fixed-k Markowitz: <family>_k{n}  →  backtests/<family>/<strategy_name>/
    for family in _MARKOWITZ_FAMILIES:
        if strategy_name.startswith(f"{family}_k"):
            return base_out_dir / family / strategy_name
    return base_out_dir / strategy_name


def _rename_old_folders(base_out_dir: Path) -> None:
    """Rename legacy cluster strategy folders to <name>_old.

    Parameters
    ----------
    base_out_dir : Path
        Root backtests output directory to scan for legacy folder names.
    """
    legacy = [
        "cluster_kmeans_equal_weight",
        "cluster_kmedoids_dtw_equal_weight",
    ]
    for name in legacy:
        old_path = base_out_dir / name
        new_path = base_out_dir / (name + "_old")
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            print(f"Renamed: {name}/ → {name}_old/")
        elif old_path.exists() and new_path.exists():
            shutil.rmtree(old_path)
            print(f"Removed duplicate legacy folder: {name}/")


def _build_strategies(mode: str, dtw_cache: str, k_values: list, k_max: int) -> dict:
    """Build the strategy-name to callable mapping for the requested run mode.

    Parameters
    ----------
    mode : str
        Run mode controlling which strategy families are included
        (e.g. ``"classical"``, ``"kmeans"``, ``"all"``).
    dtw_cache : str
        Path to the DTW distance-matrix cache directory.
    k_values : list
        List of k values to instantiate for fixed-k strategies.
    k_max : int
        Upper bound on k for adaptive strategies (typically ``n_assets - 1``).

    Returns
    -------
    dict
        Mapping from strategy name (``str``) to a zero-argument callable
        (or ``partial``) that accepts ``window_returns`` and returns weights.
    """
    strategies: dict = {}

    if mode in ("classical", "all"):
        strategies["equal_weight"] = get_equal_weight
        strategies["mean_variance"] = get_mean_variance_weights
        strategies["gmv"] = get_gmv_weights

    if mode in ("kmeans", "all"):
        for k in k_values:
            strategies[f"kmeans_k{k}"] = partial(
                get_cluster_kmeans_equal_weight,
                global_k=k,
                random_state=42,
            )

    if mode in ("kmedoids", "all"):
        for k in k_values:
            strategies[f"kmedoids_dtw_k{k}"] = partial(
                get_cluster_kmedoids_dtw_equal_weight,
                global_k=k,
                random_state=42,
                dtw_n_jobs=-1,
                dtw_cache_dir=dtw_cache,
            )

    if mode in ("adaptive", "all"):
        strategies["kmeans_adaptive_ew"] = partial(
            get_cluster_kmeans_adaptive_ew,
            random_state=42,
            k_max=k_max,
        )
        strategies["kmedoids_dtw_adaptive_ew"] = partial(
            get_cluster_kmedoids_dtw_adaptive_ew,
            random_state=42,
            dtw_n_jobs=-1,
            dtw_cache_dir=dtw_cache,
            k_max=k_max,
        )

    # ── Fixed-k Markowitz strategies ──────────────────────────────────────────
    _KMEANS_MARKOWITZ_FUNCS = {
        "kmeans_markowitz_inter_ew_intra":       get_cluster_kmeans_markowitz_inter_ew_intra,
        "kmeans_ew_inter_markowitz_intra":        get_cluster_kmeans_ew_inter_markowitz_intra,
        "kmeans_markowitz_inter_markowitz_intra": (
            get_cluster_kmeans_markowitz_inter_markowitz_intra
        ),
    }
    _KMEDOIDS_MARKOWITZ_FUNCS = {
        "kmedoids_dtw_markowitz_inter_ew_intra": (
            get_cluster_kmedoids_dtw_markowitz_inter_ew_intra
        ),
        "kmedoids_dtw_ew_inter_markowitz_intra": (
            get_cluster_kmedoids_dtw_ew_inter_markowitz_intra
        ),
        "kmedoids_dtw_markowitz_inter_markowitz_intra": (
            get_cluster_kmedoids_dtw_markowitz_inter_markowitz_intra
        ),
    }

    if mode in ("kmeans_markowitz", "all"):
        for family, func in _KMEANS_MARKOWITZ_FUNCS.items():
            for k in k_values:
                strategies[f"{family}_k{k}"] = partial(func, global_k=k, random_state=42)

    if mode in ("kmedoids_markowitz", "all"):
        for family, func in _KMEDOIDS_MARKOWITZ_FUNCS.items():
            for k in k_values:
                strategies[f"{family}_k{k}"] = partial(
                    func, global_k=k, random_state=42, dtw_n_jobs=-1,
                    dtw_cache_dir=dtw_cache,
                )

    # ── Adaptive Markowitz strategies ─────────────────────────────────────────
    if mode in ("adaptive_markowitz", "all"):
        strategies["kmeans_adaptive_markowitz_inter_ew_intra"] = partial(
            get_cluster_kmeans_adaptive_markowitz_inter_ew_intra,
            random_state=42,
            k_max=k_max
        )
        strategies["kmeans_adaptive_ew_inter_markowitz_intra"] = partial(
            get_cluster_kmeans_adaptive_ew_inter_markowitz_intra,
            random_state=42,
            k_max=k_max
        )
        strategies["kmeans_adaptive_markowitz_inter_markowitz_intra"] = partial(
            get_cluster_kmeans_adaptive_markowitz_inter_markowitz_intra,
            random_state=42,
            k_max=k_max,
        )
        strategies["kmedoids_dtw_adaptive_markowitz_inter_ew_intra"] = partial(
            get_cluster_kmedoids_dtw_adaptive_markowitz_inter_ew_intra,
            random_state=42,
            dtw_n_jobs=-1,
            dtw_cache_dir=dtw_cache,
            k_max=k_max,
        )
        strategies["kmedoids_dtw_adaptive_ew_inter_markowitz_intra"] = partial(
            get_cluster_kmedoids_dtw_adaptive_ew_inter_markowitz_intra,
            random_state=42,
            dtw_n_jobs=-1,
            dtw_cache_dir=dtw_cache,
            k_max=k_max,
        )
        strategies["kmedoids_dtw_adaptive_markowitz_inter_markowitz_intra"] = partial(
            get_cluster_kmedoids_dtw_adaptive_markowitz_inter_markowitz_intra,
            random_state=42,
            dtw_n_jobs=-1,
            dtw_cache_dir=dtw_cache,
            k_max=k_max,
        )

    return strategies


def _already_done(base_out_dir: Path, strategy_name: str) -> bool:
    """Return True if all required output files exist for this strategy.

    Parameters
    ----------
    base_out_dir : Path
        Root backtests output directory.
    strategy_name : str
        Canonical name of the strategy (used to locate its output folder
        and to construct expected file names).

    Returns
    -------
    bool
        ``True`` if every required CSV (daily returns, monthly returns,
        weights, and optionally selected_k) is present on disk.
    """
    out_dir = _strategy_out_dir(base_out_dir, strategy_name)
    required_suffixes = ["daily_portfolio_returns", "monthly_portfolio_returns", "weights"]
    if strategy_name in _ADAPTIVE_STRATEGY_NAMES:
        required_suffixes.append("selected_k")
    return all(
        (out_dir / f"{strategy_name}_{suffix}.csv").exists()
        for suffix in required_suffixes
    )


def main():
    """Parse CLI arguments and run rolling backtests for selected strategies."""
    parser = argparse.ArgumentParser(description="Run rolling backtests.")
    parser.add_argument(
        "--mode",
        choices=[
            "classical", "kmeans", "kmedoids", "adaptive",
            "kmeans_markowitz", "kmedoids_markowitz", "adaptive_markowitz",
            "all",
        ],
        default="all",
        help=(
            "classical → equal_weight/mean_variance/gmv only; "
            "kmeans → kmeans_k2..k8 (equal-weight); "
            "kmedoids → kmedoids_dtw_k2..k8 (equal-weight); "
            "adaptive → adaptive equal-weight (kmeans + kmedoids); "
            "kmeans_markowitz → 3 markowitz variants × k2..k8 (kmeans); "
            "kmedoids_markowitz → 3 markowitz variants × k2..k8 (kmedoids); "
            "adaptive_markowitz → 6 adaptive markowitz strategies; "
            "all → everything (default)"
        ),
    )
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    parser.add_argument(
        "--skip-strategies",
        default="",
        help="Comma-separated list of strategy names to skip.",
    )
    args = parser.parse_args()

    skip_set = {s.strip() for s in args.skip_strategies.split(",") if s.strip()}

    project_root = Path(__file__).resolve().parents[1]
    print(f"Project root : {project_root}")
    print(f"Mode         : {args.mode}")
    print(f"Universe     : {args.universe}")
    if skip_set:
        print(f"Skipping     : {', '.join(sorted(skip_set))}")

    returns_path = get_returns_path(args.universe)
    base_out_dir = get_backtests_dir(args.universe)
    os.makedirs(base_out_dir, exist_ok=True)

    _rename_old_folders(base_out_dir)

    returns = load_returns_csv(returns_path)
    n_tickers = returns.shape[1]
    k_values = [k for k in K_VALUES if k <= n_tickers]
    k_max = n_tickers - 1
    strategies = _build_strategies(args.mode, get_dtw_cache_dir(args.universe), k_values, k_max)

    strategy_items = list(strategies.items())
    outer_bar = tqdm(
        strategy_items,
        desc="Running strategies",
        unit="strategy",
        dynamic_ncols=True,
    )

    for strategy_name, strategy_func in outer_bar:
        outer_bar.set_postfix_str(strategy_name)
        out_dir = _strategy_out_dir(base_out_dir, strategy_name)

        if strategy_name in skip_set:
            tqdm.write(f"[SKIP] {strategy_name} — excluded via --skip-strategies")
            continue

        # Skip if all 3 output files already exist
        if _already_done(base_out_dir, strategy_name):
            tqdm.write(f"[SKIP] {strategy_name} — already complete")
            continue

        tqdm.write(f"\n>>> {strategy_name}")
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.perf_counter()
        try:
            portfolio_returns, weights_history, metadata_df = run_rolling_backtest(
                returns=returns,
                strategy_func=strategy_func,
                estimation_window=756,
                show_progress=True,
                progress_desc=strategy_name,
                detailed_progress=False,
                heartbeat_seconds=None,
            )
        except Exception as exc:
            tqdm.write(f"\n[ERROR] {strategy_name} failed — skipping. Reason: {exc}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.perf_counter() - t0
        tqdm.write(f"    Completed in {elapsed:.1f}s")

        monthly_portfolio_returns = (
            (1 + portfolio_returns)
            .resample("ME")
            .prod()
            - 1
        )

        portfolio_returns.to_csv(
            out_dir / f"{strategy_name}_daily_portfolio_returns.csv"
        )
        monthly_portfolio_returns.to_csv(
            out_dir / f"{strategy_name}_monthly_portfolio_returns.csv"
        )
        weights_history.to_csv(out_dir / f"{strategy_name}_weights.csv")

        if strategy_name in _ADAPTIVE_STRATEGY_NAMES and not metadata_df.empty:
            metadata_df.to_csv(out_dir / f"{strategy_name}_selected_k.csv")

        tqdm.write(f"    Saved to: {out_dir}")


if __name__ == "__main__":
    main()
