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

from src.data.load import load_returns_csv
from src.optimizers.equal_weight import get_equal_weight
from src.optimizers.mean_variance import get_mean_variance_weights
from src.optimizers.gmv import get_gmv_weights
from src.optimizers.cvar import get_cvar_weights
from src.optimizers.cluster_equal_weight import (
    get_cluster_kmeans_equal_weight,
    get_cluster_kmedoids_dtw_equal_weight,
)
from src.backtest.rolling import run_rolling_backtest

K_VALUES = [2, 3, 4, 5, 6, 7, 8]

CLASSICAL_NAMES = {"equal_weight", "mean_variance", "gmv", "cvar"}


def _strategy_out_dir(base_out_dir: Path, strategy_name: str) -> Path:
    """Map a strategy name to its output directory under the new folder layout.

    Classical strategies → backtests/<name>/
    KMeans clustering   → backtests/kmeans_equal_weight/<name>/
    K-Medoids clustering → backtests/kmedoids_dtw_equal_weight/<name>/
    """
    if strategy_name.startswith("kmeans_k"):
        return base_out_dir / "kmeans_equal_weight" / strategy_name
    if strategy_name.startswith("kmedoids_dtw_k"):
        return base_out_dir / "kmedoids_dtw_equal_weight" / strategy_name
    return base_out_dir / strategy_name


def _rename_old_folders(base_out_dir: Path) -> None:
    """Rename legacy cluster strategy folders to <name>_old."""
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


def _build_strategies(mode: str) -> dict:
    strategies: dict = {}

    if mode in ("classical", "all"):
        strategies["equal_weight"] = get_equal_weight
        strategies["mean_variance"] = get_mean_variance_weights
        strategies["gmv"] = get_gmv_weights
        #strategies["cvar"] = get_cvar_weights

    if mode in ("kmeans", "all"):
        for k in K_VALUES:
            strategies[f"kmeans_k{k}"] = partial(
                get_cluster_kmeans_equal_weight,
                global_k=k,
                random_state=42,
            )

    if mode in ("kmedoids", "all"):
        for k in K_VALUES:
            strategies[f"kmedoids_dtw_k{k}"] = partial(
                get_cluster_kmedoids_dtw_equal_weight,
                global_k=k,
                random_state=42,
                dtw_n_jobs=-1,
            )

    return strategies


def _already_done(base_out_dir: Path, strategy_name: str) -> bool:
    """Return True if all 3 output files exist for this strategy."""
    out_dir = _strategy_out_dir(base_out_dir, strategy_name)
    return all(
        (out_dir / f"{strategy_name}_{suffix}.csv").exists()
        for suffix in (
            "daily_portfolio_returns",
            "monthly_portfolio_returns",
            "weights",
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Run rolling backtests.")
    parser.add_argument(
        "--mode",
        choices=["classical", "kmeans", "kmedoids", "all"],
        default="all",
        help=(
            "classical → equal_weight/mean_variance/gmv/cvar only; "
            "kmeans → kmeans_k2..k8; "
            "kmedoids → kmedoids_dtw_k2..k8; "
            "all → everything (default)"
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    print(f"Project root : {project_root}")
    print(f"Mode         : {args.mode}")

    returns_path = project_root / "data" / "processed" / "returns_final.csv"
    base_out_dir = project_root / "data" / "processed" / "backtests"
    os.makedirs(base_out_dir, exist_ok=True)

    _rename_old_folders(base_out_dir)

    returns = load_returns_csv(returns_path)
    strategies = _build_strategies(args.mode)

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

        # Skip if all 3 output files already exist
        if _already_done(base_out_dir, strategy_name):
            tqdm.write(f"[SKIP] {strategy_name} — already complete")
            continue

        tqdm.write(f"\n>>> {strategy_name}")
        os.makedirs(out_dir, exist_ok=True)

        is_kmedoids = "kmedoids" in strategy_name

        t0 = time.perf_counter()
        try:
            portfolio_returns, weights_history = run_rolling_backtest(
                returns=returns,
                strategy_func=strategy_func,
                estimation_window=756,
                show_progress=True,
                progress_desc=strategy_name,
                detailed_progress=is_kmedoids,
                heartbeat_seconds=30 if is_kmedoids else None,
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

        tqdm.write(f"    Saved to: {out_dir}")


if __name__ == "__main__":
    main()
