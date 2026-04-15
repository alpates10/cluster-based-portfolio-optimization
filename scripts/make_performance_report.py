"""
Performance report generator.

--mode classical  (default)
    Reads equal_weight / mean_variance / gmv / cvar strategy folders.
    Writes outputs to data/processed/backtests/summary/classical/

--mode clustering
    Auto-discovers kmeans_equal_weight/kmeans_k*/ and
    kmedoids_dtw_equal_weight/kmedoids_dtw_k*/ sub-folders.
    Writes outputs to data/processed/backtests/summary/clustering/

--mode all
    Merges classical + clustering into one combined table.
    Writes outputs to data/processed/backtests/summary/all/
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.backtest_summary import (
    build_rolling_sharpe_comparison_table,
    build_monthly_returns_comparison_table,
    build_summary_metrics_table,
    save_rolling_sharpe_comparison_table,
    save_monthly_returns_comparison_table,
    save_summary_metrics_excel,
    save_summary_metrics_table,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSICAL_NAMES = {"equal_weight", "mean_variance", "gmv", "cvar"}
CLUSTERING_PARENTS = ["kmeans_equal_weight", "kmedoids_dtw_equal_weight"]
ROLLING_WINDOW_MONTHS = 12
RISK_FREE_RATE = 0.0
SUMMARY_ROOT = PROJECT_ROOT / "data" / "processed" / "backtests" / "summary"
# ─────────────────────────────────────────────────────────────────────────────


def _classical_dirs(backtests_dir: Path) -> list[Path]:
    return [
        p for p in sorted(backtests_dir.iterdir())
        if p.is_dir() and p.name in CLASSICAL_NAMES
    ]


def _clustering_dirs(backtests_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    for parent_name in CLUSTERING_PARENTS:
        parent = backtests_dir / parent_name
        if parent.exists():
            dirs.extend(sorted(p for p in parent.iterdir() if p.is_dir()))
    return dirs


def _build_and_save(
    backtests_dir: Path,
    strategy_dirs: list[Path],
    out_dir: Path,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build monthly, summary, and rolling-sharpe tables for *strategy_dirs*
    and save them under *out_dir*.  Returns (monthly_df, summary_df).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_df, monthly_logs = build_monthly_returns_comparison_table(
        backtests_root=backtests_dir,
        strategy_dirs=strategy_dirs,
    )
    if monthly_df.empty:
        print(f"[WARN] {label}: No usable monthly return series found.")
        return pd.DataFrame(), pd.DataFrame()

    summary_df, summary_logs = build_summary_metrics_table(
        backtests_root=backtests_dir,
        risk_free_rate=RISK_FREE_RATE,
        strategy_dirs=strategy_dirs,
    )
    rolling_sharpe_df = build_rolling_sharpe_comparison_table(
        monthly_returns_comparison=monthly_df,
        window_months=ROLLING_WINDOW_MONTHS,
    )

    save_monthly_returns_comparison_table(monthly_df, out_dir / "monthly_returns_comparison.csv")
    save_rolling_sharpe_comparison_table(rolling_sharpe_df, out_dir / "rolling_sharpe_comparison.csv")
    if not summary_df.empty:
        save_summary_metrics_table(summary_df, out_dir / "summary_metrics.csv")
        save_summary_metrics_excel(summary_df, out_dir / "summary_metrics.xlsx")

    print(f"  Monthly comparison  : {out_dir / 'monthly_returns_comparison.csv'}  shape={monthly_df.shape}")
    print(f"  Summary metrics     : {out_dir / 'summary_metrics.csv'}  rows={len(summary_df)}")
    print(f"  Rolling Sharpe      : {out_dir / 'rolling_sharpe_comparison.csv'}  shape={rolling_sharpe_df.shape}")
    print(f"  Strategies: {', '.join(monthly_df.columns)}")

    for line in monthly_logs + summary_logs:
        print(f"  {line}")

    return monthly_df, summary_df


def _run_classical(backtests_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== MODE: classical ===")
    strategy_dirs = _classical_dirs(backtests_dir)
    if not strategy_dirs:
        print("[WARN] No classical strategy folders found.")
        return pd.DataFrame(), pd.DataFrame()

    out_dir = SUMMARY_ROOT / "classical"
    monthly_df, summary_df = _build_and_save(backtests_dir, strategy_dirs, out_dir, "classical")

    if not summary_df.empty:
        print("\nSummary metrics:")
        print(summary_df.to_string(index=False))

    return monthly_df, summary_df


def _run_clustering(backtests_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== MODE: clustering ===")
    strategy_dirs = _clustering_dirs(backtests_dir)
    if not strategy_dirs:
        print("[WARN] No clustering sub-folders found — run scripts/run_backtest.py --mode kmeans/kmedoids first.")
        return pd.DataFrame(), pd.DataFrame()

    out_dir = SUMMARY_ROOT / "clustering"
    monthly_df, summary_df = _build_and_save(backtests_dir, strategy_dirs, out_dir, "clustering")

    if not summary_df.empty:
        print("\nCluster comparison table:")
        print(summary_df.to_string(index=False))

    return monthly_df, summary_df


def _run_all(backtests_dir: Path) -> None:
    print("\n=== MODE: all ===")

    all_dirs = _classical_dirs(backtests_dir) + _clustering_dirs(backtests_dir)
    if not all_dirs:
        print("[WARN] No strategy folders found.")
        return

    out_dir = SUMMARY_ROOT / "all"
    monthly_df, summary_df = _build_and_save(backtests_dir, all_dirs, out_dir, "all")

    if not summary_df.empty:
        print("\nCombined summary metrics (classical + clustering):")
        print(summary_df.to_string(index=False))


def _delete_legacy_files(backtests_dir: Path) -> None:
    """Remove old flat summary files from backtests/ root (now kept in summary/)."""
    legacy = [
        "summary_metrics.csv",
        "summary_metrics.xlsx",
        "monthly_returns_comparison.csv",
        "rolling_sharpe_comparison.csv",
    ]
    for name in legacy:
        p = backtests_dir / name
        if p.exists():
            p.unlink()
            print(f"Removed legacy file: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance reports.")
    parser.add_argument(
        "--mode",
        choices=["classical", "clustering", "all"],
        default="classical",
        help=(
            "classical (default) → equal_weight/mean_variance/gmv/cvar → summary/classical/; "
            "clustering → kmeans_k*/kmedoids_dtw_k* → summary/clustering/; "
            "all → combined → summary/all/"
        ),
    )
    args = parser.parse_args()

    backtests_dir = PROJECT_ROOT / "data" / "processed" / "backtests"
    _delete_legacy_files(backtests_dir)

    if args.mode == "classical":
        _run_classical(backtests_dir)
    elif args.mode == "clustering":
        _run_clustering(backtests_dir)
    elif args.mode == "all":
        _run_all(backtests_dir)


if __name__ == "__main__":
    main()
