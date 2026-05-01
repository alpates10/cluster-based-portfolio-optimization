from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import get_raw_dir, UNIVERSE_CHOICES

ESTIMATION_WINDOW = 756


def load_close_series(fp: Path) -> pd.Series | None:
    try:
        df = pd.read_csv(fp)
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        s = pd.to_numeric(df["Close"], errors="coerce")
        out = pd.Series(s.values, index=pd.to_datetime(df["Date"].values), name=fp.stem)
        out = out[out.notna() & (out > 0)]
        out = out[~out.index.duplicated(keep="last")]
        return out
    except Exception:
        return None


def simulate_pipeline(
    prices_wide: pd.DataFrame,
    start_year: int,
    coverage_threshold: float,
    drop_worst_k: int,
) -> dict:
    start_date = pd.Timestamp(f"{start_year}-01-01")
    prices_after = prices_wide.loc[prices_wide.index >= start_date].copy()

    eligible = prices_after.shape[1]

    coverage = prices_after.notna().mean()
    universe_cov = coverage[coverage >= coverage_threshold].index.tolist()
    prices_u = prices_after[universe_cov].copy()

    returns_u = prices_u.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    missing_by_ticker = returns_u.isna().mean().sort_values(ascending=False)
    drop_k = min(drop_worst_k, returns_u.shape[1])
    dropped = missing_by_ticker.head(drop_k).index.tolist()
    returns_tmp = returns_u.drop(columns=dropped)

    returns_final = returns_tmp.dropna(how="any")

    return {
        "eligible": eligible,
        "after_coverage": len(universe_cov),
        "final_universe": returns_final.shape[1],
        "final_days": returns_final.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulate prepare_data filtering across start years and parameter combos."
    )
    parser.add_argument("--universe", default="bist100", choices=UNIVERSE_CHOICES)
    parser.add_argument(
        "--start_years", nargs="+", type=int,
        default=[2005, 2007, 2008, 2010, 2012], metavar="YEAR",
    )
    parser.add_argument(
        "--coverage_threshold", nargs="+", type=float,
        default=[0.95], metavar="THRESH",
    )
    parser.add_argument(
        "--drop_worst_k", nargs="+", type=int,
        default=[10], metavar="K",
    )
    args = parser.parse_args()

    raw_dir = get_raw_dir(args.universe)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Loading {len(files)} raw files from {raw_dir} ...")

    series_list = []
    for fp in files:
        s = load_close_series(fp)
        if s is not None and len(s) >= ESTIMATION_WINDOW:
            series_list.append(s)

    if not series_list:
        raise RuntimeError("No eligible price series found.")

    prices_wide = pd.concat(series_list, axis=1).sort_index()
    print(f"Eligible tickers (>= {ESTIMATION_WINDOW} obs): {prices_wide.shape[1]}\n")

    years = sorted(args.start_years)
    combos = list(itertools.product(
        sorted(args.coverage_threshold, reverse=True),
        sorted(args.drop_worst_k),
    ))

    single_combo = len(combos) == 1

    if single_combo:
        cov, k = combos[0]
        header = (
            f"{'start_year':<12} | "
            f"{'eligible':>10} | "
            f"{'after_coverage':>14} | "
            f"{'final_universe':>14} | "
            f"{'final_days':>10}"
        )
        print(header)
        print("-" * len(header))
        for year in years:
            r = simulate_pipeline(prices_wide, year, cov, k)
            print(
                f"{year:<12} | "
                f"{r['eligible']:>10} | "
                f"{r['after_coverage']:>14} | "
                f"{r['final_universe']:>14} | "
                f"{r['final_days']:>10}"
            )
    else:
        # header: start_year | (cov=X, k=Y) final_universe columns
        col_labels = [f"cov={c:.2f} k={k}" for c, k in combos]
        col_w = max(len(lbl) for lbl in col_labels) + 2

        header_parts = [f"{'start_year':<12}"]
        for lbl in col_labels:
            header_parts.append(f"{lbl:>{col_w}}")
        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))

        for year in years:
            row_parts = [f"{year:<12}"]
            for cov, k in combos:
                r = simulate_pipeline(prices_wide, year, cov, k)
                row_parts.append(f"{r['final_universe']:>{col_w}}")
            print(" | ".join(row_parts))

        print(f"\n(values = final_universe ticker count)")


if __name__ == "__main__":
    main()
