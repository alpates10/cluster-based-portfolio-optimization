from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import get_raw_dir, get_processed_dir, UNIVERSE_CHOICES


def load_close_series(fp: Path) -> pd.Series:
    """Read one CSV and return a Close price series indexed by Date."""
    df = pd.read_csv(fp)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{fp.name}: expected columns Date and Close, got {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    s = pd.to_numeric(df["Close"], errors="coerce")
    out = pd.Series(s.values, index=df["Date"].values, name=fp.stem)
    out = out[~pd.isna(out)]
    out = out[out > 0]
    out = out[~pd.Index(out.index).duplicated(keep="last")]
    out.index = pd.to_datetime(out.index)
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare return data for a given universe.")
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    parser.add_argument("--start_date", default="2005-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--coverage_threshold", type=float, default=0.95)
    parser.add_argument("--drop_worst_k", type=int, default=10)
    args = parser.parse_args()

    START_DATE = args.start_date
    ESTIMATION_WINDOW_DAYS = 756
    COVERAGE_THRESHOLD = args.coverage_threshold
    DROP_WORST_K_BY_RETURN_MISSING = args.drop_worst_k
    RETURN_TYPE = "simple"

    RAW_DIR = get_raw_dir(args.universe)
    OUT_DIR = get_processed_dir(args.universe)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))
    raw_total_files = len(files)
    if raw_total_files == 0:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")


    obs_count = {}
    for fp in files:
        try:
            obs_count[fp.stem] = len(pd.read_csv(fp))
        except Exception:
            obs_count[fp.stem] = 0

    obs_series = pd.Series(obs_count).sort_index()
    eligible = obs_series[obs_series >= ESTIMATION_WINDOW_DAYS].index.tolist()
    dropped_by_min_obs = obs_series[obs_series < ESTIMATION_WINDOW_DAYS].index.tolist()


    price_series = []
    bad_files = []

    for fp in files:
        tkr = fp.stem
        if tkr not in eligible:
            continue
        try:
            price_series.append(load_close_series(fp))
        except Exception as e:
            bad_files.append({"ticker": tkr, "file": fp.name, "error": str(e)})

    if not price_series:
        raise RuntimeError("No eligible price series could be loaded.")

    prices_wide = pd.concat(price_series, axis=1).sort_index()


    prices_after = prices_wide.loc[prices_wide.index >= pd.to_datetime(START_DATE)].copy()


    coverage = prices_after.notna().mean().sort_values(ascending=False)
    universe_cov = coverage[coverage >= COVERAGE_THRESHOLD].index.tolist()
    dropped_by_coverage = coverage[coverage < COVERAGE_THRESHOLD].index.tolist()

    prices_u = prices_after[universe_cov].copy()


    if RETURN_TYPE == "log":
        returns_u = np.log(prices_u).diff()
    elif RETURN_TYPE == "simple":
        returns_u = prices_u.pct_change(fill_method=None)
    else:
        raise ValueError("RETURN_TYPE must be 'log' or 'simple'")

    returns_u = returns_u.replace([np.inf, -np.inf], np.nan)


    missing_by_ticker = returns_u.isna().mean().sort_values(ascending=False)

    drop_k = min(DROP_WORST_K_BY_RETURN_MISSING, returns_u.shape[1])
    dropped_by_missing = missing_by_ticker.head(drop_k).index.tolist()

    returns_tmp = returns_u.drop(columns=dropped_by_missing)

    returns_final = returns_tmp.dropna(how="any")

    returns_path = OUT_DIR / "returns_final.csv"
    returns_final.to_csv(returns_path)

    universe_df = pd.DataFrame({
        "ticker": returns_final.columns,
    })
    universe_path = OUT_DIR / "universe.csv"
    universe_df.to_csv(universe_path, index=False)

    notes = {
        "raw_total_files": raw_total_files,
        "start_date": START_DATE,
        "return_type": RETURN_TYPE,
        "estimation_window_days": ESTIMATION_WINDOW_DAYS,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "drop_worst_k_by_return_missing": drop_k,

        "counts": {
            "eligible_after_min_obs": len(eligible),
            "dropped_by_min_obs": len(dropped_by_min_obs),
            "loaded_series": int(prices_wide.shape[1]),
            "universe_after_coverage": len(universe_cov),
            "dropped_by_coverage": len(dropped_by_coverage),
            "final_universe": int(returns_final.shape[1]),
            "final_days": int(returns_final.shape[0]),
        },

        "dropped_tickers": {
            "min_obs": dropped_by_min_obs,
            "coverage": dropped_by_coverage[:50],
            "worst_return_missing": dropped_by_missing,
        },

        "missing_stats": {
            "return_missing_rate_top20": missing_by_ticker.head(20).to_dict(),
            "remaining_missing_days_before_dropna": int(returns_tmp.isna().any(axis=1).sum()),
        },

        "price_field_used": "Close",
        "adj_close_available": False,
        "bad_files": bad_files,
        "data_end_date": str(prices_after.index.max().date()),
    }

    notes_path = OUT_DIR / "data_notes.json"
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)

    print("Saved:")
    print(" -", returns_path)
    print(" -", universe_path)
    print(" -", notes_path)


if __name__ == "__main__":
    main()