from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_close_series(fp: Path) -> pd.Series:
    """
    Read one stock CSV and return a cleaned Close price series.

    Expected CSV columns:
    - Date
    - Close

    Output:
    - pd.Series
        index: DatetimeIndex
        values: float prices
        name: ticker (file stem)
    """
    df = pd.read_csv(fp)

    required_cols = {"Date", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{fp.name}: missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Parse and clean date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Parse and clean price
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Keep only positive prices
    df = df[df["Close"] > 0]

    # Remove duplicate dates, keep last occurrence
    df = df.drop_duplicates(subset=["Date"], keep="last")

    series = pd.Series(
        data=df["Close"].values,
        index=df["Date"],
        name=fp.stem,
        dtype="float64",
    )

    return series


def list_csv_files(data_dir: Path) -> list[Path]:
    """
    List all CSV files in a directory, sorted by name.
    """
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")
    return files