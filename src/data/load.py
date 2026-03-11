from pathlib import Path
import pandas as pd

# load processed returns matrix from CSV
def load_returns_csv(path: str | Path) -> pd.DataFrame:

    path = Path(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Basic validation
    if df.empty:
        raise ValueError(f"Loaded returns dataframe is empty: {path}")

    if df.isna().sum().sum() > 0:
        raise ValueError(f"Returns dataframe contains NaN values: {path}")

    return df