from pathlib import Path
import pandas as pd

def load_returns_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a processed daily returns matrix from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the returns CSV. The first column is expected to contain the date index.

    Returns
    -------
    pd.DataFrame
        Daily returns matrix sorted by date and validated to contain no NaN values.

    Raises
    ------
    ValueError
        If the loaded file is empty or contains NaN values.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Basic validation
    if df.empty:
        raise ValueError(f"Loaded returns dataframe is empty: {path}")

    if df.isna().sum().sum() > 0:
        raise ValueError(f"Returns dataframe contains NaN values: {path}")

    return df
