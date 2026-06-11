import numpy as np
import pandas as pd

def get_equal_weight(window_returns: pd.DataFrame) -> pd.Series:
    """
    Compute equal-weighted (1/N) portfolio weights.

    Every asset receives the same weight; no optimization is performed.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker, each equal to 1/N.
    """
    n_assets = window_returns.shape[1]
    if n_assets == 0:
        raise ValueError("No assets found in window_returns")

    weights = np.ones(n_assets) / n_assets
    return pd.Series(weights, index=window_returns.columns, name="weight")
