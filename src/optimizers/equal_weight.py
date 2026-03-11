import numpy as np
import pandas as pd

# Equal weight strategy function
def get_equal_weight(window_returns: pd.DataFrame) -> pd.Series:

    n_assets = window_returns.shape[1]
    if n_assets == 0:
        raise ValueError("No assets found in window_returns")

    weights = np.ones(n_assets) / n_assets
    return pd.Series(weights, index=window_returns.columns, name="weight")