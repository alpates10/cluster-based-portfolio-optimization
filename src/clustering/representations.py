from __future__ import annotations

import numpy as np
import pandas as pd


def raw_return_representation(window_returns: pd.DataFrame) -> np.ndarray:
    """
    Build raw return vectors per asset.

    Output shape: (n_assets, window_length)
    """
    if window_returns.empty:
        raise ValueError("window_returns is empty")

    if window_returns.isna().any().any():
        raise ValueError("window_returns contains NaN values")

    matrix = window_returns.to_numpy(dtype=float).T
    if matrix.ndim != 2:
        raise ValueError("Invalid representation shape")

    if not np.isfinite(matrix).all():
        raise ValueError("window_returns contains non-finite values")

    return matrix
