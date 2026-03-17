import pandas as pd
from pypfopt import EfficientFrontier


def get_gmv_weights(window_returns: pd.DataFrame) -> pd.Series:
    """
    Compute long-only global minimum variance portfolio weights
    using sample covariance.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily return matrix for the estimation window.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker.
    """
    if window_returns.empty:
        raise ValueError("window_returns is empty")

    mu = pd.Series(0.0, index=window_returns.columns)
    S = window_returns.cov() * 252

    ef = EfficientFrontier(
        expected_returns=mu,
        cov_matrix=S,
        weight_bounds=(0, 1),
    )

    ef.min_volatility()
    weights = pd.Series(ef.weights, index=window_returns.columns, dtype="float64")

    weights = weights.clip(lower=0.0)

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Global minimum variance optimizer returned non-positive total weight")

    weights = weights / weight_sum

    return weights