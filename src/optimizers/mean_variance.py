import pandas as pd
from pypfopt import EfficientFrontier


def get_mean_variance_weights(window_returns: pd.DataFrame) -> pd.Series:
    """
    Compute long-only mean-variance portfolio weights
    using historical mean returns and sample covariance.

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

    mu = window_returns.mean() * 252
    S = window_returns.cov() * 252

    ef = EfficientFrontier(
        expected_returns=mu,
        cov_matrix=S,
        weight_bounds=(0, 1),
        solver="SCS",
    )

    #ef.max_quadratic_utility(risk_aversion=1.0)
    ef.max_sharpe(risk_free_rate=0.01)
    
    weights = pd.Series(ef.weights, index=window_returns.columns, dtype="float64")

    weights = weights.clip(lower=0.0)

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Mean-variance optimizer returned non-positive total weight")

    weights = weights / weight_sum

    return weights