import pandas as pd
from pypfopt import EfficientCVaR


def get_cvar_weights(window_returns: pd.DataFrame) -> pd.Series:
    """
    Long-only CVaR optimization.

    Uses historical mean returns and minimizes conditional value at risk.
    """
    if window_returns.empty:
        raise ValueError("window_returns is empty")

    mu = window_returns.mean() * 252

    ef = EfficientCVaR(
        expected_returns=mu,
        returns=window_returns,
        beta=0.95,
        weight_bounds=(0, 1),
        solver="SCS",
    )

    ef.min_cvar()

    weights = pd.Series(ef.weights, index=window_returns.columns, dtype="float64")
    weights = weights.clip(lower=0.0)

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("CVaR optimizer returned non-positive total weight")

    weights = weights / weight_sum
    return weights