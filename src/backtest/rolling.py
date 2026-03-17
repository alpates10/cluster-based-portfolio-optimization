import pandas as pd
import numpy as np

# Generate monthly rebalance dates
def get_monthly_rebalance_dates(returns: pd.DataFrame, estimation_window: int) -> list[pd.Timestamp]:

    # Rebalance happens on the first available trading day of each month
    if len(returns) <= estimation_window:
        raise ValueError("Not enough data for the chosen estimation window")

    valid_index = returns.index[estimation_window:]
    rebalance_dates = []

    seen_months = set()
    for dt in valid_index:
        key = (dt.year, dt.month)
        if key not in seen_months:
            rebalance_dates.append(dt)
            seen_months.add(key)

    return rebalance_dates


def run_rolling_backtest(
    returns: pd.DataFrame,
    strategy_func,
    estimation_window: int = 756,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run a rolling monthly backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns matrix (T x N), no NaNs.
    strategy_func : callable
        Function taking window_returns and returning pd.Series weights.
    estimation_window : int
        Number of past trading days used for estimation.

    Returns
    -------
    portfolio_returns : pd.Series
        Daily out-of-sample portfolio returns.
    weights_history : pd.DataFrame
        Portfolio weights at each rebalance date.
    """
    rebalance_dates = get_monthly_rebalance_dates(returns, estimation_window)

    portfolio_returns_list = []
    weights_records = []

    for i, rebalance_date in enumerate(rebalance_dates):
        t = returns.index.get_loc(rebalance_date)

        # estimation window returns
        window_returns = returns.iloc[t - estimation_window:t]

        # find weights using the strategy function
        weights = strategy_func(window_returns)

        if not isinstance(weights, pd.Series):
            raise TypeError("strategy_func must return a pandas Series")

        # weights should be aligned with returns columns
        weights = weights.reindex(returns.columns)

        if weights.isna().any():
            raise ValueError(f"Strategy returned NaN weights at {rebalance_date}")

        weight_sum = weights.sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-4):
            raise ValueError(f"Weights do not sum to 1 at {rebalance_date}. Sum={weight_sum}")

        if (weights < -1e-12).any():
            raise ValueError(f"Negative weights found at {rebalance_date}")

        # out-of-sample returns for the period until the next rebalance
        if i < len(rebalance_dates) - 1:
            next_rebalance_date = rebalance_dates[i + 1]
            oos_returns = returns.loc[rebalance_date:next_rebalance_date].iloc[1:]
        else:
            oos_returns = returns.loc[rebalance_date:].iloc[1:]

        if oos_returns.empty:
            continue

        # daily portfolio returns 
        port_rets = oos_returns @ weights
        port_rets.name = "portfolio_return"

        portfolio_returns_list.append(port_rets)

        weight_row = weights.copy()
        weight_row.name = rebalance_date
        weights_records.append(weight_row)

    if not portfolio_returns_list:
        raise ValueError("No portfolio returns were generated")

    portfolio_returns = pd.concat(portfolio_returns_list).sort_index()
    weights_history = pd.DataFrame(weights_records)
    weights_history.index.name = "rebalance_date"

    return portfolio_returns, weights_history