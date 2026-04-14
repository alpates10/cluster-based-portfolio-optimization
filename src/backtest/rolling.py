import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import threading
import time

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
    show_progress: bool = False,
    progress_desc: str | None = None,
    detailed_progress: bool = False,
    heartbeat_seconds: int | None = None,
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
    show_progress : bool
        If True, show a progress bar over rebalance dates.
    progress_desc : str | None
        Optional label for progress bar.
    detailed_progress : bool
        If True, print detailed timing logs per rebalance.
    heartbeat_seconds : int | None
        If set, print periodic "still running" messages while strategy is computing.

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

    rebalance_iter = rebalance_dates
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            rebalance_dates,
            desc=progress_desc or "Rolling Backtest",
            unit="rebalance",
        )
        rebalance_iter = progress_bar

    for i, rebalance_date in enumerate(rebalance_iter):
        if progress_bar is not None:
            progress_bar.set_postfix({"rebalance_date": rebalance_date.date().isoformat()})

        t = returns.index.get_loc(rebalance_date)

        # estimation window returns
        window_returns = returns.iloc[t - estimation_window:t]

        # find weights using the strategy function
        strategy_start = time.perf_counter()
        if detailed_progress:
            tqdm.write(
                f"[{progress_desc or 'strategy'}] "
                f"{i + 1}/{len(rebalance_dates)} {rebalance_date.date()} strategy calc started"
            )

        stop_event = None
        heartbeat_thread = None
        if heartbeat_seconds is not None and heartbeat_seconds > 0:
            stop_event = threading.Event()

            def _heartbeat() -> None:
                while not stop_event.wait(heartbeat_seconds):
                    elapsed = time.perf_counter() - strategy_start
                    tqdm.write(
                        f"[{progress_desc or 'strategy'}] "
                        f"{rebalance_date.date()} still running ({elapsed:.1f}s)"
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()

        try:
            weights = strategy_func(window_returns)
        finally:
            if stop_event is not None:
                stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.1)

        strategy_elapsed = time.perf_counter() - strategy_start
        if detailed_progress:
            tqdm.write(
                f"[{progress_desc or 'strategy'}] "
                f"{rebalance_date.date()} strategy calc finished in {strategy_elapsed:.2f}s"
            )

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
