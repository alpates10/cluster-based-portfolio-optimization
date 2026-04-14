from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    if returns.empty:
        raise ValueError("returns series is empty")

    out = pd.to_numeric(returns, errors="coerce")
    if out.isna().any():
        raise ValueError("returns series contains NaN or non-numeric values")

    if (out <= -1.0).any():
        raise ValueError("returns contain values <= -1.0, wealth would become non-positive")

    return out.astype("float64")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _validate_returns(returns)
    n_obs = len(r)
    growth = float((1.0 + r).prod())

    if growth <= 0.0:
        return -1.0

    return growth ** (periods_per_year / n_obs) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _validate_returns(returns)
    if len(r) < 2:
        return np.nan

    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = _validate_returns(returns)
    if len(r) < 2:
        return np.nan

    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period

    vol = excess.std(ddof=1)
    if np.isclose(vol, 0.0):
        return np.nan

    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def max_drawdown(returns: pd.Series) -> float:
    r = _validate_returns(returns)

    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(returns)

    if np.isclose(mdd, 0.0):
        return np.nan

    return float(ann_ret / abs(mdd))


def compute_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | str]:
    r = _validate_returns(returns)

    return {
        "start_date": str(r.index.min().date()) if isinstance(r.index, pd.DatetimeIndex) else str(r.index.min()),
        "end_date": str(r.index.max().date()) if isinstance(r.index, pd.DatetimeIndex) else str(r.index.max()),
        "n_obs": int(len(r)),
        "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            r,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": max_drawdown(r),
        "calmar_ratio": calmar_ratio(r, periods_per_year=periods_per_year),
    }
