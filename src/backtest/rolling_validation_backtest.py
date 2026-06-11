"""
src/backtest/rolling_validation_backtest.py

Rolling validation backtest engine for k-selection strategies.

At each rebalance date the engine:
1. Computes portfolio weights for **all** k values in k_values.
2. During the first ``val_window`` rebalances (warm-up): accumulates per-k
   monthly returns in val_history but produces **no portfolio output**.
3. After the warm-up period: selects k by comparing annualised Sharpe ratios
   over the past ``val_window`` months of per-k returns, then records the
   portfolio for that rebalance.
4. After recording, updates val_history with each k's realised monthly
   return for the next period's selection.

This avoids look-ahead bias: the k selection at time t only uses returns
from the val_window periods strictly before t.  Because the warm-up months
are dropped entirely, the returned series starts from rebalance index
val_window (not index 0).

Public API
----------
run_rolling_validation_backtest(
    returns, clustering_method, weighting_method, val_window,
    estimation_window=756, k_values=[2,3,4,5,6,7,8], ...
) -> (portfolio_returns, weights_history, selected_k_df)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.backtest.rolling import get_monthly_rebalance_dates
from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import dtw_distance_matrix
from src.clustering.representations import raw_return_representation
from src.optimizers.cluster_equal_weight import _cluster_equal_weight_from_labels
from src.optimizers.cluster_rolling_validation import (
    _sharpe_from_monthly,
    _cluster_mv_from_labels,
)

_DTW_CACHE_DIR: str = str(
    Path(__file__).resolve().parents[2] / "data" / "processed" / "dtw_cache"
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_dtw_cache_key(window_returns: pd.DataFrame) -> str:
    """Build a deterministic cache key from window dates and sorted tickers.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window, indexed by date
        with ticker symbols as columns.

    Returns
    -------
    str
        A 24-character hexadecimal SHA-256 digest suitable for use as a
        cache filename stem.
    """
    idx = window_returns.index
    start = str(idx[0].date()) if hasattr(idx[0], "date") else str(idx[0])
    end = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
    tickers = ",".join(sorted(window_returns.columns.tolist()))
    raw = f"{start}|{end}|{tickers}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _build_weights_for_k(
    k: int,
    asset_matrix: np.ndarray,
    window_returns: pd.DataFrame,
    clustering_method: str,
    weighting_method: str,
    random_state: int,
    dtw_dist: np.ndarray | None,
) -> pd.Series:
    """Build portfolio weights for a single k. Falls back to EW on failure.

    Parameters
    ----------
    k : int
        Number of clusters.
    asset_matrix : np.ndarray
        Shape (n_assets, window_length) return matrix.
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).
    clustering_method : str
        ``'kmeans'`` or ``'kmedoids'``.
    weighting_method : str
        ``'ew'`` (equal weight) or ``'mv'`` (max-Sharpe within clusters).
    random_state : int
        Random seed forwarded to the clustering algorithm.
    dtw_dist : np.ndarray | None
        Precomputed DTW distance matrix; required when
        clustering_method='kmedoids', ignored otherwise.

    Returns
    -------
    pd.Series
        Non-negative portfolio weights indexed by ticker, summing to 1.0.
        Equal weights are returned on any clustering or optimisation failure.
    """
    n = window_returns.shape[1]
    ew_fallback = pd.Series(1.0 / n, index=window_returns.columns, dtype="float64")
    try:
        if clustering_method == "kmeans":
            labels = kmeans_labels(asset_matrix, n_clusters=k, random_state=random_state)
        elif clustering_method == "kmedoids":
            if dtw_dist is None:
                raise ValueError("dtw_dist must be provided for kmedoids")
            labels = kmedoids_labels_from_distance(
                dtw_dist, n_clusters=k, random_state=random_state
            )
        else:
            raise ValueError(f"Unknown clustering_method: {clustering_method!r}")

        if weighting_method == "ew":
            return _cluster_equal_weight_from_labels(labels, window_returns.columns)
        elif weighting_method == "mv":
            return _cluster_mv_from_labels(labels, window_returns)
        else:
            raise ValueError(f"Unknown weighting_method: {weighting_method!r}")
    except Exception:
        return ew_fallback


def _compute_all_k_weights(
    window_returns: pd.DataFrame,
    k_values: list[int],
    clustering_method: str,
    weighting_method: str,
    random_state: int,
    dtw_dist: np.ndarray | None,
) -> dict[int, pd.Series]:
    """Return a weights Series for every k in k_values.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).
    k_values : list[int]
        Candidate cluster counts to evaluate.
    clustering_method : str
        ``'kmeans'`` or ``'kmedoids'``.
    weighting_method : str
        ``'ew'`` or ``'mv'``.
    random_state : int
        Random seed forwarded to the clustering algorithm.
    dtw_dist : np.ndarray | None
        Precomputed DTW distance matrix; required for kmedoids.

    Returns
    -------
    dict[int, pd.Series]
        Mapping ``{k: weights_series}`` for each k in k_values.
    """
    asset_matrix = raw_return_representation(window_returns)
    return {
        k: _build_weights_for_k(
            k=k,
            asset_matrix=asset_matrix,
            window_returns=window_returns,
            clustering_method=clustering_method,
            weighting_method=weighting_method,
            random_state=random_state,
            dtw_dist=dtw_dist,
        )
        for k in k_values
    }


def _select_k(
    val_history: dict[int, list[float]],
    val_window: int,
    k_values: list[int],
) -> tuple[int, float]:
    """Select the k with the highest annualised Sharpe over the validation window.

    Called only after the warm-up, so val_history is guaranteed to have
    at least val_window entries for every k.  If all Sharpe values are
    -inf (e.g. zero variance) the first k in k_values is returned.

    Parameters
    ----------
    val_history : dict[int, list[float]]
        Accumulated monthly return lists keyed by k value.
    val_window : int
        Number of past months used to evaluate each k's Sharpe ratio.
    k_values : list[int]
        Candidate cluster counts; the fallback is the first element.

    Returns
    -------
    tuple[int, float]
        (selected_k, best_sharpe). best_sharpe is NaN when all Sharpe
        values were -inf.
    """
    best_k = k_values[0]
    best_sharpe = -np.inf

    for k in k_values:
        tail = pd.Series(val_history[k][-val_window:])
        sh = _sharpe_from_monthly(tail)
        if sh > best_sharpe:
            best_sharpe = sh
            best_k = k

    if np.isinf(best_sharpe) and best_sharpe < 0:
        return k_values[0], float("nan")

    return best_k, float(best_sharpe)


def _update_val_history(
    val_history: dict[int, list[float]],
    k_values: list[int],
    weights_by_k: dict[int, pd.Series],
    oos_returns: pd.DataFrame,
    asset_columns: pd.Index,
) -> None:
    """Append each k's realised monthly return for this out-of-sample period.

    Parameters
    ----------
    val_history : dict[int, list[float]]
        Accumulated monthly return lists keyed by k value; mutated in place.
    k_values : list[int]
        Candidate cluster counts to update.
    weights_by_k : dict[int, pd.Series]
        Portfolio weights for each k, indexed by ticker.
    oos_returns : pd.DataFrame
        Out-of-sample daily returns for the current period (T x N).
    asset_columns : pd.Index
        Full asset universe index used to align weights.
    """
    for k in k_values:
        w_k = weights_by_k[k].reindex(asset_columns).fillna(0.0)
        k_oos = oos_returns @ w_k
        val_history[k].append(float((1.0 + k_oos).prod() - 1.0))


# ── Public API ────────────────────────────────────────────────────────────────

def run_rolling_validation_backtest(
    returns: pd.DataFrame,
    clustering_method: str,
    weighting_method: str,
    val_window: int,
    estimation_window: int = 756,
    k_values: list[int] | None = None,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run a rolling validation backtest with adaptive k selection.

    The first ``val_window`` rebalances are a warm-up phase: all k portfolios
    are simulated to build history, but no output is produced.  The returned
    series therefore starts ``val_window`` rebalances later than a plain
    rolling backtest with the same estimation_window.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns matrix (T × N), no NaNs.
    clustering_method : str
        ``'kmeans'`` or ``'kmedoids'``.
    weighting_method : str
        ``'ew'`` (equal weight within/across clusters) or
        ``'mv'`` (equal weight across clusters, max-Sharpe within).
    val_window : int
        Number of past months used to evaluate Sharpe for k selection.
        Also determines the length of the warm-up phase that is skipped.
    estimation_window : int
        Trading days used for each rebalance calculation (~3 years = 756).
    k_values : list[int] | None
        Candidate k values.  Defaults to [2, 3, 4, 5, 6, 7, 8].
    random_state : int
        Random seed for clustering.
    dtw_n_jobs : int
        Parallel jobs for DTW computation (-1 = all cores).
    dtw_cache_dir : str | None
        Cache directory for DTW matrices; defaults to project dtw_cache/.
    show_progress : bool
        Show tqdm progress bar.
    progress_desc : str | None
        Label for the progress bar.

    Returns
    -------
    portfolio_returns : pd.Series
        Daily out-of-sample portfolio returns starting after the warm-up
        phase (name='portfolio_return').
    weights_history : pd.DataFrame
        Weights at each live rebalance date (index name='rebalance_date').
    selected_k_df : pd.DataFrame
        Per-rebalance k selection record (live phase only).
        Index: rebalance_date.
        Columns: selected_k, best_sharpe.
    """
    if k_values is None:
        k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]

    cache_dir = dtw_cache_dir or _DTW_CACHE_DIR
    rebalance_dates = get_monthly_rebalance_dates(returns, estimation_window)

    if len(rebalance_dates) <= val_window:
        raise ValueError(
            f"Not enough rebalance dates ({len(rebalance_dates)}) "
            f"for val_window={val_window}. Need > {val_window}."
        )

    # Per-k accumulated monthly return lists
    val_history: dict[int, list[float]] = {k: [] for k in k_values}

    portfolio_returns_list: list[pd.Series] = []
    weights_records: list[pd.Series] = []
    selected_k_records: list[dict] = []
    selected_k_dates: list[pd.Timestamp] = []

    rebalance_iter: object = rebalance_dates
    if show_progress:
        rebalance_iter = tqdm(
            rebalance_dates,
            desc=progress_desc or f"{clustering_method}_{weighting_method}_{val_window}m",
            unit="rebalance",
        )

    for i, rebalance_date in enumerate(rebalance_iter):  # type: ignore[union-attr]
        t = returns.index.get_loc(rebalance_date)
        window_returns = returns.iloc[t - estimation_window : t]

        # ── DTW matrix (shared across all k for kmedoids) ─────────────────────
        dtw_dist: np.ndarray | None = None
        if clustering_method == "kmedoids":
            asset_matrix = raw_return_representation(window_returns)
            cache_key = _make_dtw_cache_key(window_returns)
            dtw_dist = dtw_distance_matrix(
                asset_matrix,
                n_jobs=dtw_n_jobs,
                cache_dir=cache_dir,
                cache_key=cache_key,
            )

        # ── Weights for every k ───────────────────────────────────────────────
        weights_by_k = _compute_all_k_weights(
            window_returns=window_returns,
            k_values=k_values,
            clustering_method=clustering_method,
            weighting_method=weighting_method,
            random_state=random_state,
            dtw_dist=dtw_dist,
        )

        # ── OOS period ────────────────────────────────────────────────────────
        if i < len(rebalance_dates) - 1:
            next_rebalance_date = rebalance_dates[i + 1]
            oos_returns = returns.loc[rebalance_date:next_rebalance_date].iloc[1:]
        else:
            oos_returns = returns.loc[rebalance_date:].iloc[1:]

        if oos_returns.empty:
            continue

        # ── Warm-up phase: accumulate history but skip recording ──────────────
        if i < val_window:
            _update_val_history(val_history, k_values, weights_by_k, oos_returns, returns.columns)
            continue

        # ── Live phase ────────────────────────────────────────────────────────
        # Select k using history from the val_window periods strictly before i
        selected_k, best_sharpe = _select_k(val_history, val_window, k_values)

        # Record portfolio return for the selected k
        selected_weights = weights_by_k[selected_k].reindex(returns.columns).fillna(0.0)
        port_rets = oos_returns @ selected_weights
        port_rets.name = "portfolio_return"
        portfolio_returns_list.append(port_rets)

        weight_row = selected_weights.copy()
        weight_row.name = rebalance_date
        weights_records.append(weight_row)

        selected_k_records.append({"selected_k": selected_k, "best_sharpe": best_sharpe})
        selected_k_dates.append(rebalance_date)

        # Update val_history AFTER recording (used by the next iteration)
        _update_val_history(val_history, k_values, weights_by_k, oos_returns, returns.columns)

    if not portfolio_returns_list:
        raise ValueError("No portfolio returns were generated.")

    portfolio_returns = pd.concat(portfolio_returns_list).sort_index()
    portfolio_returns.name = "portfolio_return"

    weights_history = pd.DataFrame(weights_records)
    weights_history.index.name = "rebalance_date"

    selected_k_df = pd.DataFrame(
        selected_k_records,
        index=pd.DatetimeIndex(selected_k_dates),
    )
    selected_k_df.index.name = "rebalance_date"

    return portfolio_returns, weights_history, selected_k_df
