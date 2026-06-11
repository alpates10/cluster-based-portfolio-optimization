"""
src/optimizers/cluster_rolling_validation.py

Rolling-validation k-selection strategy functions.

At each rebalance, k is chosen by evaluating the past val_window months of
per-k portfolio returns and selecting the k with the highest annualised Sharpe
ratio.  Falls back to k=5 when there is insufficient history.

Public strategy functions
-------------------------
get_cluster_kmeans_rolling_val_ew
get_cluster_kmeans_rolling_val_mv
get_cluster_kmedoids_dtw_rolling_val_ew
get_cluster_kmedoids_dtw_rolling_val_mv

Each function returns a (weights, metadata_dict) tuple where metadata_dict
contains ``selected_k`` (int) and ``best_sharpe`` (float).

Parameters shared by all functions
-----------------------------------
window_returns : pd.DataFrame
    Daily returns matrix for the estimation window (T × N).
val_returns_by_k : dict[int, pd.Series]
    Accumulated monthly return history for each k, keyed by k value.
    Built and maintained by the backtest engine; passed in at call time.
val_window : int
    Number of past months used to evaluate Sharpe for k selection.
random_state : int
    Random seed for clustering.
dtw_cache_dir : str | None   (kmedoids variants only)
    Directory for DTW distance matrix cache.  Defaults to the project-level
    data/processed/dtw_cache/ directory.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import dtw_distance_matrix
from src.clustering.representations import raw_return_representation
from src.optimizers.cluster_equal_weight import _cluster_equal_weight_from_labels

K_VALUES: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
FALLBACK_K: int = 5

_DTW_CACHE_DIR: str = str(
    Path(__file__).resolve().parents[2] / "data" / "processed" / "dtw_cache"
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_dtw_cache_key(window_returns: pd.DataFrame) -> str:
    """
    Build a deterministic SHA-256 digest from the window bounds and ticker list.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Returns matrix indexed by date with ticker symbols as columns.

    Returns
    -------
    str
        A 24-character hexadecimal digest suitable for use as a cache filename stem.
    """
    idx = window_returns.index
    start = str(idx[0].date()) if hasattr(idx[0], "date") else str(idx[0])
    end = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
    tickers = ",".join(sorted(window_returns.columns.tolist()))
    raw = f"{start}|{end}|{tickers}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _sharpe_from_monthly(monthly_returns: pd.Series) -> float:
    """
    Compute annualised Sharpe ratio from monthly returns (risk-free rate = 0).

    Parameters
    ----------
    monthly_returns : pd.Series
        Monthly return series.

    Returns
    -------
    float
        Annualised Sharpe ratio, or -inf when fewer than two observations
        exist or volatility is zero.
    """
    if len(monthly_returns) < 2:
        return -np.inf
    mu = float(monthly_returns.mean()) * 12.0
    sigma = float(monthly_returns.std(ddof=1)) * np.sqrt(12.0)
    if np.isclose(sigma, 0.0):
        return -np.inf
    return mu / sigma


def _select_k_by_sharpe(
    val_returns_by_k: dict[int, pd.Series],
    val_window: int,
) -> tuple[int, float]:
    """
    Select the k with the highest annualised Sharpe over the last val_window months.

    Falls back to FALLBACK_K when any k lacks val_window months of history
    (i.e., during the warm-up phase).

    Parameters
    ----------
    val_returns_by_k : dict[int, pd.Series]
        Accumulated monthly return history keyed by k value.
    val_window : int
        Number of past months used to evaluate Sharpe for each k.

    Returns
    -------
    tuple[int, float]
        (selected_k, best_sharpe). best_sharpe is NaN when falling back
        to FALLBACK_K due to insufficient history.
    """
    if not val_returns_by_k:
        return FALLBACK_K, float("nan")

    # All k values must have at least val_window months
    for series in val_returns_by_k.values():
        if len(series) < val_window:
            return FALLBACK_K, float("nan")

    best_k = FALLBACK_K
    best_sharpe = -np.inf

    for k, series in val_returns_by_k.items():
        tail = series.iloc[-val_window:]
        sh = _sharpe_from_monthly(tail)
        if sh > best_sharpe:
            best_sharpe = sh
            best_k = k

    if np.isinf(best_sharpe) and best_sharpe < 0:
        # All Sharpe values were -inf (e.g. zero variance)
        return FALLBACK_K, float("nan")

    return best_k, float(best_sharpe)


def _markowitz_or_ew(cluster_returns: pd.DataFrame) -> pd.Series:
    """
    Compute max-Sharpe weights within a cluster; fall back to equal weight on any failure.

    Parameters
    ----------
    cluster_returns : pd.DataFrame
        Daily returns matrix for the assets in a single cluster.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker, summing to 1.0.
    """
    n = len(cluster_returns.columns)
    ew = pd.Series(1.0 / n, index=cluster_returns.columns, dtype="float64")
    if n < 2:
        return ew
    try:
        mu = cluster_returns.mean() * 252.0
        S = cluster_returns.cov() * 252.0
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=S,
            weight_bounds=(0, 1),
            solver="SCS",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.max_sharpe(risk_free_rate=0.0)
        w = pd.Series(ef.weights, index=cluster_returns.columns, dtype="float64")
        w = w.clip(lower=0.0)
        total = w.sum()
        if total <= 0:
            return ew
        return w / total
    except Exception:
        return ew


def _cluster_mv_from_labels(
    labels: np.ndarray,
    window_returns: pd.DataFrame,
) -> pd.Series:
    """
    Build portfolio weights: equal weight between clusters, max-Sharpe within each cluster.

    Falls back to equal weight within a cluster when MV optimisation fails.

    Parameters
    ----------
    labels : np.ndarray
        One-dimensional array of cluster labels, length n_assets.
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).

    Returns
    -------
    pd.Series
        Non-negative portfolio weights indexed by ticker, summing to 1.0.
    """
    asset_index = window_returns.columns
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    cluster_budget = 1.0 / n_clusters

    weights_arr = np.zeros(len(asset_index), dtype=float)

    for cid in unique_clusters:
        member_positions = np.where(labels == cid)[0]
        members = asset_index[member_positions]
        cluster_returns = window_returns[members]
        w_within = _markowitz_or_ew(cluster_returns)
        for pos, ticker in zip(member_positions, members):
            weights_arr[pos] = cluster_budget * float(w_within[ticker])

    total = float(weights_arr.sum())
    if total <= 0.0:
        weights_arr[:] = 1.0 / len(asset_index)
    else:
        weights_arr /= total

    weights = pd.Series(weights_arr, index=asset_index, dtype="float64", name="weight")

    assert abs(weights.sum() - 1.0) < 1e-9, (
        f"_cluster_mv_from_labels: weights sum to {weights.sum():.10f}"
    )
    return weights


# ── Public strategy functions ─────────────────────────────────────────────────

def get_cluster_kmeans_rolling_val_ew(
    window_returns: pd.DataFrame,
    val_returns_by_k: dict[int, pd.Series],
    val_window: int,
    random_state: int = 42,
) -> tuple[pd.Series, dict]:
    """
    KMeans clustering with equal-weight allocation and rolling-validation k selection.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).
    val_returns_by_k : dict[int, pd.Series]
        Accumulated monthly return history keyed by k value.
    val_window : int
        Number of past months used to evaluate Sharpe for k selection.
    random_state : int
        Random seed for reproducible clustering.

    Returns
    -------
    tuple[pd.Series, dict]
        Portfolio weights indexed by ticker, and metadata dict with
        'selected_k' (int) and 'best_sharpe' (float).
    """
    selected_k, best_sharpe = _select_k_by_sharpe(val_returns_by_k, val_window)

    asset_matrix = raw_return_representation(window_returns)
    labels = kmeans_labels(asset_matrix, n_clusters=selected_k, random_state=random_state)
    weights = _cluster_equal_weight_from_labels(labels, window_returns.columns)

    return weights, {"selected_k": selected_k, "best_sharpe": best_sharpe}


def get_cluster_kmeans_rolling_val_mv(
    window_returns: pd.DataFrame,
    val_returns_by_k: dict[int, pd.Series],
    val_window: int,
    random_state: int = 42,
) -> tuple[pd.Series, dict]:
    """
    KMeans clustering with MV allocation and rolling-validation k selection.

    Uses equal weight between clusters and max-Sharpe within each cluster.
    Falls back to equal weight within a cluster when MV optimisation fails.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).
    val_returns_by_k : dict[int, pd.Series]
        Accumulated monthly return history keyed by k value.
    val_window : int
        Number of past months used to evaluate Sharpe for k selection.
    random_state : int
        Random seed for reproducible clustering.

    Returns
    -------
    tuple[pd.Series, dict]
        Portfolio weights indexed by ticker, and metadata dict with
        'selected_k' (int) and 'best_sharpe' (float).
    """
    selected_k, best_sharpe = _select_k_by_sharpe(val_returns_by_k, val_window)

    asset_matrix = raw_return_representation(window_returns)
    labels = kmeans_labels(asset_matrix, n_clusters=selected_k, random_state=random_state)
    weights = _cluster_mv_from_labels(labels, window_returns)

    return weights, {"selected_k": selected_k, "best_sharpe": best_sharpe}


def get_cluster_kmedoids_dtw_rolling_val_ew(
    window_returns: pd.DataFrame,
    val_returns_by_k: dict[int, pd.Series],
    val_window: int,
    random_state: int = 42,
    dtw_cache_dir: str | None = None,
) -> tuple[pd.Series, dict]:
    """
    KMedoids-DTW clustering with equal-weight allocation and rolling-validation k selection.

    The DTW distance matrix is cached so repeated calls for the same
    estimation window are cheap.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).
    val_returns_by_k : dict[int, pd.Series]
        Accumulated monthly return history keyed by k value.
    val_window : int
        Number of past months used to evaluate Sharpe for k selection.
    random_state : int
        Random seed for reproducible clustering.
    dtw_cache_dir : str | None
        Directory for the DTW distance matrix cache; None uses the project default.

    Returns
    -------
    tuple[pd.Series, dict]
        Portfolio weights indexed by ticker, and metadata dict with
        'selected_k' (int) and 'best_sharpe' (float).
    """
    selected_k, best_sharpe = _select_k_by_sharpe(val_returns_by_k, val_window)

    cache_dir = dtw_cache_dir or _DTW_CACHE_DIR
    asset_matrix = raw_return_representation(window_returns)
    cache_key = _make_dtw_cache_key(window_returns)
    dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=-1,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    labels = kmedoids_labels_from_distance(dist, n_clusters=selected_k, random_state=random_state)
    weights = _cluster_equal_weight_from_labels(labels, window_returns.columns)

    return weights, {"selected_k": selected_k, "best_sharpe": best_sharpe}


def get_cluster_kmedoids_dtw_rolling_val_mv(
    window_returns: pd.DataFrame,
    val_returns_by_k: dict[int, pd.Series],
    val_window: int,
    random_state: int = 42,
    dtw_cache_dir: str | None = None,
) -> tuple[pd.Series, dict]:
    """
    KMedoids-DTW clustering with MV allocation and rolling-validation k selection.

    Uses equal weight between clusters and max-Sharpe within each cluster.
    Falls back to equal weight within a cluster when MV optimisation fails.

    Parameters
    ----------
    window_returns : pd.DataFrame
        Daily returns matrix for the estimation window (T x N).
    val_returns_by_k : dict[int, pd.Series]
        Accumulated monthly return history keyed by k value.
    val_window : int
        Number of past months used to evaluate Sharpe for k selection.
    random_state : int
        Random seed for reproducible clustering.
    dtw_cache_dir : str | None
        Directory for the DTW distance matrix cache; None uses the project default.

    Returns
    -------
    tuple[pd.Series, dict]
        Portfolio weights indexed by ticker, and metadata dict with
        'selected_k' (int) and 'best_sharpe' (float).
    """
    selected_k, best_sharpe = _select_k_by_sharpe(val_returns_by_k, val_window)

    cache_dir = dtw_cache_dir or _DTW_CACHE_DIR
    asset_matrix = raw_return_representation(window_returns)
    cache_key = _make_dtw_cache_key(window_returns)
    dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=-1,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    labels = kmedoids_labels_from_distance(dist, n_clusters=selected_k, random_state=random_state)
    weights = _cluster_mv_from_labels(labels, window_returns)

    return weights, {"selected_k": selected_k, "best_sharpe": best_sharpe}
