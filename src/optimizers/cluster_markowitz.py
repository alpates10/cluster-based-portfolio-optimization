"""
src/optimizers/cluster_markowitz.py

Six cluster-then-Markowitz strategies with fixed k:
  A) markowitz_inter_ew_intra      — Markowitz between clusters, EW within
  B) ew_inter_markowitz_intra      — EW between clusters, Markowitz within
  C) markowitz_inter_markowitz_intra — Markowitz at both levels

Each strategy is provided for both KMeans (raw returns) and
KMedoids-DTW, giving 6 public functions.
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

_DTW_CACHE_DIR: str = str(
    Path(__file__).resolve().parents[2] / "data" / "processed" / "dtw_cache"
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_dtw_cache_key(window_returns: pd.DataFrame) -> str:
    idx   = window_returns.index
    start = str(idx[0].date())  if hasattr(idx[0],  "date") else str(idx[0])
    end   = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
    tickers = ",".join(sorted(window_returns.columns.tolist()))
    raw = f"{start}|{end}|{tickers}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _markowitz_or_ew(window_returns: pd.DataFrame) -> pd.Series:
    """Max-Sharpe portfolio; falls back to equal weight on any failure."""
    n  = len(window_returns.columns)
    ew = pd.Series(1.0 / n, index=window_returns.columns, dtype="float64")
    if n < 2:
        return ew
    try:
        mu = window_returns.mean() * 252
        S  = window_returns.cov()  * 252
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=S,
            weight_bounds=(0, 1),
            solver="SCS",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.max_sharpe(risk_free_rate=0.0)
        w = pd.Series(ef.weights, index=window_returns.columns, dtype="float64")
        w = w.clip(lower=0.0)
        total = w.sum()
        if total <= 0:
            return ew
        return w / total
    except Exception:
        return ew


def _build_cluster_map(labels: np.ndarray, asset_index: pd.Index) -> dict[int, list]:
    """cluster_id → [ticker, ...]  (sorted for determinism)."""
    cluster_map: dict[int, list] = {}
    for i, cid in enumerate(labels):
        cluster_map.setdefault(int(cid), []).append(asset_index[i])
    return cluster_map


def _get_labels_kmeans(
    window_returns: pd.DataFrame,
    global_k: int,
    random_state: int,
) -> np.ndarray:
    asset_matrix = raw_return_representation(window_returns)
    return kmeans_labels(asset_matrix, n_clusters=global_k, random_state=random_state)


def _get_labels_kmedoids(
    window_returns: pd.DataFrame,
    global_k: int,
    random_state: int,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
) -> np.ndarray:
    asset_matrix = raw_return_representation(window_returns)
    cache_key    = _make_dtw_cache_key(window_returns)
    dtw_dist     = dtw_distance_matrix(
        asset_matrix,
        n_jobs=dtw_n_jobs,
        cache_dir=dtw_cache_dir or _DTW_CACHE_DIR,
        cache_key=cache_key,
    )
    return kmedoids_labels_from_distance(
        dtw_dist, n_clusters=global_k, random_state=random_state
    )


def _normalise(weights_arr: np.ndarray, asset_index: pd.Index) -> pd.Series:
    total = weights_arr.sum()
    if total <= 0:
        n = len(asset_index)
        return pd.Series(1.0 / n, index=asset_index, dtype="float64")
    return pd.Series(weights_arr / total, index=asset_index, dtype="float64")


# ── Strategy implementations (shared between KMeans and KMedoids) ─────────────

def _markowitz_inter_ew_intra(
    window_returns: pd.DataFrame,
    labels: np.ndarray,
) -> pd.Series:
    """
    Strategy A: Markowitz between clusters, equal weight within.

    Representative series per cluster = mean return series of cluster members.
    Cluster weights from Markowitz on representatives.
    Within-cluster weight = cluster_weight / cluster_size.
    """
    cluster_map = _build_cluster_map(labels, window_returns.columns)
    clusters    = sorted(cluster_map.keys())

    # Representative return series per cluster
    rep_df = pd.DataFrame(
        {cid: window_returns[cluster_map[cid]].mean(axis=1) for cid in clusters}
    )
    cluster_weights = _markowitz_or_ew(rep_df)  # index = cluster ids

    col_pos   = {t: i for i, t in enumerate(window_returns.columns)}
    w_arr     = np.zeros(len(window_returns.columns), dtype=float)
    for cid in clusters:
        tickers  = cluster_map[cid]
        cw       = float(cluster_weights[cid])
        per_asset = cw / len(tickers)
        for t in tickers:
            w_arr[col_pos[t]] = per_asset

    return _normalise(w_arr, window_returns.columns)


def _ew_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    labels: np.ndarray,
) -> pd.Series:
    """
    Strategy B: Equal weight between clusters, Markowitz within.

    Cluster budget = 1 / n_clusters.
    Within-cluster: Markowitz on cluster assets.
    """
    cluster_map   = _build_cluster_map(labels, window_returns.columns)
    clusters      = sorted(cluster_map.keys())
    cluster_budget = 1.0 / len(clusters)

    col_pos = {t: i for i, t in enumerate(window_returns.columns)}
    w_arr   = np.zeros(len(window_returns.columns), dtype=float)

    for cid in clusters:
        tickers       = cluster_map[cid]
        intra_w       = _markowitz_or_ew(window_returns[tickers])
        for t in tickers:
            w_arr[col_pos[t]] = cluster_budget * float(intra_w[t])

    return _normalise(w_arr, window_returns.columns)


def _markowitz_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    labels: np.ndarray,
) -> pd.Series:
    """
    Strategy C: Markowitz at both levels.

    Cluster weights from Markowitz on representative series.
    Within-cluster weights from Markowitz on cluster assets.
    Final weight = cluster_weight * intra_weight.
    """
    cluster_map = _build_cluster_map(labels, window_returns.columns)
    clusters    = sorted(cluster_map.keys())

    rep_df          = pd.DataFrame(
        {cid: window_returns[cluster_map[cid]].mean(axis=1) for cid in clusters}
    )
    cluster_weights = _markowitz_or_ew(rep_df)

    col_pos = {t: i for i, t in enumerate(window_returns.columns)}
    w_arr   = np.zeros(len(window_returns.columns), dtype=float)

    for cid in clusters:
        tickers = cluster_map[cid]
        cw      = float(cluster_weights[cid])
        intra_w = _markowitz_or_ew(window_returns[tickers])
        for t in tickers:
            w_arr[col_pos[t]] = cw * float(intra_w[t])

    return _normalise(w_arr, window_returns.columns)


# ── Public API ────────────────────────────────────────────────────────────────

def get_cluster_kmeans_markowitz_inter_ew_intra(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
) -> pd.Series:
    labels = _get_labels_kmeans(window_returns, global_k, random_state)
    return _markowitz_inter_ew_intra(window_returns, labels)


def get_cluster_kmeans_ew_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
) -> pd.Series:
    labels = _get_labels_kmeans(window_returns, global_k, random_state)
    return _ew_inter_markowitz_intra(window_returns, labels)


def get_cluster_kmeans_markowitz_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
) -> pd.Series:
    labels = _get_labels_kmeans(window_returns, global_k, random_state)
    return _markowitz_inter_markowitz_intra(window_returns, labels)


def get_cluster_kmedoids_dtw_markowitz_inter_ew_intra(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
) -> pd.Series:
    labels = _get_labels_kmedoids(window_returns, global_k, random_state, dtw_n_jobs, dtw_cache_dir)
    return _markowitz_inter_ew_intra(window_returns, labels)


def get_cluster_kmedoids_dtw_ew_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
) -> pd.Series:
    labels = _get_labels_kmedoids(window_returns, global_k, random_state, dtw_n_jobs, dtw_cache_dir)
    return _ew_inter_markowitz_intra(window_returns, labels)


def get_cluster_kmedoids_dtw_markowitz_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
) -> pd.Series:
    labels = _get_labels_kmedoids(window_returns, global_k, random_state, dtw_n_jobs, dtw_cache_dir)
    return _markowitz_inter_markowitz_intra(window_returns, labels)
