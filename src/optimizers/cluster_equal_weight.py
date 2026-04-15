from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.clustering.pipeline import cluster_labels_kmeans_raw, cluster_labels_kmedoids_dtw
from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import correlation_distance_matrix, dtw_distance_matrix
from src.clustering.representations import raw_return_representation

_DTW_CACHE_DIR: str = str(
    Path(__file__).resolve().parents[2] / "data" / "processed" / "dtw_cache"
)

K_VALUES_ADAPTIVE = [2, 3, 4, 5, 6, 7, 8]


def _make_dtw_cache_key_for_window(window_returns: pd.DataFrame) -> str:
    idx = window_returns.index
    start = str(idx[0].date()) if hasattr(idx[0], "date") else str(idx[0])
    end = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
    tickers = ",".join(sorted(window_returns.columns.tolist()))
    raw = f"{start}|{end}|{tickers}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _select_adaptive_k(silhouette_scores: dict[int, float]) -> int:
    """Select k using the incremental-delta rule.

    Compute delta[k] = sil[k] - sil[k-1] for each consecutive pair starting
    from k=3.  Return the k whose delta is largest.  If all deltas are
    negative (monotone decline), fall back to k=2.
    """
    k_vals = sorted(silhouette_scores.keys())  # [2, 3, 4, 5, 6, 7, 8]
    deltas: dict[int, float] = {}
    for i in range(1, len(k_vals)):
        k = k_vals[i]
        deltas[k] = silhouette_scores[k] - silhouette_scores[k_vals[i - 1]]

    if all(d < 0 for d in deltas.values()):
        return k_vals[0]  # fallback to k=2

    return max(deltas, key=lambda k: deltas[k])


def _cluster_equal_weight_from_labels(
    labels: np.ndarray,
    asset_index: pd.Index,
) -> pd.Series:
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    if len(labels) != len(asset_index):
        raise ValueError("labels length must match number of assets")

    if pd.isna(labels).any():
        raise ValueError("labels contain NaN values")

    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    if n_clusters == 0:
        raise ValueError("No clusters were produced")

    # Use positional numpy assignment to avoid pandas .loc issues with
    # duplicate index labels (which cause values to be summed/overwritten
    # in unexpected ways, producing weights > 1).
    weights_arr = np.zeros(len(asset_index), dtype=float)
    cluster_budget = 1.0 / n_clusters

    for cluster_id in unique_clusters:
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            continue
        member_weight = cluster_budget / members.size
        weights_arr[members] = member_weight

    weights = pd.Series(weights_arr, index=asset_index, dtype="float64", name="weight")

    if (weights_arr < -1e-12).any():
        raise ValueError("Negative weights were generated")

    if np.isnan(weights_arr).any():
        raise ValueError("NaN weights were generated")

    total = float(weights_arr.sum())
    if total <= 0:
        raise ValueError("Non-positive total weight was generated")

    weights = weights / total

    # Sanity assertions — catch any arithmetic or indexing regression
    assert (weights.values >= -1e-9).all(), (
        f"Weight below 0 after normalisation: min={weights.min():.6f}"
    )
    assert (weights.values <= 1.0 + 1e-9).all(), (
        f"Weight above 1.0 after normalisation: max={weights.max():.6f}"
    )
    assert abs(weights.sum() - 1.0) < 1e-9, (
        f"Weights do not sum to 1.0: sum={weights.sum():.10f}"
    )

    return weights


def get_cluster_kmeans_equal_weight(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
) -> pd.Series:
    labels = cluster_labels_kmeans_raw(
        window_returns=window_returns,
        k_mode="global",
        global_k=global_k,
        random_state=random_state,
    )
    return _cluster_equal_weight_from_labels(labels, window_returns.columns)


def get_cluster_kmedoids_dtw_equal_weight(
    window_returns: pd.DataFrame,
    global_k: int = 5,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
) -> pd.Series:
    labels = cluster_labels_kmedoids_dtw(
        window_returns=window_returns,
        k_mode="global",
        global_k=global_k,
        random_state=random_state,
        dtw_n_jobs=dtw_n_jobs,
    )
    return _cluster_equal_weight_from_labels(labels, window_returns.columns)


def get_cluster_kmeans_adaptive_ew(
    window_returns: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.Series, dict]:
    """KMeans clustering with adaptive k selection based on silhouette scores.

    Returns
    -------
    (weights, metadata_dict) where metadata_dict contains
    ``selected_k`` (int) and ``silhouette_score`` (float).
    """
    asset_matrix = raw_return_representation(window_returns)
    corr_dist = correlation_distance_matrix(asset_matrix)

    sil_scores: dict[int, float] = {}
    for k in K_VALUES_ADAPTIVE:
        labels = kmeans_labels(asset_matrix, n_clusters=k, random_state=random_state)
        sil_scores[k] = float(silhouette_score(corr_dist, labels, metric="precomputed"))

    selected_k = _select_adaptive_k(sil_scores)
    final_labels = kmeans_labels(asset_matrix, n_clusters=selected_k, random_state=random_state)
    weights = _cluster_equal_weight_from_labels(final_labels, window_returns.columns)

    return weights, {"selected_k": selected_k, "silhouette_score": sil_scores[selected_k]}


def get_cluster_kmedoids_dtw_adaptive_ew(
    window_returns: pd.DataFrame,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
) -> tuple[pd.Series, dict]:
    """KMedoids-DTW clustering with adaptive k selection based on silhouette scores.

    The DTW distance matrix is computed once and cached in
    ``data/processed/dtw_cache/`` (same mechanism as the fixed-k variant).

    Returns
    -------
    (weights, metadata_dict) where metadata_dict contains
    ``selected_k`` (int) and ``silhouette_score`` (float).
    """
    asset_matrix = raw_return_representation(window_returns)
    cache_key = _make_dtw_cache_key_for_window(window_returns)
    dtw_dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=dtw_n_jobs,
        cache_dir=_DTW_CACHE_DIR,
        cache_key=cache_key,
    )

    sil_scores: dict[int, float] = {}
    for k in K_VALUES_ADAPTIVE:
        labels = kmedoids_labels_from_distance(dtw_dist, n_clusters=k, random_state=random_state)
        sil_scores[k] = float(silhouette_score(dtw_dist, labels, metric="precomputed"))

    selected_k = _select_adaptive_k(sil_scores)
    final_labels = kmedoids_labels_from_distance(dtw_dist, n_clusters=selected_k, random_state=random_state)
    weights = _cluster_equal_weight_from_labels(final_labels, window_returns.columns)

    return weights, {"selected_k": selected_k, "silhouette_score": sil_scores[selected_k]}
