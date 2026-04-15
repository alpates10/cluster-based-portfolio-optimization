from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering.pipeline import cluster_labels_kmeans_raw, cluster_labels_kmedoids_dtw


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
