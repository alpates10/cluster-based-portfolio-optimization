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

    weights = pd.Series(0.0, index=asset_index, dtype="float64", name="weight")
    cluster_budget = 1.0 / n_clusters

    for cluster_id in unique_clusters:
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            continue

        member_weight = cluster_budget / members.size
        member_assets = asset_index[members]
        weights.loc[member_assets] = member_weight

    if (weights < -1e-12).any():
        raise ValueError("Negative weights were generated")

    if weights.isna().any():
        raise ValueError("NaN weights were generated")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Non-positive total weight was generated")

    weights = weights / total
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
    dtw_use_gpu: bool = True,
) -> pd.Series:
    labels = cluster_labels_kmedoids_dtw(
        window_returns=window_returns,
        k_mode="global",
        global_k=global_k,
        random_state=random_state,
        dtw_n_jobs=dtw_n_jobs,
        dtw_use_gpu=dtw_use_gpu,
    )
    return _cluster_equal_weight_from_labels(labels, window_returns.columns)
