from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import dtw_distance_matrix
from src.clustering.representations import raw_return_representation
from src.clustering.selection import resolve_k


def cluster_labels_kmeans_raw(
    window_returns: pd.DataFrame,
    k_mode: str = "global",
    global_k: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    asset_matrix = raw_return_representation(window_returns)
    n_clusters = resolve_k(
        n_assets=asset_matrix.shape[0],
        mode=k_mode,
        global_k=global_k,
    )
    return kmeans_labels(
        asset_return_matrix=asset_matrix,
        n_clusters=n_clusters,
        random_state=random_state,
    )


def cluster_labels_kmedoids_dtw(
    window_returns: pd.DataFrame,
    k_mode: str = "global",
    global_k: int = 5,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
    dtw_use_gpu: bool = False,
) -> np.ndarray:
    asset_matrix = raw_return_representation(window_returns)
    n_clusters = resolve_k(
        n_assets=asset_matrix.shape[0],
        mode=k_mode,
        global_k=global_k,
    )
    dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=dtw_n_jobs,
        use_gpu=dtw_use_gpu,
    )
    return kmedoids_labels_from_distance(
        distance_matrix=dist,
        n_clusters=n_clusters,
        random_state=random_state,
    )
