from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import dtw_distance_matrix
from src.clustering.representations import raw_return_representation
from src.clustering.selection import resolve_k

# Default cache directory: <project_root>/data/processed/dtw_cache/
# pipeline.py lives at src/clustering/pipeline.py → parents[2] = project root
_DTW_CACHE_DIR: str = str(
    Path(__file__).resolve().parents[2] / "data" / "processed" / "dtw_cache"
)


def _make_dtw_cache_key(window_returns: pd.DataFrame) -> str:
    """
    Build a deterministic hash from (start_date, end_date, sorted_tickers).
    This uniquely identifies a rebalance window so the DTW matrix can be
    cached on disk and reused across multiple k values or runs.
    """
    idx = window_returns.index
    start = str(idx[0].date()) if hasattr(idx[0], "date") else str(idx[0])
    end = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
    tickers = ",".join(sorted(window_returns.columns.tolist()))
    raw = f"{start}|{end}|{tickers}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


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
    cache_dir: str | None = _DTW_CACHE_DIR,
) -> np.ndarray:
    """
    Cluster assets with K-Medoids on a precomputed DTW distance matrix (CPU).

    The DTW matrix is cached under *cache_dir* (default:
    data/processed/dtw_cache/) keyed by (start_date, end_date, tickers),
    so subsequent calls for the same window skip the expensive computation.
    """
    asset_matrix = raw_return_representation(window_returns)
    n_clusters = resolve_k(
        n_assets=asset_matrix.shape[0],
        mode=k_mode,
        global_k=global_k,
    )

    cache_key: str | None = None
    if cache_dir is not None:
        cache_key = _make_dtw_cache_key(window_returns)

    dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=dtw_n_jobs,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    return kmedoids_labels_from_distance(
        distance_matrix=dist,
        n_clusters=n_clusters,
        random_state=random_state,
    )
