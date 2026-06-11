from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import compute_distance_matrix


def cluster_size_balance(labels: np.ndarray) -> float:
    """
    Compute cluster size balance as coefficient of variation (lower is better).

    Calculated as std(cluster_sizes) / mean(cluster_sizes).

    Parameters
    ----------
    labels : np.ndarray
        One-dimensional array of cluster labels.

    Returns
    -------
    float
        Coefficient of variation of cluster sizes; NaN if there are no clusters
        or mean cluster size is zero.
    """
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size == 0:
        return np.nan
    mean_size = counts.mean()
    if np.isclose(mean_size, 0.0):
        return np.nan
    return float(counts.std(ddof=0) / mean_size)


def davies_bouldin_from_distance(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Davies-Bouldin index from a precomputed distance matrix (lower is better).

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square symmetric distance matrix of shape (n_assets, n_assets).
    labels : np.ndarray
        One-dimensional array of cluster labels, length n_assets.

    Returns
    -------
    float
        Davies-Bouldin index; NaN if fewer than two clusters exist or
        any cluster is empty.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return np.nan

    medoids: list[int] = []
    scatters: list[float] = []

    for lbl in unique_labels:
        members = np.where(labels == lbl)[0]
        if members.size == 0:
            return np.nan
        intra = distance_matrix[np.ix_(members, members)]
        total = intra.sum(axis=1)
        medoid = int(members[np.argmin(total)])
        medoids.append(medoid)
        scatters.append(float(distance_matrix[members, medoid].mean()))

    medoids = list(medoids)
    scatters_arr = np.asarray(scatters, dtype=float)

    db_terms = []
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i == j:
                continue
            sep = float(distance_matrix[medoids[i], medoids[j]])
            if sep <= 0:
                continue
            ratio = float((scatters_arr[i] + scatters_arr[j]) / sep)
            if ratio > max_ratio:
                max_ratio = ratio
        if np.isfinite(max_ratio):
            db_terms.append(max_ratio)

    if not db_terms:
        return np.nan
    return float(np.mean(db_terms))


def evaluate_clustering_config(
    asset_return_matrix: np.ndarray,
    method: str,
    distance: str,
    k: int,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
    dtw_cache_key: str | None = None,
) -> dict:
    """
    Evaluate a single (method, distance, k) clustering configuration.

    Parameters
    ----------
    asset_return_matrix : np.ndarray
        Shape (n_assets, window_length).
    method : str
        Clustering algorithm: 'kmeans' or 'kmedoids'.
    distance : str
        Distance metric: 'euclidean', 'dtw', or 'correlation'.
    k : int
        Number of clusters to evaluate.
    random_state : int
        Random seed for reproducible clustering.
    dtw_n_jobs : int
        Parallel jobs for DTW computation (-1 = all cores).
    dtw_cache_dir : str | None
        Directory for DTW distance matrix cache.
    dtw_cache_key : str | None
        Explicit cache filename stem for DTW.

    Returns
    -------
    dict
        Keys: 'status' ('ok', 'skip', or 'error'), 'notes', 'silhouette_score',
        'davies_bouldin', 'cluster_balance', 'n_clusters_found'.
    """
    method_key = method.lower()
    distance_key = distance.lower()

    try:
        if method_key == "kmeans":
            if distance_key not in {"euclidean", "raw", "raw_euclidean"}:
                return {
                    "status": "skip",
                    "notes": (
                        f"Unsupported combination: method={method}, distance={distance}. "
                        "KMeans requires feature vectors (euclidean/raw)."
                    ),
                }

            labels = kmeans_labels(
                asset_return_matrix=asset_return_matrix,
                n_clusters=k,
                random_state=random_state,
            )
            n_clusters_found = int(np.unique(labels).size)

            sil = np.nan
            db = np.nan
            if k == 1:
                sil = 0.0  # One cluster: silhouette is undefined, displayed as 0.
            elif 2 <= n_clusters_found < asset_return_matrix.shape[0]:
                sil = float(silhouette_score(asset_return_matrix, labels, metric="euclidean"))
                db = float(davies_bouldin_score(asset_return_matrix, labels))

            return {
                "status": "ok",
                "notes": "",
                "silhouette_score": sil,
                "davies_bouldin": db,
                "cluster_balance": cluster_size_balance(labels),
                "n_clusters_found": n_clusters_found,
            }

        if method_key == "kmedoids":
            dist = compute_distance_matrix(
                asset_return_matrix=asset_return_matrix,
                distance=distance_key,
                dtw_n_jobs=dtw_n_jobs,
                dtw_cache_dir=dtw_cache_dir,
                dtw_cache_key=dtw_cache_key,
            )
            labels = kmedoids_labels_from_distance(
                distance_matrix=dist.copy(),
                n_clusters=k,
                random_state=random_state,
            )
            n_clusters_found = int(np.unique(labels).size)

            sil = np.nan
            db = np.nan
            if k == 1:
                sil = 0.0  # One cluster: silhouette is undefined, displayed as 0.
            elif 2 <= n_clusters_found < asset_return_matrix.shape[0]:
                sil = float(silhouette_score(dist, labels, metric="precomputed"))
                db = float(davies_bouldin_from_distance(dist, labels))

            return {
                "status": "ok",
                "notes": "",
                "silhouette_score": sil,
                "davies_bouldin": db,
                "cluster_balance": cluster_size_balance(labels),
                "n_clusters_found": n_clusters_found,
            }

        return {
            "status": "skip",
            "notes": f"Unsupported method: {method}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "notes": str(exc),
        }


def recommend_global_k(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recommend the best k per (method, distance) pair using rank aggregation across windows.

    Ranking objective: silhouette_score (higher is better), davies_bouldin (lower is better),
    cluster_balance (lower is better). Each metric is ranked per window and the composite
    mean rank across windows determines the recommended k.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of evaluate_clustering_config calls; must contain columns
        'status', 'method', 'distance', 'k', 'window_id', 'silhouette_score',
        'davies_bouldin', and 'cluster_balance'.

    Returns
    -------
    pd.DataFrame
        One row per (method, distance) with columns 'method', 'distance',
        'recommended_k', 'mean_composite_rank', and 'reason'.
        Returns an empty DataFrame if no valid results exist.
    """
    if results_df.empty:
        return pd.DataFrame()

    valid = results_df[results_df["status"] == "ok"].copy()
    valid = valid.dropna(
        subset=["silhouette_score", "davies_bouldin", "cluster_balance"]
    )
    if valid.empty:
        return pd.DataFrame()

    ranked_frames: list[pd.DataFrame] = []
    group_cols = ["method", "distance", "window_id"]
    for _, grp in valid.groupby(group_cols, dropna=False):
        g = grp.copy()
        g["rank_silhouette"] = g["silhouette_score"].rank(ascending=False, method="average")
        g["rank_db"] = g["davies_bouldin"].rank(ascending=True, method="average")
        g["rank_balance"] = g["cluster_balance"].rank(ascending=True, method="average")
        g["composite_rank"] = g[["rank_silhouette", "rank_db", "rank_balance"]].mean(axis=1)
        ranked_frames.append(g)

    ranked = pd.concat(ranked_frames, ignore_index=True)
    agg = (
        ranked.groupby(["method", "distance", "k"], as_index=False)["composite_rank"]
        .mean()
        .rename(columns={"composite_rank": "mean_composite_rank"})
    )

    rows: list[dict] = []
    for (method, distance), grp in agg.groupby(["method", "distance"], dropna=False):
        best = grp.sort_values(["mean_composite_rank", "k"]).iloc[0]
        rows.append(
            {
                "method": method,
                "distance": distance,
                "recommended_k": int(best["k"]),
                "mean_composite_rank": float(best["mean_composite_rank"]),
                "reason": (
                    "Chosen by lowest average composite rank across windows "
                    "(silhouette high, Davies-Bouldin low, balance low)."
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["method", "distance"]).reset_index(drop=True)
