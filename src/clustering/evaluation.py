from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import compute_distance_matrix


def cluster_size_balance(labels: np.ndarray) -> float:
    """
    Lower is better.
    Uses coefficient of variation = std(cluster_sizes) / mean(cluster_sizes).
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
    Distance-matrix variant of Davies-Bouldin index.
    Lower is better.
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
    Evaluate one (method, distance, k) configuration.
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
                sil = 0.0  # tek cluster: silhouette tanımsız, 0 olarak gösterilir
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
                sil = 0.0  # tek cluster: silhouette tanımsız, 0 olarak gösterilir
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
    Recommend k per (method, distance) using rank aggregation across windows.

    Ranking objective:
    - silhouette_score: higher better
    - davies_bouldin: lower better
    - cluster_balance: lower better
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
