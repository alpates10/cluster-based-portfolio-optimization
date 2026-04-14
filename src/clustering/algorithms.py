from __future__ import annotations

import warnings
import numpy as np
from sklearn.cluster import KMeans


def kmeans_labels(
    asset_return_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
) -> np.ndarray:
    """
    Cluster assets with KMeans on raw vectors.
    """
    if asset_return_matrix.ndim != 2:
        raise ValueError("asset_return_matrix must be 2D")

    n_assets = asset_return_matrix.shape[0]
    if n_clusters < 1 or n_clusters > n_assets:
        raise ValueError(f"n_clusters must be in [1, {n_assets}]")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(asset_return_matrix)
    return np.asarray(labels, dtype=int)


def _kmedoids_labels_pam(
    distance_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Lightweight PAM-style fallback when sklearn-extra is unavailable.
    """
    n_assets = distance_matrix.shape[0]
    rng = np.random.default_rng(random_state)

    medoids = rng.choice(n_assets, size=n_clusters, replace=False)

    def assign(medoids_idx: np.ndarray) -> np.ndarray:
        return np.argmin(distance_matrix[:, medoids_idx], axis=1)

    def dedupe_medoids(medoids_idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique = list(dict.fromkeys(medoids_idx.tolist()))
        if len(unique) == n_clusters:
            return np.asarray(unique, dtype=int)

        used = set(unique)
        candidates = [i for i in range(n_assets) if i not in used]
        if not candidates:
            return np.asarray(unique[:n_clusters], dtype=int)

        min_dist_to_unique = distance_matrix[:, np.asarray(unique, dtype=int)].min(axis=1)
        refill_order = np.argsort(-min_dist_to_unique)  # farthest first
        for idx in refill_order:
            if idx in used:
                continue
            unique.append(int(idx))
            used.add(int(idx))
            if len(unique) == n_clusters:
                break

        return np.asarray(unique[:n_clusters], dtype=int)

    for _ in range(max_iter):
        labels = assign(medoids)
        new_medoids = medoids.copy()

        for cluster_id in range(n_clusters):
            members = np.where(labels == cluster_id)[0]

            if members.size == 0:
                min_dist = distance_matrix[:, medoids].min(axis=1)
                replacement = int(np.argmax(min_dist))
                new_medoids[cluster_id] = replacement
                continue

            intra = distance_matrix[np.ix_(members, members)]
            total_cost = intra.sum(axis=1)
            best_member = int(members[np.argmin(total_cost)])
            new_medoids[cluster_id] = best_member

        new_medoids = dedupe_medoids(new_medoids, labels)
        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            medoids = new_medoids
            break

        medoids = new_medoids

    labels = assign(medoids)
    return np.asarray(labels, dtype=int)


def kmedoids_labels_from_distance(
    distance_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster with precomputed distances using KMedoids.
    """
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D matrix")

    n_assets = distance_matrix.shape[0]
    if n_clusters < 1 or n_clusters > n_assets:
        raise ValueError(f"n_clusters must be in [1, {n_assets}]")

    if not np.isfinite(distance_matrix).all():
        raise ValueError("distance_matrix contains non-finite values")

    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
        raise ValueError("distance_matrix must be symmetric")

    np.fill_diagonal(distance_matrix, 0.0)

    try:
        from sklearn_extra.cluster import KMedoids

        model = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            method="pam",
            init="k-medoids++",
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Function stable_cumsum is deprecated; "
                    r"`sklearn\.utils\.extmath\.stable_cumsum` is deprecated.*"
                ),
                category=FutureWarning,
            )
            labels = model.fit_predict(distance_matrix)
        return np.asarray(labels, dtype=int)
    except ImportError:
        return _kmedoids_labels_pam(
            distance_matrix=distance_matrix,
            n_clusters=n_clusters,
            random_state=random_state,
        )
