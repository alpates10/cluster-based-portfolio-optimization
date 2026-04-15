from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances


def dtw_distance_matrix(
    asset_return_matrix: np.ndarray,
    n_jobs: int = -1,
    cache_dir: str | None = None,
    cache_key: str | None = None,
) -> np.ndarray:
    """
    Compute pairwise DTW distances between assets (CPU, via tslearn).

    Parameters
    ----------
    asset_return_matrix : np.ndarray
        Shape (n_assets, window_length)
    n_jobs : int
        Number of parallel jobs passed to tslearn's cdist_dtw (-1 = all cores).
    cache_dir : str | None
        If given, cache the result as <cache_dir>/<key>.npy and load on
        subsequent calls for the same window (skipping recomputation).
    cache_key : str | None
        Explicit cache filename stem.  If omitted and cache_dir is set, a
        SHA-256 of the matrix bytes is used as the key.

    Returns
    -------
    np.ndarray
        Shape (n_assets, n_assets), symmetric distance matrix, float64.
    """
    if asset_return_matrix.ndim != 2:
        raise ValueError("asset_return_matrix must be 2D")

    n_assets = asset_return_matrix.shape[0]
    if n_assets == 0:
        raise ValueError("asset_return_matrix has no assets")

    if not np.isfinite(asset_return_matrix).all():
        raise ValueError("asset_return_matrix contains non-finite values")

    # ── Cache: try to load ────────────────────────────────────────────────────
    cache_path: Path | None = None
    if cache_dir is not None:
        key = (
            cache_key
            if cache_key
            else hashlib.sha256(asset_return_matrix.tobytes()).hexdigest()[:24]
        )
        cache_path = Path(cache_dir) / f"{key}.npy"
        if cache_path.exists():
            loaded = np.load(str(cache_path))
            if loaded.shape == (n_assets, n_assets):
                return loaded
            # Shape mismatch → recompute and overwrite
    # ─────────────────────────────────────────────────────────────────────────

    try:
        from tslearn.metrics import cdist_dtw
    except ImportError as exc:
        raise ImportError(
            "DTW distance requires tslearn. Install it with: pip install tslearn"
        ) from exc

    try:
        dist = cdist_dtw(asset_return_matrix, n_jobs=n_jobs)
    except TypeError:
        # Older tslearn versions may not support the n_jobs argument.
        dist = cdist_dtw(asset_return_matrix)

    dist = np.asarray(dist, dtype=float)

    if dist.shape != (n_assets, n_assets):
        raise ValueError("Invalid DTW distance matrix shape")

    # Numerical guardrails
    dist = np.maximum(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    if not np.isfinite(dist).all():
        raise ValueError("DTW distance matrix contains non-finite values")

    # ── Cache: save ───────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), dist)
    # ─────────────────────────────────────────────────────────────────────────

    return dist


def euclidean_distance_matrix(asset_return_matrix: np.ndarray) -> np.ndarray:
    if asset_return_matrix.ndim != 2:
        raise ValueError("asset_return_matrix must be 2D")
    dist = pairwise_distances(asset_return_matrix, metric="euclidean")
    dist = np.asarray(dist, dtype=float)
    dist = np.maximum(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def correlation_distance_matrix(asset_return_matrix: np.ndarray) -> np.ndarray:
    if asset_return_matrix.ndim != 2:
        raise ValueError("asset_return_matrix must be 2D")
    dist = pairwise_distances(asset_return_matrix, metric="correlation")
    dist = np.asarray(dist, dtype=float)
    dist = np.maximum(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_distance_matrix(
    asset_return_matrix: np.ndarray,
    distance: str,
    dtw_n_jobs: int = -1,
    dtw_cache_dir: str | None = None,
    dtw_cache_key: str | None = None,
) -> np.ndarray:
    distance_key = distance.lower()
    if distance_key in {"euclidean", "l2"}:
        return euclidean_distance_matrix(asset_return_matrix)
    if distance_key in {"dtw"}:
        return dtw_distance_matrix(
            asset_return_matrix,
            n_jobs=dtw_n_jobs,
            cache_dir=dtw_cache_dir,
            cache_key=dtw_cache_key,
        )
    if distance_key in {"correlation", "corr"}:
        return correlation_distance_matrix(asset_return_matrix)

    raise ValueError(f"Unsupported distance: {distance}")
