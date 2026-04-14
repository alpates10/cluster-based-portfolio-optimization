from __future__ import annotations

import numpy as np
import warnings
from sklearn.metrics import pairwise_distances

_GPU_FALLBACK_WARNED = False


def dtw_distance_matrix(
    asset_return_matrix: np.ndarray,
    n_jobs: int = -1,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Compute pairwise DTW distances between assets.

    Parameters
    ----------
    asset_return_matrix : np.ndarray
        Shape (n_assets, window_length)

    Returns
    -------
    np.ndarray
        Shape (n_assets, n_assets), symmetric distance matrix.
    """
    if asset_return_matrix.ndim != 2:
        raise ValueError("asset_return_matrix must be 2D")

    n_assets = asset_return_matrix.shape[0]
    if n_assets == 0:
        raise ValueError("asset_return_matrix has no assets")

    if not np.isfinite(asset_return_matrix).all():
        raise ValueError("asset_return_matrix contains non-finite values")

    try:
        from tslearn.metrics import cdist_dtw
    except ImportError as exc:
        raise ImportError(
            "DTW distance requires tslearn. Install it with: pip install tslearn"
        ) from exc

    dist = None
    if use_gpu:
        global _GPU_FALLBACK_WARNED
        try:
            # Best-effort GPU path (if tslearn backend + torch/cuda are available)
            dist = cdist_dtw(asset_return_matrix, n_jobs=n_jobs, be="pytorch")
        except Exception:
            if not _GPU_FALLBACK_WARNED:
                warnings.warn(
                    "GPU DTW backend is unavailable; falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _GPU_FALLBACK_WARNED = True

    if dist is None:
        try:
            dist = cdist_dtw(asset_return_matrix, n_jobs=n_jobs)
        except TypeError:
            # Older tslearn versions may not support n_jobs in this function.
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
    dtw_use_gpu: bool = False,
) -> np.ndarray:
    distance_key = distance.lower()
    if distance_key in {"euclidean", "l2"}:
        return euclidean_distance_matrix(asset_return_matrix)
    if distance_key in {"dtw"}:
        return dtw_distance_matrix(
            asset_return_matrix,
            n_jobs=dtw_n_jobs,
            use_gpu=dtw_use_gpu,
        )
    if distance_key in {"correlation", "corr"}:
        return correlation_distance_matrix(asset_return_matrix)

    raise ValueError(f"Unsupported distance: {distance}")
