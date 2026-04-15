"""
src/optimizers/cluster_markowitz_adaptive.py

Six cluster-then-Markowitz strategies with adaptive k selection:
  A) markowitz_inter_ew_intra
  B) ew_inter_markowitz_intra
  C) markowitz_inter_markowitz_intra

Each strategy is provided for both KMeans and KMedoids-DTW.
Each function returns (weights: pd.Series, {"selected_k": int, "silhouette_score": float}).

Adaptive k: for k in [2..8], compute silhouette with the appropriate distance matrix,
then select k via the delta rule (same as cluster_equal_weight.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import correlation_distance_matrix, dtw_distance_matrix
from src.clustering.representations import raw_return_representation

# Re-use the adaptive-k helpers from cluster_equal_weight without duplication
from src.optimizers.cluster_equal_weight import (
    K_VALUES_ADAPTIVE,
    _select_adaptive_k,
    _DTW_CACHE_DIR,
    _make_dtw_cache_key_for_window,
)

# Re-use the shared strategy implementations from cluster_markowitz
from src.optimizers.cluster_markowitz import (
    _markowitz_inter_ew_intra,
    _ew_inter_markowitz_intra,
    _markowitz_inter_markowitz_intra,
)

RANDOM_SEED_DEFAULT = 42


# ── KMeans adaptive helpers ───────────────────────────────────────────────────

def _kmeans_adaptive_labels_and_sil(
    window_returns: pd.DataFrame,
    random_state: int,
) -> tuple[dict[int, np.ndarray], dict[int, float], int]:
    """
    Compute KMeans labels + silhouette (on correlation distance) for every k
    in K_VALUES_ADAPTIVE, then return:
      labels_per_k : {k: labels}
      sil_scores   : {k: score}
      selected_k   : best k via delta rule
    """
    asset_matrix = raw_return_representation(window_returns)
    corr_dist    = correlation_distance_matrix(asset_matrix)

    labels_per_k: dict[int, np.ndarray] = {}
    sil_scores:   dict[int, float]      = {}
    for k in K_VALUES_ADAPTIVE:
        lbl = kmeans_labels(asset_matrix, n_clusters=k, random_state=random_state)
        labels_per_k[k] = lbl
        sil_scores[k]   = float(silhouette_score(corr_dist, lbl, metric="precomputed"))

    selected_k = _select_adaptive_k(sil_scores)
    return labels_per_k, sil_scores, selected_k


# ── KMedoids adaptive helpers ─────────────────────────────────────────────────

def _kmedoids_adaptive_labels_and_sil(
    window_returns: pd.DataFrame,
    random_state: int,
    dtw_n_jobs: int,
) -> tuple[dict[int, np.ndarray], dict[int, float], int]:
    """
    Compute KMedoids labels + silhouette (on DTW distance) for every k
    in K_VALUES_ADAPTIVE.  DTW matrix is loaded from cache when available.
    """
    asset_matrix = raw_return_representation(window_returns)
    cache_key    = _make_dtw_cache_key_for_window(window_returns)
    dtw_dist     = dtw_distance_matrix(
        asset_matrix,
        n_jobs=dtw_n_jobs,
        cache_dir=_DTW_CACHE_DIR,
        cache_key=cache_key,
    )

    labels_per_k: dict[int, np.ndarray] = {}
    sil_scores:   dict[int, float]      = {}
    for k in K_VALUES_ADAPTIVE:
        lbl = kmedoids_labels_from_distance(dtw_dist, n_clusters=k, random_state=random_state)
        labels_per_k[k] = lbl
        sil_scores[k]   = float(silhouette_score(dtw_dist, lbl, metric="precomputed"))

    selected_k = _select_adaptive_k(sil_scores)
    return labels_per_k, sil_scores, selected_k


# ── Public functions ──────────────────────────────────────────────────────────

def get_cluster_kmeans_adaptive_markowitz_inter_ew_intra(
    window_returns: pd.DataFrame,
    random_state: int = RANDOM_SEED_DEFAULT,
) -> tuple[pd.Series, dict]:
    labels_per_k, sil_scores, k = _kmeans_adaptive_labels_and_sil(window_returns, random_state)
    weights = _markowitz_inter_ew_intra(window_returns, labels_per_k[k])
    return weights, {"selected_k": k, "silhouette_score": sil_scores[k]}


def get_cluster_kmeans_adaptive_ew_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    random_state: int = RANDOM_SEED_DEFAULT,
) -> tuple[pd.Series, dict]:
    labels_per_k, sil_scores, k = _kmeans_adaptive_labels_and_sil(window_returns, random_state)
    weights = _ew_inter_markowitz_intra(window_returns, labels_per_k[k])
    return weights, {"selected_k": k, "silhouette_score": sil_scores[k]}


def get_cluster_kmeans_adaptive_markowitz_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    random_state: int = RANDOM_SEED_DEFAULT,
) -> tuple[pd.Series, dict]:
    labels_per_k, sil_scores, k = _kmeans_adaptive_labels_and_sil(window_returns, random_state)
    weights = _markowitz_inter_markowitz_intra(window_returns, labels_per_k[k])
    return weights, {"selected_k": k, "silhouette_score": sil_scores[k]}


def get_cluster_kmedoids_dtw_adaptive_markowitz_inter_ew_intra(
    window_returns: pd.DataFrame,
    random_state: int = RANDOM_SEED_DEFAULT,
    dtw_n_jobs: int = -1,
) -> tuple[pd.Series, dict]:
    labels_per_k, sil_scores, k = _kmedoids_adaptive_labels_and_sil(
        window_returns, random_state, dtw_n_jobs
    )
    weights = _markowitz_inter_ew_intra(window_returns, labels_per_k[k])
    return weights, {"selected_k": k, "silhouette_score": sil_scores[k]}


def get_cluster_kmedoids_dtw_adaptive_ew_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    random_state: int = RANDOM_SEED_DEFAULT,
    dtw_n_jobs: int = -1,
) -> tuple[pd.Series, dict]:
    labels_per_k, sil_scores, k = _kmedoids_adaptive_labels_and_sil(
        window_returns, random_state, dtw_n_jobs
    )
    weights = _ew_inter_markowitz_intra(window_returns, labels_per_k[k])
    return weights, {"selected_k": k, "silhouette_score": sil_scores[k]}


def get_cluster_kmedoids_dtw_adaptive_markowitz_inter_markowitz_intra(
    window_returns: pd.DataFrame,
    random_state: int = RANDOM_SEED_DEFAULT,
    dtw_n_jobs: int = -1,
) -> tuple[pd.Series, dict]:
    labels_per_k, sil_scores, k = _kmedoids_adaptive_labels_and_sil(
        window_returns, random_state, dtw_n_jobs
    )
    weights = _markowitz_inter_markowitz_intra(window_returns, labels_per_k[k])
    return weights, {"selected_k": k, "silhouette_score": sil_scores[k]}
