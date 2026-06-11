"""
src/backtest/permutation_backtest.py

Permutation (random-baseline) strategy factory for ablation studies.

Usage
-----
    from src.backtest.permutation_backtest import make_permuted_strategy
    strategy_fn = make_permuted_strategy(method="kmeans", k=5, seed=0)
    # strategy_fn(window_returns) → pd.Series of weights

How it works
------------
For each rebalance call the returned function:
  1. Computes the *real* cluster labels (identical to the live strategy).
  2. Permutes the label assignments with a reproducible RNG — cluster sizes
     are preserved, but which assets belong to each cluster is random.
  3. Applies _ew_inter_markowitz_intra with the permuted labels to produce
     portfolio weights.

The RNG state advances across rebalance dates (one rng.permutation call per
date), so every date gets a different permutation while the full sequence
remains reproducible given the same seed.

DTW cache: kmedoids permutation calls share the same cache keys as the live
kmedoids backtest (same window bounds + tickers → same SHA-256 hash), so
the distance matrix is never recomputed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.optimizers.cluster_markowitz import (
    _DTW_CACHE_DIR,
    _ew_inter_markowitz_intra,
    _get_labels_kmeans,
    _get_labels_kmedoids,
    _make_dtw_cache_key,
)
from src.clustering.distances import dtw_distance_matrix
from src.clustering.representations import raw_return_representation
from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance


def make_permuted_strategy(
    method: str,
    k: int,
    seed: int,
    random_state: int = 42,
    dtw_n_jobs: int = -1,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Return a callable strategy that permutes cluster label assignments.

    Parameters
    ----------
    method : str
        "kmeans" or "kmedoids".
    k : int
        Number of clusters.
    seed : int
        RNG seed for the permutation sequence.  Same seed → same sequence
        of label shuffles across all rebalance dates.
    random_state : int
        Random state forwarded to the underlying clustering algorithm
        (same default as live strategies).
    dtw_n_jobs : int
        Parallelism for DTW distance computation (kmedoids only).

    Returns
    -------
    Callable[[pd.DataFrame], pd.Series]
        Strategy function that accepts window returns and returns permuted weights.
    """
    if method not in ("kmeans", "kmedoids"):
        raise ValueError(f"method must be 'kmeans' or 'kmedoids', got {method!r}")

    rng = np.random.default_rng(seed)

    def _strategy(window_returns: pd.DataFrame) -> pd.Series:
        """Compute permuted-label EW-inter/Markowitz-intra weights for one window.

        Clusters the assets in ``window_returns`` with the configured
        method, randomly shuffles the resulting cluster labels, and then
        applies equal-weight inter-cluster / Markowitz intra-cluster
        allocation on the shuffled assignment.

        Parameters
        ----------
        window_returns : pd.DataFrame
            Daily return matrix for the estimation window, shape
            ``(n_days, n_assets)``.

        Returns
        -------
        pd.Series
            Portfolio weights indexed by asset ticker, non-negative and
            summing to 1.
        """
        if method == "kmeans":
            labels = _get_labels_kmeans(window_returns, global_k=k, random_state=random_state)
        else:
            labels = _get_labels_kmedoids(
                window_returns,
                global_k=k,
                random_state=random_state,
                dtw_n_jobs=dtw_n_jobs,
            )

        shuffled_labels = rng.permutation(labels)
        return _ew_inter_markowitz_intra(window_returns, shuffled_labels)

    return _strategy
