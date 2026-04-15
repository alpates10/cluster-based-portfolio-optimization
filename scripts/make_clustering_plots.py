"""
MDS clustering visualisation and silhouette comparison plots.

For a sample window centred around 2016-06-30 this script:

1. Computes cluster labels for each (method, k) combination.
2. Projects assets to 2D via MDS on the relevant distance matrix
   (correlation for KMeans, DTW for K-Medoids) and saves one PNG per
   (method, k) as mds_kmeans_k5.png / mds_kmedoids_dtw_k5.png.
3. Produces silhouette-vs-k comparison plots for both methods.

Run *after* scripts/run_backtest.py so that the DTW cache exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

from src.data.load import load_returns_csv
from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import (
    dtw_distance_matrix,
    correlation_distance_matrix,
)
from src.clustering.representations import raw_return_representation
from src.clustering.pipeline import _DTW_CACHE_DIR, _make_dtw_cache_key

# ── Configuration ────────────────────────────────────────────────────────────
TARGET_DATE = "2019-12-31"   # approximate window end; nearest available used
ESTIMATION_WINDOW = 756      # trading days (~3 years), matches run_backtest.py
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
RANDOM_STATE = 42
MDS_MAX_ITER = 300
MDS_N_INIT = 4
# ─────────────────────────────────────────────────────────────────────────────


def _get_sample_window(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the ESTIMATION_WINDOW trading days ending at or before TARGET_DATE."""
    target = pd.Timestamp(TARGET_DATE)
    available = returns.index[returns.index <= target]
    if len(available) < ESTIMATION_WINDOW:
        raise ValueError(
            f"Not enough data before {TARGET_DATE} for estimation window "
            f"({len(available)} < {ESTIMATION_WINDOW})"
        )
    end_idx = returns.index.get_loc(available[-1])
    start_idx = end_idx - ESTIMATION_WINDOW + 1
    window = returns.iloc[start_idx : end_idx + 1]
    print(
        f"Sample window: {window.index[0].date()} → {window.index[-1].date()} "
        f"({len(window)} days, {window.shape[1]} assets)"
    )
    return window


def _mds_embedding(dist_matrix: np.ndarray) -> np.ndarray:
    """2-D MDS embedding from a precomputed distance matrix."""
    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__) >= Version("1.8"):
        # sklearn ≥ 1.8: metric_mds + metric='precomputed'
        mds = MDS(
            n_components=2,
            metric_mds=True,
            metric="precomputed",
            normalized_stress=False,
            init="random",
            n_init=MDS_N_INIT,
            max_iter=MDS_MAX_ITER,
            random_state=RANDOM_STATE,
        )
    else:
        # sklearn < 1.8: eski API
        mds = MDS(
            n_components=2,
            metric=True,
            dissimilarity="precomputed",
            normalized_stress="auto",
            init="random",
            n_init=MDS_N_INIT,
            max_iter=MDS_MAX_ITER,
            random_state=RANDOM_STATE,
        )
    return mds.fit_transform(dist_matrix)


def _discrete_cmap(n: int):
    """Return n distinct colours from tab20 / tab10."""
    base = matplotlib.colormaps["tab20" if n > 10 else "tab10"]
    return [base(i / max(n - 1, 1)) for i in range(n)]


def plot_mds(
    embedding: np.ndarray,
    labels: np.ndarray,
    k: int,
    method_label: str,
    output_path: Path,
) -> None:
    unique = sorted(np.unique(labels))
    colours = _discrete_cmap(len(unique))
    colour_map = {c: colours[i] for i, c in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in unique:
        mask = labels == cid
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colour_map[cid]],
            label=f"Cluster {cid}",
            s=30,
            alpha=0.8,
            edgecolors="none",
        )

    ax.set_title(f"MDS — {method_label}  k={k}\nwindow ending {TARGET_DATE}")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_silhouette_comparison(
    k_values: list[int],
    scores: list[float],
    method_label: str,
    output_path: Path,
) -> None:
    valid = [(k, s) for k, s in zip(k_values, scores) if not np.isnan(s)]
    if not valid:
        print(f"  No valid silhouette scores for {method_label}, skipping plot.")
        return
    ks, ss = zip(*valid)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, ss, marker="o", linewidth=2, color="steelblue")

    ax.set_title(f"Silhouette Score vs k — {method_label}\nwindow ending {TARGET_DATE}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(list(ks))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main() -> None:
    # ── Paths ────────────────────────────────────────────────────────────────
    returns_path = PROJECT_ROOT / "data" / "processed" / "returns_final.csv"
    plots_dir = PROJECT_ROOT / "data" / "processed" / "clustering_analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading returns...")
    returns = load_returns_csv(returns_path)
    window_returns = _get_sample_window(returns)
    asset_matrix = raw_return_representation(window_returns)   # (n_assets, T)
    n_assets = asset_matrix.shape[0]

    # ── Distance matrices ─────────────────────────────────────────────────────
    print("\nComputing correlation distance matrix (KMeans visualisation)...")
    corr_dist = correlation_distance_matrix(asset_matrix)

    print("Computing / loading DTW distance matrix (K-Medoids)...")
    cache_key = _make_dtw_cache_key(window_returns)
    dtw_dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=-1,
        cache_dir=_DTW_CACHE_DIR,
        cache_key=cache_key,
    )

    # ── MDS embeddings (computed once per distance metric) ────────────────────
    print("\nComputing MDS embedding on correlation distance...")
    mds_corr = _mds_embedding(corr_dist)

    print("Computing MDS embedding on DTW distance...")
    mds_dtw = _mds_embedding(dtw_dist)

    # ── Per-k plots and silhouette scores ─────────────────────────────────────
    sil_kmeans: list[float] = []
    sil_kmedoids: list[float] = []

    for k in K_VALUES:
        print(f"\n── k={k} ──────────────────────────────────")

        # KMeans
        km_labels = kmeans_labels(asset_matrix, n_clusters=k, random_state=RANDOM_STATE)
        plot_mds(
            mds_corr,
            km_labels,
            k=k,
            method_label="KMeans (correlation distance)",
            output_path=plots_dir / f"mds_kmeans_k{k}.png",
        )
        if k == 1:
            sil = 0.0  # tek cluster: silhouette tanımsız, 0 olarak gösterilir
        elif 2 <= len(np.unique(km_labels)) < n_assets:
            sil = float(silhouette_score(corr_dist, km_labels, metric="precomputed"))
        else:
            sil = float("nan")
        sil_kmeans.append(sil)
        print(f"  KMeans   silhouette: {sil:.4f}" if not np.isnan(sil) else "  KMeans   silhouette: N/A")

        # K-Medoids DTW
        if k == 1:
            kmed_labels = np.zeros(dtw_dist.shape[0], dtype=int)
        else:
            kmed_labels = kmedoids_labels_from_distance(
                dtw_dist.copy(), n_clusters=k, random_state=RANDOM_STATE
            )
        plot_mds(
            mds_dtw,
            kmed_labels,
            k=k,
            method_label="K-Medoids (DTW distance)",
            output_path=plots_dir / f"mds_kmedoids_dtw_k{k}.png",
        )
        if k == 1:
            sil_kmed = 0.0  # tek cluster: silhouette tanımsız, 0 olarak gösterilir
        elif 2 <= len(np.unique(kmed_labels)) < n_assets:
            sil_kmed = float(silhouette_score(dtw_dist, kmed_labels, metric="precomputed"))
        else:
            sil_kmed = float("nan")
        sil_kmedoids.append(sil_kmed)
        print(f"  K-Medoids silhouette: {sil_kmed:.4f}" if not np.isnan(sil_kmed) else "  K-Medoids silhouette: N/A")

    # ── Silhouette comparison plots ───────────────────────────────────────────
    print("\nGenerating silhouette comparison plots...")
    plot_silhouette_comparison(
        K_VALUES,
        sil_kmeans,
        method_label="KMeans",
        output_path=plots_dir / "silhouette_comparison_kmeans.png",
    )
    plot_silhouette_comparison(
        K_VALUES,
        sil_kmedoids,
        method_label="K-Medoids DTW",
        output_path=plots_dir / "silhouette_comparison_kmedoids.png",
    )

    # ── Combined silhouette plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_VALUES, sil_kmeans, marker="o", linewidth=2,
            label="KMeans (euclidean)", color="steelblue")
    ax.plot(K_VALUES, sil_kmedoids, marker="s", linewidth=2,
            label="K-Medoids DTW (precomputed)", color="darkorange")

    ax.set_title(f"Silhouette Score Comparison\nwindow ending {TARGET_DATE}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(K_VALUES)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    combined_path = plots_dir / "silhouette_comparison_all.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {combined_path.name}")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
