"""
MDS clustering visualisation and silhouette comparison plots.

For a sample window centred around 2016-06-30 this script:

1. Computes cluster labels for each (method, k) combination.
2. Projects assets to 2D via MDS on the relevant distance matrix
   (correlation for KMeans, DTW for K-Medoids) and saves one PNG per
   (method, k) as mds_kmeans_k5.png / mds_kmedoids_dtw_k5.png.
3. Produces silhouette-vs-k comparison plots for both methods.
4. Produces a large comparison grid (k=1..8 × KMeans / KMedoids)
   as mds_comparison_grid.png.

Run *after* scripts/run_backtest.py so that the DTW cache exists.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import silhouette_score, pairwise_distances

from src.data.load import load_returns_csv
from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import (
    dtw_distance_matrix,
    correlation_distance_matrix,
)
from src.clustering.representations import raw_return_representation
from src.clustering.pipeline import _make_dtw_cache_key
from src.paths import (
    get_returns_path, get_clustering_visualization_dir,
    get_dtw_cache_dir, UNIVERSE_CHOICES,
)

# ── Configuration ────────────────────────────────────────────────────────────
TARGET_DATE = "2008-04-01"   # approximate window end; nearest available used
ESTIMATION_WINDOW = 756      # trading days (~3 years), matches run_backtest.py
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]   # k values to plot and compare
RANDOM_STATE = 42
MDS_MAX_ITER = 300
MDS_N_INIT = 4
# ─────────────────────────────────────────────────────────────────────────────


def _get_sample_window(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the first ESTIMATION_WINDOW trading days from the start of the data.

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily return matrix with a DatetimeIndex, shape
        ``(n_days, n_assets)``.

    Returns
    -------
    pd.DataFrame
        Slice containing exactly ``ESTIMATION_WINDOW`` rows starting from
        the first available date.
    """
    if len(returns) < ESTIMATION_WINDOW:
        raise ValueError(
            f"Not enough data for estimation window "
            f"({len(returns)} < {ESTIMATION_WINDOW})"
        )
    window = returns.iloc[:ESTIMATION_WINDOW]
    print(
        f"Sample window: {window.index[0].date()} → {window.index[-1].date()} "
        f"({len(window)} days, {window.shape[1]} assets)"
    )
    return window


def _mds_embedding(dist_matrix: np.ndarray) -> np.ndarray:
    """2-D MDS embedding from a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Square symmetric distance matrix of shape ``(n_assets, n_assets)``.

    Returns
    -------
    np.ndarray
        2-D coordinates array of shape ``(n_assets, 2)``.
    """
    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__) >= Version("1.8"):
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
    """Return n distinct colours from tab20 / tab10.

    Parameters
    ----------
    n : int
        Number of distinct colours required.

    Returns
    -------
    list
        List of ``n`` RGBA colour tuples evenly sampled from the
        ``tab20`` colormap (or ``tab10`` when ``n <= 10``).
    """
    base = matplotlib.colormaps["tab20" if n > 10 else "tab10"]
    return [base(i / max(n - 1, 1)) for i in range(n)]


def _axis_limits_with_padding(
    embedding: np.ndarray,
    pad_fraction: float = 0.10,
) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) with pad_fraction padding on each side.

    Parameters
    ----------
    embedding : np.ndarray
        2-D coordinate array of shape ``(n_assets, 2)``.
    pad_fraction : float, optional
        Fraction of the data range added as padding on each side.

    Returns
    -------
    tuple[float, float, float, float]
        ``(xmin, xmax, ymin, ymax)`` axis limits with padding applied.
    """
    xmin, xmax = float(embedding[:, 0].min()), float(embedding[:, 0].max())
    ymin, ymax = float(embedding[:, 1].min()), float(embedding[:, 1].max())
    x_pad = (xmax - xmin) * pad_fraction
    y_pad = (ymax - ymin) * pad_fraction
    return xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad


def _scatter_clusters(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Draw one scatter series per cluster on *ax*.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes on which to draw the scatter plot.
    embedding : np.ndarray
        2-D coordinate array of shape ``(n_assets, 2)``.
    labels : np.ndarray
        Integer cluster label for each asset, shape ``(n_assets,)``.
    """
    unique = sorted(np.unique(labels))
    colours = _discrete_cmap(len(unique))
    colour_map = {c: colours[i] for i, c in enumerate(unique)}
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


# ── Change 1: single plot with axes fitted to data (10% padding) ──────────────

def plot_mds(
    embedding: np.ndarray,
    labels: np.ndarray,
    k: int,
    method_label: str,
    output_path: Path,
) -> None:
    """Save a single MDS scatter plot for one method and k.

    Parameters
    ----------
    embedding : np.ndarray
        2-D MDS coordinates of shape ``(n_assets, 2)``.
    labels : np.ndarray
        Integer cluster assignment for each asset, shape ``(n_assets,)``.
    k : int
        Number of clusters (used only for the plot title).
    method_label : str
        Human-readable method description for the plot title.
    output_path : Path
        Full path (including filename) where the PNG is saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_clusters(ax, embedding, labels)

    # Axis limits derived from the embedding — no more corner-crowding
    xlo, xhi, ylo, yhi = _axis_limits_with_padding(embedding)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    ax.set_title(f"MDS — {method_label}  k={k}")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
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
    """Save a silhouette-score line plot for one clustering method.

    Parameters
    ----------
    k_values : list[int]
        Ordered list of k values on the x-axis.
    scores : list[float]
        Corresponding silhouette scores; NaN entries are excluded from
        the plot.
    method_label : str
        Human-readable method name used in the plot title.
    output_path : Path
        Full path where the PNG is saved.
    """
    valid = [(k, s) for k, s in zip(k_values, scores) if not np.isnan(s)]
    if not valid:
        print(f"  No valid silhouette scores for {method_label}, skipping plot.")
        return
    ks, ss = zip(*valid)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, ss, marker="o", linewidth=2, color="steelblue")
    ax.set_title(f"Silhouette Score vs k — {method_label}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(list(ks))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ── Change 2: large comparison grid ──────────────────────────────────────────

def _local_axis_limits(
    embedding: np.ndarray,
    pad_fraction: float = 0.20,
    pct_lo: float = 5.0,
    pct_hi: float = 95.0,
) -> tuple[float, float, float, float]:
    """Per-subplot limits: percentile clip + pad_fraction padding.

    Parameters
    ----------
    embedding : np.ndarray
        2-D coordinate array of shape ``(n_assets, 2)``.
    pad_fraction : float, optional
        Fraction of the clipped range added as padding on each side.
    pct_lo : float, optional
        Lower percentile used to clip outliers before computing limits.
    pct_hi : float, optional
        Upper percentile used to clip outliers before computing limits.

    Returns
    -------
    tuple[float, float, float, float]
        ``(xmin, xmax, ymin, ymax)`` after percentile clipping and
        padding.
    """
    xlo = float(np.percentile(embedding[:, 0], pct_lo))
    xhi = float(np.percentile(embedding[:, 0], pct_hi))
    ylo = float(np.percentile(embedding[:, 1], pct_lo))
    yhi = float(np.percentile(embedding[:, 1], pct_hi))
    x_pad = (xhi - xlo) * pad_fraction
    y_pad = (yhi - ylo) * pad_fraction
    return xlo - x_pad, xhi + x_pad, ylo - y_pad, yhi + y_pad


def plot_mds_comparison_grid(
    k_values: list[int],
    mds_eucl: np.ndarray,
    mds_corr: np.ndarray,
    mds_dtw: np.ndarray,
    km_labels_by_k: dict[int, np.ndarray],
    kmed_labels_by_k: dict[int, np.ndarray],
    plots_dir: Path,
) -> None:
    """k × 4 comparison grid:
      col 0 — KMeans Euclidean MDS   (global limits)
      col 1 — KMeans Correlation MDS (global limits)
      col 2 — K-Medoids DTW          (global limits)
      col 3 — K-Medoids DTW          (local limits: p5–p95 + 20 % padding)
    Global limits are derived from the combined extent of cols 0, 1, and 2.
    """
    n_rows = len(k_values)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(22, n_rows * 4),
    )

    # ── Global axis limits: cols 0, 1, 2 (euclidean, correlation, DTW) ───────
    all_x = np.concatenate([mds_eucl[:, 0], mds_corr[:, 0], mds_dtw[:, 0]])
    all_y = np.concatenate([mds_eucl[:, 1], mds_corr[:, 1], mds_dtw[:, 1]])
    gx_lo, gx_hi, gy_lo, gy_hi = _axis_limits_with_padding(
        np.column_stack([all_x, all_y])
    )

    # ── Column headers ────────────────────────────────────────────────────────
    col_titles = [
        "KMeans — Euclidean MDS",
        "KMeans — Correlation MDS",
        "K-Medoids DTW — global scale",
        "K-Medoids DTW — local scale",
    ]
    for col_idx, title in enumerate(col_titles):
        x_pos = (col_idx + 0.5) / 4
        fig.text(
            x_pos, 1.0,
            title,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            transform=fig.transFigure,
        )

    # ── Subplots ──────────────────────────────────────────────────────────────
    def _style(ax: plt.Axes, title: str) -> None:
        """Apply shared subplot titles, labels, ticks, and legend style.

        Parameters
        ----------
        ax : plt.Axes
            Subplot axes to configure.
        title : str
            Title string set on the axes.
        """
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("MDS dim 1", fontsize=8)
        ax.set_ylabel("MDS dim 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=6, markerscale=1.2,
                  framealpha=0.7, borderpad=0.4)

    for row_idx, k in enumerate(k_values):
        ax_eucl, ax_corr, ax_glob, ax_loc = axes[row_idx]

        # Col 0 — KMeans, Euclidean MDS, global limits
        _scatter_clusters(ax_eucl, mds_eucl, km_labels_by_k[k])
        ax_eucl.set_xlim(gx_lo, gx_hi)
        ax_eucl.set_ylim(gy_lo, gy_hi)
        _style(ax_eucl, f"KMeans  k={k}")

        # Col 1 — KMeans, Correlation MDS, global limits
        _scatter_clusters(ax_corr, mds_corr, km_labels_by_k[k])
        ax_corr.set_xlim(gx_lo, gx_hi)
        ax_corr.set_ylim(gy_lo, gy_hi)
        _style(ax_corr, f"KMeans  k={k}")

        # Col 2 — K-Medoids, global limits
        _scatter_clusters(ax_glob, mds_dtw, kmed_labels_by_k[k])
        ax_glob.set_xlim(gx_lo, gx_hi)
        ax_glob.set_ylim(gy_lo, gy_hi)
        _style(ax_glob, f"K-Medoids  k={k}")

        # Col 3 — K-Medoids, local limits (p5–p95 + 20 % padding)
        _scatter_clusters(ax_loc, mds_dtw, kmed_labels_by_k[k])
        lx_lo, lx_hi, ly_lo, ly_hi = _local_axis_limits(mds_dtw)
        ax_loc.set_xlim(lx_lo, lx_hi)
        ax_loc.set_ylim(ly_lo, ly_hi)
        _style(ax_loc, f"K-Medoids  k={k}  (local)")

    fig.suptitle(
        "MDS Clustering Comparison",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    output_path = plots_dir / "mds_comparison_grid.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate clustering MDS and silhouette plots for one universe."""
    parser = argparse.ArgumentParser(description="Generate clustering visualisation plots.")
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────────
    returns_path = get_returns_path(args.universe)
    plots_dir = get_clustering_visualization_dir(args.universe) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dtw_cache = get_dtw_cache_dir(args.universe)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading returns...")
    returns = load_returns_csv(returns_path)
    window_returns = _get_sample_window(returns)
    asset_matrix = raw_return_representation(window_returns)   # (n_assets, T)
    n_assets = asset_matrix.shape[0]

    # Clip K_VALUES so we never request k > n_assets
    k_values_plot = [k for k in K_VALUES if k <= n_assets]
    window_end_str = str(window_returns.index[-1].date())

    # ── Distance matrices ─────────────────────────────────────────────────────
    print("\nComputing Euclidean distance matrix (KMeans col 1 visualisation)...")
    eucl_dist = pairwise_distances(asset_matrix, metric="euclidean")

    print("Computing correlation distance matrix (KMeans col 2 visualisation)...")
    corr_dist = correlation_distance_matrix(asset_matrix)

    print("Computing / loading DTW distance matrix (K-Medoids)...")
    cache_key = _make_dtw_cache_key(window_returns)
    dtw_dist = dtw_distance_matrix(
        asset_matrix,
        n_jobs=-1,
        cache_dir=dtw_cache,
        cache_key=cache_key,
    )

    # ── MDS embeddings (computed once per distance metric) ────────────────────
    print("\nComputing MDS embedding on Euclidean distance...")
    mds_eucl = _mds_embedding(eucl_dist)

    print("Computing MDS embedding on correlation distance...")
    mds_corr = _mds_embedding(corr_dist)

    print("Computing MDS embedding on DTW distance...")
    mds_dtw = _mds_embedding(dtw_dist)

    # ── Per-k plots, silhouette scores, and label collection ─────────────────
    sil_kmeans: list[float] = []
    sil_kmedoids: list[float] = []
    km_labels_by_k: dict[int, np.ndarray] = {}
    kmed_labels_by_k: dict[int, np.ndarray] = {}

    for k in k_values_plot:
        print(f"\n── k={k} ──────────────────────────────────")

        # KMeans
        km_labels = kmeans_labels(asset_matrix, n_clusters=k, random_state=RANDOM_STATE)
        km_labels_by_k[k] = km_labels
        plot_mds(
            mds_corr,
            km_labels,
            k=k,
            method_label="KMeans (correlation distance)",
            output_path=plots_dir / f"mds_kmeans_k{k}.png",
        )
        if k == 1:
            sil = 0.0
        elif 2 <= len(np.unique(km_labels)) < n_assets:
            sil = float(silhouette_score(corr_dist, km_labels, metric="precomputed"))
        else:
            sil = float("nan")
        sil_kmeans.append(sil)
        msg = (f"  KMeans   silhouette: {sil:.4f}"
               if not np.isnan(sil) else "  KMeans   silhouette: N/A")
        print(msg)

        # K-Medoids DTW
        if k == 1:
            kmed_labels = np.zeros(dtw_dist.shape[0], dtype=int)
        else:
            kmed_labels = kmedoids_labels_from_distance(
                dtw_dist.copy(), n_clusters=k, random_state=RANDOM_STATE
            )
        kmed_labels_by_k[k] = kmed_labels
        plot_mds(
            mds_dtw,
            kmed_labels,
            k=k,
            method_label="K-Medoids (DTW distance)",
            output_path=plots_dir / f"mds_kmedoids_dtw_k{k}.png",
        )
        if k == 1:
            sil_kmed = 0.0
        elif 2 <= len(np.unique(kmed_labels)) < n_assets:
            sil_kmed = float(silhouette_score(dtw_dist, kmed_labels, metric="precomputed"))
        else:
            sil_kmed = float("nan")
        sil_kmedoids.append(sil_kmed)
        msg = (f"  K-Medoids silhouette: {sil_kmed:.4f}"
               if not np.isnan(sil_kmed) else "  K-Medoids silhouette: N/A")
        print(msg)

    # ── Silhouette comparison plots ───────────────────────────────────────────
    print("\nGenerating silhouette comparison plots...")
    plot_silhouette_comparison(
        k_values_plot,
        sil_kmeans,
        method_label="KMeans",
        output_path=plots_dir / "silhouette_comparison_kmeans.png",
    )
    plot_silhouette_comparison(
        k_values_plot,
        sil_kmedoids,
        method_label="K-Medoids DTW",
        output_path=plots_dir / "silhouette_comparison_kmedoids.png",
    )

    # ── Combined silhouette plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values_plot, sil_kmeans, marker="o", linewidth=2,
            label="KMeans (euclidean)", color="steelblue")
    ax.plot(k_values_plot, sil_kmedoids, marker="s", linewidth=2,
            label="K-Medoids DTW (precomputed)", color="darkorange")
    ax.set_title(f"Silhouette Score Comparison\nwindow ending {window_end_str}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(k_values_plot)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    combined_path = plots_dir / "silhouette_comparison_all.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {combined_path.name}")

    # ── Large comparison grid (k=1..8, KMeans vs K-Medoids) ──────────────────
    print("\nGenerating MDS comparison grid...")
    grid_k_values = [k for k in range(1, 9) if k <= n_assets]
    plot_mds_comparison_grid(
        k_values=grid_k_values,
        mds_eucl=mds_eucl,
        mds_corr=mds_corr,
        mds_dtw=mds_dtw,
        km_labels_by_k=km_labels_by_k,
        kmed_labels_by_k=kmed_labels_by_k,
        plots_dir=plots_dir,
    )

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
