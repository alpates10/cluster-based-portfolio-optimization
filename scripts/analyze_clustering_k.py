from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_portfolio_robustness")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.evaluation import evaluate_clustering_config, recommend_global_k
from src.clustering.pipeline import _make_dtw_cache_key
from src.clustering.representations import raw_return_representation
from src.clustering.selection import DEFAULT_K_CANDIDATES, select_non_overlapping_windows
from src.data.load import load_returns_csv
from src.paths import get_returns_path, get_clustering_dir, get_dtw_cache_dir, UNIVERSE_CHOICES


def _plot_metric_lines(results_df: pd.DataFrame, out_dir: Path, method: str, distance: str, metric: str) -> None:
    subset = results_df[
        (results_df["method"] == method)
        & (results_df["distance"] == distance)
        & (results_df["status"] == "ok")
    ].copy()
    if subset.empty:
        return

    plt.figure(figsize=(9, 5))
    for window_id, grp in subset.groupby("window_id"):
        g = grp.sort_values("k")
        plt.plot(g["k"], g[metric], marker="o", alpha=0.6, label=str(window_id))

    avg = subset.groupby("k", as_index=False)[metric].mean().sort_values("k")
    plt.plot(avg["k"], avg[metric], marker="o", linewidth=2.5, color="black", label="average")

    k_values = sorted(subset["k"].unique())
    plt.xticks(k_values)
    plt.title(f"{method} + {distance} | {metric}")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend(title="window")
    plt.tight_layout()

    out_path = out_dir / f"{method}_{distance}_{metric}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_heatmap_silhouette(results_df: pd.DataFrame, out_dir: Path, method: str, distance: str) -> None:
    subset = results_df[
        (results_df["method"] == method)
        & (results_df["distance"] == distance)
        & (results_df["status"] == "ok")
    ].copy()
    if subset.empty:
        return

    pivot = subset.pivot_table(
        index="k",
        columns="window_id",
        values="silhouette_score",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return

    plt.figure(figsize=(7, 5))
    arr = pivot.to_numpy(dtype=float)
    im = plt.imshow(arr, aspect="auto", cmap="YlGnBu", origin="lower")
    plt.colorbar(im, label="silhouette_score")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=0)
    plt.yticks(range(len(pivot.index)), [int(x) for x in pivot.index])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    plt.title(f"{method} + {distance} | silhouette heatmap")
    plt.xlabel("window")
    plt.ylabel("k")
    plt.tight_layout()
    out_path = out_dir / f"{method}_{distance}_silhouette_heatmap.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze optimal k for clustering.")
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    args = parser.parse_args()

    returns_path = get_returns_path(args.universe)
    out_root = get_clustering_dir(args.universe)
    plots_dir = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dtw_cache = get_dtw_cache_dir(args.universe)

    window_length = 756
    n_windows = 5
    k_candidates = [1] + DEFAULT_K_CANDIDATES

    # Core configs; pipeline is modular and can be extended with more combos.
    configs = [
        {"method": "kmeans", "distance": "euclidean"},
        {"method": "kmedoids", "distance": "dtw"},
    ]

    returns = load_returns_csv(returns_path)
    windows_df = select_non_overlapping_windows(
        returns_index=returns.index,
        window_length=window_length,
        n_windows=n_windows,
    )
    windows_df.to_csv(out_root / "selected_windows.csv", index=False)
    print("Selected windows:")
    print(windows_df[["window_id", "start_date", "end_date", "n_obs"]].to_string(index=False))

    tasks = []
    for _, w in windows_df.iterrows():
        start_iloc = int(w["start_iloc"])
        end_iloc = int(w["end_iloc"])
        window_returns = returns.iloc[start_iloc : end_iloc + 1]
        asset_matrix = raw_return_representation(window_returns)
        cache_key = _make_dtw_cache_key(window_returns)
        for cfg in configs:
            for k in k_candidates:
                tasks.append((w, cfg, k, asset_matrix, cache_key))

    has_tslearn = importlib.util.find_spec("tslearn") is not None

    rows: list[dict] = []
    for w, cfg, k, asset_matrix, cache_key in tqdm(tasks, desc="Evaluating clustering configs", unit="cfg"):
        if cfg["method"] == "kmedoids" and cfg["distance"] == "dtw" and not has_tslearn:
            rows.append(
                {
                    "window_id": w["window_id"],
                    "window_start": pd.Timestamp(w["start_date"]).date().isoformat(),
                    "window_end": pd.Timestamp(w["end_date"]).date().isoformat(),
                    "method": cfg["method"],
                    "distance": cfg["distance"],
                    "k": int(k),
                    "silhouette_score": np.nan,
                    "davies_bouldin": np.nan,
                    "cluster_balance": np.nan,
                    "n_clusters_found": np.nan,
                    "status": "skip",
                    "notes": "tslearn is not installed; skipping DTW-based evaluation",
                }
            )
            continue

        eval_out = evaluate_clustering_config(
            asset_return_matrix=asset_matrix,
            method=cfg["method"],
            distance=cfg["distance"],
            k=int(k),
            random_state=42,
            dtw_n_jobs=-1,
            dtw_cache_dir=dtw_cache,
            dtw_cache_key=cache_key,
        )
        row = {
            "window_id": w["window_id"],
            "window_start": pd.Timestamp(w["start_date"]).date().isoformat(),
            "window_end": pd.Timestamp(w["end_date"]).date().isoformat(),
            "method": cfg["method"],
            "distance": cfg["distance"],
            "k": int(k),
            "silhouette_score": eval_out.get("silhouette_score"),
            "davies_bouldin": eval_out.get("davies_bouldin"),
            "cluster_balance": eval_out.get("cluster_balance"),
            "n_clusters_found": eval_out.get("n_clusters_found"),
            "status": eval_out.get("status", "error"),
            "notes": eval_out.get("notes", ""),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        ["method", "distance", "window_id", "k"]
    ).reset_index(drop=True)
    results_path = out_root / "k_selection_results.csv"
    results_df.to_csv(results_path, index=False)

    recommended_df = recommend_global_k(results_df)
    recommended_path = out_root / "recommended_k.csv"
    recommended_df.to_csv(recommended_path, index=False)

    for cfg in configs:
        method = cfg["method"]
        distance = cfg["distance"]
        _plot_metric_lines(results_df, plots_dir, method, distance, "silhouette_score")
        _plot_metric_lines(results_df, plots_dir, method, distance, "davies_bouldin")
        _plot_metric_lines(results_df, plots_dir, method, distance, "cluster_balance")
        _plot_heatmap_silhouette(results_df, plots_dir, method, distance)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {recommended_path}")
    print(f"Saved plots under: {plots_dir}")
    if not recommended_df.empty:
        print("\nRecommended k:")
        print(recommended_df.to_string(index=False))
    else:
        print("\nNo recommended_k could be produced (insufficient valid rows).")


if __name__ == "__main__":
    main()
