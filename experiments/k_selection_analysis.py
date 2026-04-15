"""
experiments/k_selection_analysis.py

Research script: measure how well different k-selection methods predict
the best-performing k in each rebalance period.

Run from the project root:
    python experiments/k_selection_analysis.py

Outputs
-------
experiments/plots/kmeans_cumulative_returns.png
experiments/plots/kmedoids_cumulative_returns.png
experiments/plots/kmeans_hit_rate.png
experiments/plots/kmedoids_hit_rate.png
experiments/plots/kmeans_k_distribution.png
experiments/plots/kmedoids_k_distribution.png
experiments/plots/silhouette_vs_return_correlation.png
experiments/k_selection_summary_kmeans.csv
experiments/k_selection_summary_kmedoids.csv
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.algorithms import kmeans_labels, kmedoids_labels_from_distance
from src.clustering.distances import correlation_distance_matrix, dtw_distance_matrix
from src.clustering.representations import raw_return_representation
from src.data.load import load_returns_csv

# ── Config ────────────────────────────────────────────────────────────────────
K_VALUES        = list(range(2, 9))           # [2, 3, 4, 5, 6, 7, 8]
ESTIMATION_WINDOW = 756
BACKTESTS_DIR   = PROJECT_ROOT / "data" / "processed" / "backtests"
DTW_CACHE_DIR   = PROJECT_ROOT / "data" / "processed" / "dtw_cache"
PLOTS_DIR       = PROJECT_ROOT / "experiments" / "plots"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RANDOM_SEED     = 42

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS_ORDER = [
    "argmax_all", "argmax_no2", "delta_method",
    "fixed_k2", "fixed_k3", "fixed_k4", "fixed_k5",
    "oracle", "random_k",
]

PALETTE = {
    "argmax_all":   "#1f77b4",
    "argmax_no2":   "#ff7f0e",
    "delta_method": "#2ca02c",
    "fixed_k2":     "#d62728",
    "fixed_k3":     "#9467bd",
    "fixed_k4":     "#8c564b",
    "fixed_k5":     "#e377c2",
    "oracle":       "#111111",
    "random_k":     "#aaaaaa",
}

LINE_STYLES = {"oracle": "--", "random_k": ":"}
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Load monthly returns per k
# ═══════════════════════════════════════════════════════════════════════════════

def _load_monthly_returns_for_type(strategy_type: str) -> pd.DataFrame:
    """Return DataFrame: index=month-end date, columns=k (int), values=monthly return."""
    if strategy_type == "kmeans":
        parent    = BACKTESTS_DIR / "kmeans_equal_weight"
        name_tmpl = "kmeans_k{k}"
    else:
        parent    = BACKTESTS_DIR / "kmedoids_dtw_equal_weight"
        name_tmpl = "kmedoids_dtw_k{k}"

    series: list[pd.Series] = []
    for k in K_VALUES:
        name = name_tmpl.format(k=k)
        path = parent / name / f"{name}_monthly_portfolio_returns.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        s  = df.iloc[:, 0].rename(k)
        series.append(s)

    monthly_df = pd.concat(series, axis=1).sort_index()
    monthly_df.columns = pd.Index(K_VALUES, dtype=int)
    return monthly_df


def _rebalance_dates_from_weights() -> list[pd.Timestamp]:
    path = BACKTESTS_DIR / "kmeans_equal_weight" / "kmeans_k5" / "kmeans_k5_weights.csv"
    w    = pd.read_csv(path, index_col=0, parse_dates=True)
    return w.index.tolist()


def _align_returns_to_rebalance(
    monthly_df: pd.DataFrame,
    month_to_rebalance: dict[tuple[int, int], pd.Timestamp],
) -> pd.DataFrame:
    """Re-index monthly_df so each row is keyed by the rebalance_date of that month."""
    rows, dates = [], []
    for date in monthly_df.index:
        key = (date.year, date.month)
        if key in month_to_rebalance:
            dates.append(month_to_rebalance[key])
            rows.append(monthly_df.loc[date].values)

    result = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=K_VALUES)
    result.index.name = "rebalance_date"
    return result.sort_index()


print("=" * 70)
print("ADIM 1 — Loading monthly returns per k")
print("=" * 70)

monthly_kmeans   = _load_monthly_returns_for_type("kmeans")
monthly_kmedoids = _load_monthly_returns_for_type("kmedoids")

rebalance_dates   = _rebalance_dates_from_weights()
month_to_rebalance: dict[tuple[int, int], pd.Timestamp] = {
    (d.year, d.month): d for d in rebalance_dates
}

returns_kmeans   = _align_returns_to_rebalance(monthly_kmeans,   month_to_rebalance)
returns_kmedoids = _align_returns_to_rebalance(monthly_kmedoids, month_to_rebalance)

print(f"  KMeans   returns: {returns_kmeans.shape}   "
      f"[{returns_kmeans.index[0].date()} .. {returns_kmeans.index[-1].date()}]")
print(f"  KMedoids returns: {returns_kmedoids.shape}   "
      f"[{returns_kmedoids.index[0].date()} .. {returns_kmedoids.index[-1].date()}]")


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — Silhouette scores per rebalance window
# ═══════════════════════════════════════════════════════════════════════════════

def _dtw_cache_key(window_returns: pd.DataFrame) -> str:
    idx   = window_returns.index
    start = str(idx[0].date())  if hasattr(idx[0], "date") else str(idx[0])
    end   = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
    tickers = ",".join(sorted(window_returns.columns.tolist()))
    raw = f"{start}|{end}|{tickers}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def compute_silhouette_kmeans(
    all_returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    records: list[dict] = []
    for rd in tqdm(rebalance_dates, desc="Silhouette KMeans"):
        t = all_returns.index.get_loc(rd)
        if t < ESTIMATION_WINDOW:
            continue
        window       = all_returns.iloc[t - ESTIMATION_WINDOW:t]
        asset_matrix = raw_return_representation(window)
        corr_dist    = correlation_distance_matrix(asset_matrix)

        row: dict = {"rebalance_date": rd}
        for k in K_VALUES:
            labels = kmeans_labels(asset_matrix, n_clusters=k, random_state=RANDOM_SEED)
            row[k] = float(silhouette_score(corr_dist, labels, metric="precomputed"))
        records.append(row)

    df = pd.DataFrame(records).set_index("rebalance_date")
    df.columns = pd.Index([int(c) for c in df.columns], dtype=int)
    return df


def compute_silhouette_kmedoids(
    all_returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    records: list[dict] = []
    for rd in tqdm(rebalance_dates, desc="Silhouette KMedoids"):
        t = all_returns.index.get_loc(rd)
        if t < ESTIMATION_WINDOW:
            continue
        window       = all_returns.iloc[t - ESTIMATION_WINDOW:t]
        asset_matrix = raw_return_representation(window)
        cache_key    = _dtw_cache_key(window)
        dtw_dist     = dtw_distance_matrix(
            asset_matrix,
            n_jobs=-1,
            cache_dir=str(DTW_CACHE_DIR),
            cache_key=cache_key,
        )

        row: dict = {"rebalance_date": rd}
        for k in K_VALUES:
            labels = kmedoids_labels_from_distance(dtw_dist, n_clusters=k, random_state=RANDOM_SEED)
            row[k] = float(silhouette_score(dtw_dist, labels, metric="precomputed"))
        records.append(row)

    df = pd.DataFrame(records).set_index("rebalance_date")
    df.columns = pd.Index([int(c) for c in df.columns], dtype=int)
    return df


print("\n" + "=" * 70)
print("ADIM 2 — Computing silhouette scores per rebalance window")
print("=" * 70)

all_returns = load_returns_csv(PROJECT_ROOT / "data" / "processed" / "returns_final.csv")

print("\nKMeans silhouette (uses correlation distance):")
silhouette_kmeans_df = compute_silhouette_kmeans(all_returns, rebalance_dates)

print("\nKMedoids silhouette (uses DTW distance, loads from cache when available):")
silhouette_kmedoids_df = compute_silhouette_kmedoids(all_returns, rebalance_dates)

print(f"\n  Silhouette KMeans   shape: {silhouette_kmeans_df.shape}")
print(f"  Silhouette KMedoids shape: {silhouette_kmedoids_df.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Apply k-selection methods per rebalance period
# ═══════════════════════════════════════════════════════════════════════════════

def _delta_rule(sil_row: pd.Series) -> int:
    """Select k by largest silhouette-delta; fallback to k=2 if all deltas negative."""
    k_vals = sorted(sil_row.index.tolist())
    deltas = {
        k_vals[i]: sil_row[k_vals[i]] - sil_row[k_vals[i - 1]]
        for i in range(1, len(k_vals))
    }
    if all(d < 0 for d in deltas.values()):
        return k_vals[0]
    return max(deltas, key=lambda k: deltas[k])


def apply_methods(
    silhouette_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each rebalance date select a k under each method and look up
    the actual monthly return for that k.

    Returns
    -------
    selected_k_df   : shape (T, n_methods)  — integer k choices
    returns_sel_df  : shape (T, n_methods)  — realised monthly returns
    """
    common = silhouette_df.index.intersection(returns_df.index).sort_values()
    sil    = silhouette_df.loc[common]
    rets   = returns_df.loc[common]
    n      = len(common)

    rng        = np.random.default_rng(RANDOM_SEED)
    random_ks  = [int(k) for k in rng.choice(K_VALUES, size=n)]

    methods_k: dict[str, list[int]] = {m: [] for m in METHODS_ORDER}

    for i, date in enumerate(common):
        row_sil = sil.loc[date]
        row_ret = rets.loc[date]

        methods_k["argmax_all"].append(int(row_sil.idxmax()))
        methods_k["argmax_no2"].append(int(row_sil.loc[row_sil.index[row_sil.index != 2]].idxmax()))
        methods_k["delta_method"].append(_delta_rule(row_sil))
        methods_k["fixed_k2"].append(2)
        methods_k["fixed_k3"].append(3)
        methods_k["fixed_k4"].append(4)
        methods_k["fixed_k5"].append(5)
        methods_k["oracle"].append(int(row_ret.idxmax()))
        methods_k["random_k"].append(random_ks[i])

    selected_k_df = pd.DataFrame(methods_k, index=common)
    selected_k_df.index.name = "rebalance_date"

    # Vectorised return lookup: for each (date, method), pick rets[date, k]
    k_to_col = {k: idx for idx, k in enumerate(K_VALUES)}
    rets_arr  = rets.values  # shape (T, 7)

    returns_dict: dict[str, np.ndarray] = {}
    for method, ks in methods_k.items():
        col_idxs             = np.array([k_to_col[k] for k in ks])
        row_idxs             = np.arange(n)
        returns_dict[method] = rets_arr[row_idxs, col_idxs]

    returns_sel_df = pd.DataFrame(returns_dict, index=common)
    returns_sel_df.index.name = "rebalance_date"

    return selected_k_df, returns_sel_df


print("\n" + "=" * 70)
print("ADIM 3 — Applying k-selection methods")
print("=" * 70)

selected_k_kmeans,   returns_sel_kmeans   = apply_methods(silhouette_kmeans_df,   returns_kmeans)
selected_k_kmedoids, returns_sel_kmedoids = apply_methods(silhouette_kmedoids_df, returns_kmedoids)

print(f"  KMeans   selected_k shape : {selected_k_kmeans.shape}")
print(f"  KMedoids selected_k shape : {selected_k_kmedoids.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _portfolio_metrics(r: pd.Series) -> dict:
    """Annualised metrics from a monthly return series."""
    n             = len(r)
    ppy           = 12
    growth        = float((1 + r).prod())
    ann_ret       = (growth ** (ppy / n) - 1.0) if growth > 0 else -1.0
    ann_vol       = float(r.std(ddof=1) * np.sqrt(ppy))
    sharpe        = float(r.mean() / r.std(ddof=1) * np.sqrt(ppy)) if r.std(ddof=1) > 0 else np.nan
    wealth        = (1 + r).cumprod()
    max_dd        = float((wealth / wealth.cummax() - 1).min())
    return {
        "ann_return_%":    round(ann_ret  * 100, 4),
        "ann_volatility_%": round(ann_vol * 100, 4),
        "sharpe_ratio":    round(sharpe,          4),
        "max_drawdown_%":  round(max_dd   * 100,  4),
    }


def compute_all_metrics(
    selected_k_df:  pd.DataFrame,
    returns_sel_df: pd.DataFrame,
    silhouette_df:  pd.DataFrame,
    returns_df:     pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    metrics_df   : per-method performance + hit rate + k-distribution string
    spearman_df  : per-k Spearman correlation of silhouette vs next-month return
    """
    oracle_k = selected_k_df["oracle"]
    rows = []

    for method in METHODS_ORDER:
        r      = returns_sel_df[method]
        sel_k  = selected_k_df[method]
        perf   = _portfolio_metrics(r)
        hit    = float((sel_k == oracle_k).mean()) * 100.0
        k_dist = dict(sorted(sel_k.value_counts().to_dict().items()))
        rows.append({
            "method":          method,
            "hit_rate_%":      round(hit, 2),
            "k_distribution":  str(k_dist),
            **perf,
        })

    metrics_df = pd.DataFrame(rows).set_index("method")

    # Spearman: sil[k] → next-month return[k]
    common_sil   = silhouette_df.index.intersection(returns_df.index).sort_values()
    sil_aligned  = silhouette_df.loc[common_sil]
    ret_aligned  = returns_df.loc[common_sil]

    spearman_rows = []
    for k in K_VALUES:
        x = sil_aligned[k].values[:-1]   # current-month silhouette
        y = ret_aligned[k].values[1:]    # next-month return
        rho, pval = stats.spearmanr(x, y)
        spearman_rows.append({
            "k":            k,
            "spearman_rho": round(float(rho),  4),
            "p_value":      round(float(pval), 4),
        })

    spearman_df = pd.DataFrame(spearman_rows).set_index("k")
    return metrics_df, spearman_df


print("\n" + "=" * 70)
print("ADIM 4 — Computing metrics")
print("=" * 70)

metrics_kmeans,   spearman_kmeans   = compute_all_metrics(
    selected_k_kmeans,   returns_sel_kmeans,   silhouette_kmeans_df,   returns_kmeans)
metrics_kmedoids, spearman_kmedoids = compute_all_metrics(
    selected_k_kmedoids, returns_sel_kmedoids, silhouette_kmedoids_df, returns_kmedoids)


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Plots
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 1 & 2: Cumulative returns ────────────────────────────────────────────

def plot_cumulative_returns(returns_sel_df: pd.DataFrame, title: str, save_path: Path) -> None:
    cum = (1 + returns_sel_df).cumprod() - 1
    fig, ax = plt.subplots(figsize=(14, 7))

    for method in METHODS_ORDER:
        ax.plot(
            cum.index, cum[method],
            label=method,
            color=PALETTE[method],
            linestyle=LINE_STYLES.get(method, "-"),
            linewidth=2.0 if method in ("oracle", "random_k") else 1.4,
            alpha=0.85,
        )

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Cumulative Return", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.7)
    ax.grid(True, alpha=0.25)
    _save(fig, save_path)


# ── Plot 3 & 4: Hit rate ──────────────────────────────────────────────────────

def plot_hit_rate(metrics_df: pd.DataFrame, title: str, save_path: Path) -> None:
    compare_methods = [m for m in METHODS_ORDER if m != "oracle"]
    hr     = metrics_df.loc[compare_methods, "hit_rate_%"]
    colors = [PALETTE[m] for m in compare_methods]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(compare_methods, hr.values, color=colors, alpha=0.85, edgecolor="white", zorder=3)

    random_hr = metrics_df.loc["random_k", "hit_rate_%"] if "random_k" in metrics_df.index else None
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.2, label="oracle (100%)", zorder=4)
    if random_hr is not None:
        ax.axhline(random_hr, color="#aaaaaa", linestyle=":", linewidth=1.2,
                   label=f"random ({random_hr:.1f}%)", zorder=4)

    for x_pos, val in enumerate(hr.values):
        ax.text(x_pos, val + 0.4, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    _save(fig, save_path)


# ── Plot 5 & 6: k-distribution ───────────────────────────────────────────────

def plot_k_distribution(selected_k_df: pd.DataFrame, title: str, save_path: Path) -> None:
    plot_methods = [m for m in METHODS_ORDER
                    if m not in ("fixed_k2", "fixed_k3", "fixed_k4", "fixed_k5")]
    # Count matrix: rows=method, cols=k
    mat = pd.DataFrame(0, index=plot_methods, columns=K_VALUES, dtype=int)
    for method in plot_methods:
        for k, cnt in selected_k_df[method].value_counts().items():
            if k in mat.columns:
                mat.loc[method, k] = int(cnt)

    k_colors = matplotlib.colormaps["tab10"].resampled(len(K_VALUES)).colors

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(plot_methods))
    for i, k in enumerate(K_VALUES):
        vals = mat[k].values.astype(float)
        ax.bar(plot_methods, vals, bottom=bottom,
               label=f"k={k}", color=k_colors[i], alpha=0.85, edgecolor="white")
        bottom += vals

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel("Number of Rebalance Periods", fontsize=10)
    ax.legend(title="k", loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, save_path)


# ── Plot 7: Silhouette vs next-month return (2×7 combined) ───────────────────

def plot_silhouette_vs_return_combined(
    silhouette_kmeans:   pd.DataFrame,
    returns_kmeans:      pd.DataFrame,
    silhouette_kmedoids: pd.DataFrame,
    returns_kmedoids:    pd.DataFrame,
    save_path: Path,
) -> None:
    """2-row × 7-column grid: row 0 = KMeans, row 1 = KMedoids."""
    fig, axes = plt.subplots(2, 7, figsize=(21, 8), sharey=False)

    rows_data = [
        ("KMeans (corr dist)",  silhouette_kmeans,   returns_kmeans),
        ("KMedoids (DTW dist)", silhouette_kmedoids, returns_kmedoids),
    ]

    for row_idx, (row_label, sil_df, ret_df) in enumerate(rows_data):
        common = sil_df.index.intersection(ret_df.index).sort_values()
        sil    = sil_df.loc[common]
        ret    = ret_df.loc[common]

        for col_idx, k in enumerate(K_VALUES):
            ax     = axes[row_idx, col_idx]
            x      = sil[k].values[:-1]    # current-month silhouette
            y      = ret[k].values[1:]     # next-month return
            rho, p = stats.spearmanr(x, y)

            ax.scatter(x, y, s=12, alpha=0.45, color=f"C{col_idx}", linewidths=0)

            # Trend line
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 80)
            ax.plot(x_line, slope * x_line + intercept,
                    color="crimson", linewidth=1.0, alpha=0.7)

            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.set_title(f"k={k}  ρ={rho:.3f}{star}", fontsize=8)
            ax.set_xlabel("Silhouette", fontsize=7)
            ax.tick_params(labelsize=6)

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nNext-Month Ret", fontsize=7)

    fig.suptitle(
        "Silhouette Score vs Next-Month Return  (* p<0.05  ** p<0.01)",
        fontsize=12, fontweight="bold",
    )
    _save(fig, save_path)


print("\n" + "=" * 70)
print("ADIM 5 — Generating plots")
print("=" * 70)

plot_cumulative_returns(
    returns_sel_kmeans,
    "KMeans — Cumulative Returns by k-Selection Method",
    PLOTS_DIR / "kmeans_cumulative_returns.png",
)
plot_cumulative_returns(
    returns_sel_kmedoids,
    "KMedoids DTW — Cumulative Returns by k-Selection Method",
    PLOTS_DIR / "kmedoids_cumulative_returns.png",
)
plot_hit_rate(
    metrics_kmeans,
    "KMeans — Hit Rate vs Oracle k",
    PLOTS_DIR / "kmeans_hit_rate.png",
)
plot_hit_rate(
    metrics_kmedoids,
    "KMedoids DTW — Hit Rate vs Oracle k",
    PLOTS_DIR / "kmedoids_hit_rate.png",
)
plot_k_distribution(
    selected_k_kmeans,
    "KMeans — Distribution of Selected k per Method",
    PLOTS_DIR / "kmeans_k_distribution.png",
)
plot_k_distribution(
    selected_k_kmedoids,
    "KMedoids DTW — Distribution of Selected k per Method",
    PLOTS_DIR / "kmedoids_k_distribution.png",
)
plot_silhouette_vs_return_combined(
    silhouette_kmeans_df,
    returns_kmeans,
    silhouette_kmedoids_df,
    returns_kmedoids,
    PLOTS_DIR / "silhouette_vs_return_correlation.png",
)


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 6 — Summary tables
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ADIM 6 — Saving summary tables")
print("=" * 70)

metrics_kmeans.to_csv(  EXPERIMENTS_DIR / "k_selection_summary_kmeans.csv")
metrics_kmedoids.to_csv(EXPERIMENTS_DIR / "k_selection_summary_kmedoids.csv")

print(f"  Saved: k_selection_summary_kmeans.csv")
print(f"  Saved: k_selection_summary_kmedoids.csv")

_col_order = ["hit_rate_%", "ann_return_%", "ann_volatility_%", "sharpe_ratio", "max_drawdown_%"]

print("\n" + "=" * 70)
print("KMeans — k-Selection Method Summary")
print("=" * 70)
print(metrics_kmeans[_col_order].to_string())

print("\n" + "=" * 70)
print("KMedoids DTW — k-Selection Method Summary")
print("=" * 70)
print(metrics_kmedoids[_col_order].to_string())

print("\n" + "=" * 70)
print("Spearman ρ: Silhouette Score → Next-Month Return")
print("=" * 70)
combined_spearman = spearman_kmeans.rename(
    columns=lambda c: f"kmeans_{c}"
).join(
    spearman_kmedoids.rename(columns=lambda c: f"kmedoids_{c}")
)
print(combined_spearman.to_string())

print("\n" + "=" * 70)
print(f"Done.  Plots → {PLOTS_DIR}")
print(f"       CSVs  → {EXPERIMENTS_DIR}")
print("=" * 70)
