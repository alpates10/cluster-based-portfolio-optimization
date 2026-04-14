from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.metrics.performance import compute_performance_metrics

_DAILY_CANDIDATES = (
    "{strategy}_daily_portfolio_returns.csv",
    "daily_portfolio_returns.csv",
)

_MONTHLY_CANDIDATES = (
    "{strategy}_monthly_portfolio_returns.csv",
    "monthly_portfolio_returns.csv",
)


def _load_return_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"Return file is empty: {csv_path}")

    if "portfolio_return" in df.columns:
        series = df["portfolio_return"]
    else:
        series = df.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce")
    if series.isna().any():
        raise ValueError(f"Return file has NaN/non-numeric values: {csv_path}")

    series = series.sort_index().astype("float64")
    series.name = "portfolio_return"
    return series


def _find_returns_file(strategy_dir: Path, strategy_name: str, frequency: str) -> Path | None:
    if frequency == "daily":
        name_candidates = [n.format(strategy=strategy_name) for n in _DAILY_CANDIDATES]
        glob_pattern = "*_daily_portfolio_returns.csv"
    elif frequency == "monthly":
        name_candidates = [n.format(strategy=strategy_name) for n in _MONTHLY_CANDIDATES]
        glob_pattern = "*_monthly_portfolio_returns.csv"
    else:
        raise ValueError("frequency must be either 'daily' or 'monthly'")

    for name in name_candidates:
        candidate = strategy_dir / name
        if candidate.exists():
            return candidate

    matches = sorted(strategy_dir.glob(glob_pattern))
    if matches:
        return matches[0]

    return None


def list_strategy_directories(backtests_root: Path) -> list[Path]:
    if not backtests_root.exists():
        raise FileNotFoundError(f"Backtests directory not found: {backtests_root}")

    strategy_dirs = [p for p in backtests_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(strategy_dirs, key=lambda p: p.name)


def _build_monthly_from_daily(daily_returns: pd.Series) -> pd.Series:
    monthly_returns = (1.0 + daily_returns).resample("M").prod() - 1.0
    monthly_returns = monthly_returns.dropna().astype("float64")
    monthly_returns.name = "portfolio_return"
    return monthly_returns


def load_strategy_monthly_returns(strategy_dir: Path) -> tuple[pd.Series, str]:
    strategy_name = strategy_dir.name

    monthly_path = _find_returns_file(strategy_dir, strategy_name, frequency="monthly")
    if monthly_path is not None:
        monthly_returns = _load_return_series(monthly_path)
        return monthly_returns, f"monthly_file:{monthly_path.name}"

    daily_path = _find_returns_file(strategy_dir, strategy_name, frequency="daily")
    if daily_path is None:
        raise FileNotFoundError(
            f"No monthly or daily return file found under: {strategy_dir}"
        )

    daily_returns = _load_return_series(daily_path)
    monthly_returns = _build_monthly_from_daily(daily_returns)
    return monthly_returns, f"aggregated_from_daily:{daily_path.name}"


def load_strategy_daily_returns(strategy_dir: Path) -> tuple[pd.Series, str]:
    strategy_name = strategy_dir.name
    daily_path = _find_returns_file(strategy_dir, strategy_name, frequency="daily")
    if daily_path is None:
        raise FileNotFoundError(f"No daily return file found under: {strategy_dir}")

    daily_returns = _load_return_series(daily_path)
    return daily_returns, f"daily_file:{daily_path.name}"


def load_strategy_returns(strategy_dir: Path) -> tuple[pd.Series | None, pd.Series | None, list[str]]:
    logs: list[str] = []

    daily_returns: pd.Series | None = None
    monthly_returns: pd.Series | None = None

    try:
        daily_returns, src_daily = load_strategy_daily_returns(strategy_dir)
        logs.append(src_daily)
    except Exception as exc:
        logs.append(f"daily_unavailable:{exc}")

    try:
        monthly_returns, src_monthly = load_strategy_monthly_returns(strategy_dir)
        logs.append(src_monthly)
    except Exception as exc:
        logs.append(f"monthly_unavailable:{exc}")

    return daily_returns, monthly_returns, logs


def build_monthly_returns_comparison_table(
    backtests_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    monthly_series_by_strategy: dict[str, pd.Series] = {}
    skipped_messages: list[str] = []

    for strategy_dir in list_strategy_directories(backtests_root):
        strategy_name = strategy_dir.name
        try:
            monthly_returns, source = load_strategy_monthly_returns(strategy_dir)
            monthly_series_by_strategy[strategy_name] = monthly_returns.rename(strategy_name)
            skipped_messages.append(f"[OK] {strategy_name} ({source})")
        except Exception as exc:
            skipped_messages.append(f"[SKIP] {strategy_name}: {exc}")

    if not monthly_series_by_strategy:
        return pd.DataFrame(), skipped_messages

    comparison_df = pd.concat(monthly_series_by_strategy.values(), axis=1, join="outer")
    comparison_df = comparison_df.sort_index()
    comparison_df.index.name = "date"
    comparison_df = comparison_df.sort_index(axis=1)

    return comparison_df, skipped_messages


def build_summary_metrics_table(
    backtests_root: Path,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    logs: list[str] = []

    for strategy_dir in list_strategy_directories(backtests_root):
        strategy_name = strategy_dir.name
        daily_returns, monthly_returns, source_logs = load_strategy_returns(strategy_dir)

        try:
            if daily_returns is not None:
                metrics = compute_performance_metrics(
                    daily_returns,
                    periods_per_year=252,
                    risk_free_rate=risk_free_rate,
                )
                logs.append(f"[OK] {strategy_name} (metrics_from_daily)")
            elif monthly_returns is not None:
                metrics = compute_performance_metrics(
                    monthly_returns,
                    periods_per_year=12,
                    risk_free_rate=risk_free_rate,
                )
                logs.append(f"[OK] {strategy_name} (metrics_from_monthly)")
            else:
                logs.append(f"[SKIP] {strategy_name}: no usable daily/monthly return series")
                for src_log in source_logs:
                    logs.append(f"  - {strategy_name}: {src_log}")
                continue

            rows.append(
                {
                    "strategy": strategy_name,
                    "annualized_return": metrics["annualized_return"],
                    "annualized_volatility": metrics["annualized_volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "calmar_ratio": metrics["calmar_ratio"],
                    "n_obs": metrics["n_obs"],
                    "start_date": metrics["start_date"],
                    "end_date": metrics["end_date"],
                }
            )
        except Exception as exc:
            logs.append(
                f"[SKIP] {strategy_name}: metrics computation failed ({exc})"
            )

        for src_log in source_logs:
            logs.append(f"  - {strategy_name}: {src_log}")

    if not rows:
        return pd.DataFrame(), logs

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("strategy").reset_index(drop=True)
    summary_df = summary_df[
        [
            "strategy",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "n_obs",
            "start_date",
            "end_date",
        ]
    ]
    return summary_df, logs


def build_rolling_sharpe_comparison_table(
    monthly_returns_comparison: pd.DataFrame,
    window_months: int = 12,
) -> pd.DataFrame:
    if monthly_returns_comparison.empty:
        return pd.DataFrame()

    rolling_mean = monthly_returns_comparison.rolling(window=window_months).mean()
    rolling_std = monthly_returns_comparison.rolling(window=window_months).std(ddof=1)

    annualized_mean = rolling_mean * 12.0
    annualized_std = rolling_std * np.sqrt(12.0)
    rolling_sharpe = annualized_mean / annualized_std
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
    rolling_sharpe.index.name = "date"
    rolling_sharpe = rolling_sharpe.sort_index(axis=1)
    return rolling_sharpe


def save_monthly_returns_comparison_table(comparison_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=True, index_label="date")


def save_summary_metrics_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def save_summary_metrics_excel(summary_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save summary metrics to a color-formatted Excel file.

    Green = better, Red = worse.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        summary_df.to_excel(output_path, index=False, sheet_name="summary_metrics")
        return

    high_is_good_cols = [
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    ]
    low_is_good_cols = [
        "annualized_volatility",
    ]

    available_high = [c for c in high_is_good_cols if c in summary_df.columns]
    available_low = [c for c in low_is_good_cols if c in summary_df.columns]

    styler = summary_df.style
    if available_high:
        styler = styler.background_gradient(cmap="RdYlGn", subset=available_high)
    if available_low:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=available_low)

    # Basic numeric formatting for readability
    fmt_map = {
        "annualized_return": "{:.4f}",
        "annualized_volatility": "{:.4f}",
        "sharpe_ratio": "{:.4f}",
        "max_drawdown": "{:.4f}",
        "calmar_ratio": "{:.4f}",
    }
    fmt_map = {k: v for k, v in fmt_map.items() if k in summary_df.columns}
    if fmt_map:
        styler = styler.format(fmt_map)

    styler.to_excel(output_path, index=False, sheet_name="summary_metrics")


def save_rolling_sharpe_comparison_table(rolling_sharpe_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_sharpe_df.to_csv(output_path, index=True, index_label="date")
