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
    """
    Load and validate a return series from a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the return CSV file.

    Returns
    -------
    pd.Series
        Portfolio return series sorted by date and converted to float64.

    Raises
    ------
    ValueError
        If the file is empty or contains NaN/non-numeric values.
    """
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
    """
    Find the return CSV file for a requested frequency inside a strategy directory.

    Known candidate filenames are checked first; if none match, a glob pattern
    is used as a fallback.

    Parameters
    ----------
    strategy_dir : Path
        Strategy output directory.
    strategy_name : str
        Strategy name, expected to match the directory name.
    frequency : str
        'daily' or 'monthly'.

    Returns
    -------
    Path | None
        Matching file path, or None when no file is found.
    """
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
    """
    List all strategy subdirectories under the backtest root in alphabetical order.

    Parameters
    ----------
    backtests_root : Path
        Root directory containing strategy directories.

    Returns
    -------
    list[Path]
        Sorted strategy directory list, excluding hidden directories.

    Raises
    ------
    FileNotFoundError
        If the root directory does not exist.
    """
    if not backtests_root.exists():
        raise FileNotFoundError(f"Backtests directory not found: {backtests_root}")

    strategy_dirs = [
        p for p in backtests_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(strategy_dirs, key=lambda p: p.name)


def _build_monthly_from_daily(daily_returns: pd.Series) -> pd.Series:
    """
    Convert daily returns to compounded monthly returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio return series.

    Returns
    -------
    pd.Series
        Monthly compounded return series with NaN values removed.
    """
    monthly_returns = (1.0 + daily_returns).resample("M").prod() - 1.0
    monthly_returns = monthly_returns.dropna().astype("float64")
    monthly_returns.name = "portfolio_return"
    return monthly_returns


def load_strategy_monthly_returns(strategy_dir: Path) -> tuple[pd.Series, str]:
    """
    Load a monthly return series from a strategy directory.

    The monthly CSV is preferred; if it is unavailable, monthly returns are
    compounded from the daily CSV.

    Parameters
    ----------
    strategy_dir : Path
        Strategy output directory.

    Returns
    -------
    tuple[pd.Series, str]
        Monthly return series and a log message describing the data source.

    Raises
    ------
    FileNotFoundError
        If neither monthly nor daily return files can be found.
    """
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
    """
    Load a daily return series from a strategy directory.

    Parameters
    ----------
    strategy_dir : Path
        Strategy output directory.

    Returns
    -------
    tuple[pd.Series, str]
        Daily return series and a log message describing the data source.

    Raises
    ------
    FileNotFoundError
        If no daily return CSV can be found.
    """
    strategy_name = strategy_dir.name
    daily_path = _find_returns_file(strategy_dir, strategy_name, frequency="daily")
    if daily_path is None:
        raise FileNotFoundError(f"No daily return file found under: {strategy_dir}")

    daily_returns = _load_return_series(daily_path)
    return daily_returns, f"daily_file:{daily_path.name}"


def load_strategy_monthly_returns_direct(strategy_dir: Path) -> pd.Series | None:
    """Load monthly returns from the monthly CSV file only; no daily aggregation fallback.

    Parameters
    ----------
    strategy_dir : Path
        Strategy output directory containing the monthly return CSV.

    Returns
    -------
    pd.Series | None
        Monthly return series, or None if no monthly file exists or the
        file cannot be loaded.
    """
    strategy_name = strategy_dir.name
    monthly_path = _find_returns_file(strategy_dir, strategy_name, frequency="monthly")
    if monthly_path is None:
        return None
    try:
        return _load_return_series(monthly_path)
    except Exception:
        return None


def load_strategy_returns(
    strategy_dir: Path,
) -> tuple[pd.Series | None, pd.Series | None, list[str]]:
    """
    Try to load both daily and monthly return series from a strategy directory.

    Parameters
    ----------
    strategy_dir : Path
        Strategy output directory.

    Returns
    -------
    tuple[pd.Series | None, pd.Series | None, list[str]]
        Daily returns, monthly returns, and source/error log messages. Any
        unavailable series is returned as None.
    """
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
    strategy_dirs: list[Path] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a comparison table that combines monthly returns for all strategies.

    Parameters
    ----------
    backtests_root : Path
        Root directory containing strategy directories.
    strategy_dirs : list[Path] | None
        Strategy directories to process; if None, the root directory is scanned.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Comparison table and log messages. Each column is a strategy and each
        row is a month.
    """
    monthly_series_by_strategy: dict[str, pd.Series] = {}
    skipped_messages: list[str] = []

    dirs = strategy_dirs if strategy_dirs is not None else list_strategy_directories(backtests_root)
    for strategy_dir in dirs:
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
    strategy_dirs: list[Path] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a summary table by computing performance metrics for all strategies.

    Parameters
    ----------
    backtests_root : Path
        Root directory containing strategy directories.
    risk_free_rate : float
        Annual risk-free rate used in Sharpe ratio calculation.
    strategy_dirs : list[Path] | None
        Strategy directories to process; if None, the root directory is scanned.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Summary table and log messages. The table includes annualized return,
        volatility, Sharpe ratio, maximum drawdown, and Calmar ratio.
    """
    rows: list[dict] = []
    logs: list[str] = []

    dirs = strategy_dirs if strategy_dirs is not None else list_strategy_directories(backtests_root)
    for strategy_dir in dirs:
        strategy_name = strategy_dir.name
        daily_returns, monthly_returns, source_logs = load_strategy_returns(strategy_dir)

        # n_days: count from the daily CSV file only
        n_days: int | float = float("nan")
        if daily_returns is not None:
            n_days = int(len(daily_returns))

        # n_months: count from the monthly CSV file only (no daily aggregation fallback)
        n_months: int | float = float("nan")
        monthly_direct = load_strategy_monthly_returns_direct(strategy_dir)
        if monthly_direct is not None:
            n_months = int(len(monthly_direct))

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
                    "n_days": n_days,
                    "n_months": n_months,
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
            "n_days",
            "n_months",
            "start_date",
            "end_date",
        ]
    ]
    return summary_df, logs


def build_rolling_sharpe_comparison_table(
    monthly_returns_comparison: pd.DataFrame,
    window_months: int = 12,
) -> pd.DataFrame:
    """
    Compute rolling-window Sharpe ratios from a monthly returns comparison table.

    Parameters
    ----------
    monthly_returns_comparison : pd.DataFrame
        Monthly returns table where each column is a strategy.
    window_months : int
        Rolling Sharpe window length in months.

    Returns
    -------
    pd.DataFrame
        Rolling annualized Sharpe ratios for each strategy.
    """
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
    """
    Save the monthly returns comparison table to a CSV file.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table to save.
    output_path : Path
        Target CSV path; parent directories are created automatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=True, index_label="date")


def save_summary_metrics_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the summary metrics table to a CSV file.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary metrics table to save.
    output_path : Path
        Target CSV path; parent directories are created automatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def save_summary_metrics_excel(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save summary metrics to a color-formatted Excel file.

    Numeric metric columns are highlighted green (better) or red (worse)
    using a RdYlGn background gradient.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary metrics table as returned by build_summary_metrics_table.
    output_path : Path
        Target Excel path (.xlsx); parent directories are created
        automatically.
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


def save_rolling_sharpe_comparison_table(
    rolling_sharpe_df: pd.DataFrame, output_path: Path
) -> None:
    """
    Save the rolling Sharpe comparison table to a CSV file.

    Parameters
    ----------
    rolling_sharpe_df : pd.DataFrame
        Rolling Sharpe table to save.
    output_path : Path
        Target CSV path; parent directories are created automatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_sharpe_df.to_csv(output_path, index=True, index_label="date")
