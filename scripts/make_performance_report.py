from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.backtest_summary import (  
    build_rolling_sharpe_comparison_table,
    build_monthly_returns_comparison_table,
    build_summary_metrics_table,
    save_rolling_sharpe_comparison_table,
    save_monthly_returns_comparison_table,
    save_summary_metrics_excel,
    save_summary_metrics_table,
)


def main() -> None:
    backtests_dir = PROJECT_ROOT / "data" / "processed" / "backtests"
    monthly_output = backtests_dir / "monthly_returns_comparison.csv"
    summary_output = backtests_dir / "summary_metrics.csv"
    summary_excel_output = backtests_dir / "summary_metrics.xlsx"
    rolling_sharpe_output = backtests_dir / "rolling_sharpe_comparison.csv"
    rolling_window_months = 12
    risk_free_rate = 0.0

    monthly_df, monthly_logs = build_monthly_returns_comparison_table(
        backtests_root=backtests_dir,
    )

    if monthly_df.empty:
        raise ValueError(
            f"No usable monthly return series found under strategy folders in: {backtests_dir}"
        )

    summary_df, summary_logs = build_summary_metrics_table(
        backtests_root=backtests_dir,
        risk_free_rate=risk_free_rate,
    )
    rolling_sharpe_df = build_rolling_sharpe_comparison_table(
        monthly_returns_comparison=monthly_df,
        window_months=rolling_window_months,
    )

    save_monthly_returns_comparison_table(monthly_df, monthly_output)
    if not summary_df.empty:
        save_summary_metrics_table(summary_df, summary_output)
        save_summary_metrics_excel(summary_df, summary_excel_output)
    save_rolling_sharpe_comparison_table(rolling_sharpe_df, rolling_sharpe_output)

    print(f"Monthly comparison saved to: {monthly_output} | shape={monthly_df.shape}")
    print(f"Summary metrics saved to: {summary_output} | rows={len(summary_df)}")
    print(f"Summary metrics Excel saved to: {summary_excel_output}")
    print(
        f"Rolling Sharpe saved to: {rolling_sharpe_output} | "
        f"shape={rolling_sharpe_df.shape} | window={rolling_window_months}"
    )
    print("Strategies:", ", ".join(monthly_df.columns))

    print("\nMonthly load logs:")
    for line in monthly_logs:
        print(line)

    print("\nSummary logs:")
    for line in summary_logs:
        print(line)

    print("\nMonthly comparison (head):")
    print(monthly_df.head(12).to_string())
    print("\nSummary metrics:")
    if summary_df.empty:
        print("No summary rows were generated.")
    else:
        print(summary_df.to_string(index=False))
    print("\nRolling Sharpe (head):")
    print(rolling_sharpe_df.head(12).to_string())


if __name__ == "__main__":
    main()
