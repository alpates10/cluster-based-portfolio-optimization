"""Full pipeline runner: backtest → rolling val → permutation → reports → plots.

Prerequisite: data/processed/{universe}/returns_final.csv must exist.
If it is missing, the runner exits; run prepare_data.py first.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

UNIVERSE_CHOICES = ["sp500", "bist100", "nifty50", "sse"]

STEPS = [
    {
        "desc": "Backtest — all strategies (run_backtest --mode all)",
        "script": "scripts/run_backtest.py",
        "extra_args": ["--mode", "all"],
    },
    {
        "desc": "Rolling validation backtest (run_rolling_validation_backtest)",
        "script": "scripts/run_rolling_validation_backtest.py",
        "extra_args": [],
    },
    {
        "desc": "Permutation backtest — both methods (run_permutation_backtest)",
        "script": "scripts/run_permutation_backtest.py",
        "extra_args": ["--method", "both"],
    },
    {
        "desc": "Performance report — classical (make_performance_report --mode classical)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "classical"],
    },
    {
        "desc": "Performance report — clustering (make_performance_report --mode clustering)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "clustering"],
    },
    {
        "desc": "Performance report — rolling validation (make_performance_report)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "rolling_validation"],
    },
    {
        "desc": "Performance report — all, no rolling (make_performance_report)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "all_no_rolling"],
    },
    {
        "desc": "Performance report — all (make_performance_report --mode all)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "all"],
    },
    {
        "desc": "Permutation summary report (make_permutation_summary)",
        "script": "scripts/make_permutation_summary.py",
        "extra_args": [],
    },
    {
        "desc": "Clustering visuals (make_clustering_plots)",
        "script": "scripts/make_clustering_plots.py",
        "extra_args": [],
    },
]


def fmt_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact minutes/seconds string.

    Parameters
    ----------
    seconds : float
        Elapsed time in seconds.

    Returns
    -------
    str
        Human-readable string such as ``"2d 15s"`` or ``"43s"``.
    """
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}d {secs}s"
    return f"{secs}s"


def main() -> None:
    """Run the full project pipeline for one universe."""
    parser = argparse.ArgumentParser(
        description="Full pipeline: backtest → rolling val → permutation → reports → plots."
    )
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of running them.",
    )
    parser.add_argument(
        "--skip-strategies",
        default="",
        help="Comma-separated strategy names to skip; forwarded to run_backtest.py.",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip permutation backtest and summary report.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    returns_csv = project_root / "data" / "processed" / args.universe / "returns_final.csv"

    if not returns_csv.exists():
        print(f"Error: data for {args.universe} was not found. Run prepare_data.py first.")
        sys.exit(1)

    python = sys.executable
    total_steps = len(STEPS)

    print(f"\nUniverse : {args.universe}")
    if args.dry_run:
        print("Mode     : dry-run (commands will not be executed)")
    print(f"Step count: {total_steps}\n")
    print("=" * 60)

    pipeline_start = time.monotonic()

    for i, step in enumerate(STEPS, start=1):
        print(f"\n[{i}/{total_steps}] {step['desc']}")

        if args.skip_permutation and step["script"] in (
            "scripts/run_permutation_backtest.py",
            "scripts/make_permutation_summary.py",
        ):
            print("  Skipped — --skip-permutation is active.")
            continue

        cmd = [python, str(project_root / step["script"]), "--universe", args.universe]
        cmd += step["extra_args"]
        if args.skip_strategies and step["script"] == "scripts/run_backtest.py":
            cmd += ["--skip-strategies", args.skip_strategies]
        cmd_str = " ".join(cmd[1:])  # Hide the Python binary for readability.

        if args.dry_run:
            print(f"  $ python {cmd_str}")
            continue

        step_start = time.monotonic()
        result = subprocess.run(cmd, cwd=project_root)
        elapsed = time.monotonic() - step_start

        if result.returncode != 0:
            print(f"\nERROR: step {i} failed (exit code {result.returncode}). Stopping pipeline.")
            sys.exit(result.returncode)

        print(f"  Completed — {fmt_duration(elapsed)}")

    if not args.dry_run:
        total_elapsed = time.monotonic() - pipeline_start
        print("\n" + "=" * 60)
        print(f"Pipeline completed. Total duration: {fmt_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
