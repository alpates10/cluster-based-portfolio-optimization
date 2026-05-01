"""Full pipeline runner: backtest → rolling val → permutation → reports → plots.

Ön koşul: data/processed/{universe}/returns_final.csv mevcut olmalı.
Yoksa hata verir; önce prepare_data.py çalıştırın.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

UNIVERSE_CHOICES = ["sp500", "bist100", "nifty50", "sse"]

STEPS = [
    {
        "desc": "Backtest — tüm stratejiler (run_backtest --mode all)",
        "script": "scripts/run_backtest.py",
        "extra_args": ["--mode", "all"],
    },
    {
        "desc": "Rolling validation backtest (run_rolling_validation_backtest)",
        "script": "scripts/run_rolling_validation_backtest.py",
        "extra_args": [],
    },
    {
        "desc": "Permutation backtest — her iki yöntem (run_permutation_backtest)",
        "script": "scripts/run_permutation_backtest.py",
        "extra_args": ["--method", "both"],
    },
    {
        "desc": "Performans raporu — klasik (make_performance_report --mode classical)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "classical"],
    },
    {
        "desc": "Performans raporu — clustering (make_performance_report --mode clustering)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "clustering"],
    },
    {
        "desc": "Performans raporu — rolling validation (make_performance_report --mode rolling_validation)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "rolling_validation"],
    },
    {
        "desc": "Performans raporu — tümü (make_performance_report --mode all_no_rolling)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "all_no_rolling"],
    },
    {
        "desc": "Performans raporu — all (make_performance_report --mode all)",
        "script": "scripts/make_performance_report.py",
        "extra_args": ["--mode", "all"],
    },
    {
        "desc": "Permutation özet raporu (make_permutation_summary)",
        "script": "scripts/make_permutation_summary.py",
        "extra_args": [],
    },
    {
        "desc": "Clustering görselleri (make_clustering_plots)",
        "script": "scripts/make_clustering_plots.py",
        "extra_args": [],
    },
]


def fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}d {secs}s"
    return f"{secs}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tam pipeline: backtest → rolling val → permutation → raporlar → görseller."
    )
    parser.add_argument("--universe", default="sp500", choices=UNIVERSE_CHOICES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Komutları çalıştırmak yerine sadece yazdır.",
    )
    parser.add_argument(
        "--skip-strategies",
        default="",
        help="Comma-separated strategy names to skip; forwarded to run_backtest.py.",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Permutation backtest ve özet raporunu atla.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    returns_csv = project_root / "data" / "processed" / args.universe / "returns_final.csv"

    if not returns_csv.exists():
        print(f"Hata: {args.universe} için veri bulunamadı. Önce prepare_data.py çalıştırın.")
        sys.exit(1)

    python = sys.executable
    total_steps = len(STEPS)

    print(f"\nUniverse : {args.universe}")
    if args.dry_run:
        print("Mod      : dry-run (komutlar çalıştırılmayacak)")
    print(f"Adım sayısı: {total_steps}\n")
    print("=" * 60)

    pipeline_start = time.monotonic()

    for i, step in enumerate(STEPS, start=1):
        print(f"\n[{i}/{total_steps}] {step['desc']}")

        if args.skip_permutation and step["script"] in (
            "scripts/run_permutation_backtest.py",
            "scripts/make_permutation_summary.py",
        ):
            print("  Atlandı — --skip-permutation aktif.")
            continue

        cmd = [python, str(project_root / step["script"]), "--universe", args.universe]
        cmd += step["extra_args"]
        if args.skip_strategies and step["script"] == "scripts/run_backtest.py":
            cmd += ["--skip-strategies", args.skip_strategies]
        cmd_str = " ".join(cmd[1:])  # python binary'yi gizle, okunabilirlik için

        if args.dry_run:
            print(f"  $ python {cmd_str}")
            continue

        step_start = time.monotonic()
        result = subprocess.run(cmd, cwd=project_root)
        elapsed = time.monotonic() - step_start

        if result.returncode != 0:
            print(f"\nHATA: Adım {i} basarisiz oldu (exit code {result.returncode}). Pipeline durduruluyor.")
            sys.exit(result.returncode)

        print(f"  Tamamlandi — {fmt_duration(elapsed)}")

    if not args.dry_run:
        total_elapsed = time.monotonic() - pipeline_start
        print("\n" + "=" * 60)
        print(f"Pipeline tamamlandi. Toplam sure: {fmt_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
