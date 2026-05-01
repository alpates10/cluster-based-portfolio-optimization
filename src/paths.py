from __future__ import annotations

from pathlib import Path

UNIVERSE_CHOICES = ["sp500", "bist100", "nifty50", "sse"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_processed_dir(universe: str) -> Path:
    return _PROJECT_ROOT / "data" / "processed" / universe


def get_returns_path(universe: str) -> Path:
    return get_processed_dir(universe) / "returns_final.csv"


def get_dtw_cache_dir(universe: str) -> str:
    return str(get_processed_dir(universe) / "dtw_cache")


def get_backtests_dir(universe: str) -> Path:
    return get_processed_dir(universe) / "backtests"


def get_clustering_dir(universe: str) -> Path:
    return get_processed_dir(universe) / "clustering"


def get_clustering_analysis_dir(universe: str) -> Path:
    return get_processed_dir(universe) / "clustering_analysis"


def get_raw_dir(universe: str) -> Path:
    return _PROJECT_ROOT / "data" / "raw" / f"{universe}_stocks"
