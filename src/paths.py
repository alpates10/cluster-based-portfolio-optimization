from __future__ import annotations

from pathlib import Path

UNIVERSE_CHOICES = ["sp500", "bist100", "nifty50", "sse"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_processed_dir(universe: str) -> Path:
    """Return the processed-data directory for a universe.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    Path
        Absolute path to data/processed/<universe>/.
    """
    return _PROJECT_ROOT / "data" / "processed" / universe


def get_returns_path(universe: str) -> Path:
    """Return the processed returns CSV path for a universe.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    Path
        Absolute path to data/processed/<universe>/returns_final.csv.
    """
    return get_processed_dir(universe) / "returns_final.csv"


def get_dtw_cache_dir(universe: str) -> str:
    """Return the DTW cache directory path for a universe as a string.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    str
        Absolute path string to data/processed/<universe>/dtw_cache/.
    """
    return str(get_processed_dir(universe) / "dtw_cache")


def get_backtests_dir(universe: str) -> Path:
    """Return the backtest output directory for a universe.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    Path
        Absolute path to data/processed/<universe>/backtests/.
    """
    return get_processed_dir(universe) / "backtests"


def get_clustering_k_selection_dir(universe: str) -> Path:
    """Return the clustering k-selection output directory for a universe.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    Path
        Absolute path to data/processed/<universe>/clustering_k_selection/.
    """
    return get_processed_dir(universe) / "clustering_k_selection"


def get_clustering_visualization_dir(universe: str) -> Path:
    """Return the clustering visualization output directory for a universe.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    Path
        Absolute path to data/processed/<universe>/clustering_visualization/.
    """
    return get_processed_dir(universe) / "clustering_visualization"


def get_raw_dir(universe: str) -> Path:
    """Return the raw stock-price CSV directory for a universe.

    Parameters
    ----------
    universe : str
        Universe identifier, e.g. 'sp500' or 'bist100'.

    Returns
    -------
    Path
        Absolute path to data/raw/<universe>_stocks/.
    """
    return _PROJECT_ROOT / "data" / "raw" / f"{universe}_stocks"
