from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


DEFAULT_K_CANDIDATES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]



def resolve_k(
    n_assets: int,
    mode: str = "global",
    global_k: int = 5,
) -> int:
    """
    Resolve the number of clusters to use.

    Parameters
    ----------
    n_assets : int
        Number of assets; the resolved cluster count cannot exceed this value.
    mode : str
        Cluster-count selection mode. Currently only 'global' is supported.
    global_k : int
        Fixed cluster count used when mode='global'.

    Returns
    -------
    int
        Cluster count to use.

    Raises
    ------
    ValueError
        If n_assets <= 0 or global_k < 1.
    NotImplementedError
        If mode is anything other than 'global'.
    """
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    if mode != "global":
        raise NotImplementedError(
            f"k selection mode '{mode}' is not implemented yet. Use mode='global'."
        )

    if global_k < 1:
        raise ValueError("global_k must be >= 1")

    if global_k > n_assets:
        warnings.warn(
            f"global_k={global_k} is larger than n_assets={n_assets}. "
            f"Using k={n_assets}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return n_assets

    return global_k


def select_non_overlapping_windows(
    returns_index: pd.Index,
    window_length: int = 756,
    n_windows: int = 5,
) -> pd.DataFrame:
    """
    Select non-overlapping evaluation windows distributed evenly across the full sample.

    Windows are spaced systematically by distributing the available slack
    (n_obs - n_windows * window_length) uniformly across the chosen positions.

    Parameters
    ----------
    returns_index : pd.Index
        Full date index of the returns series.
    window_length : int
        Number of observations in each window.
    n_windows : int
        Number of non-overlapping windows to select.

    Returns
    -------
    pd.DataFrame
        One row per window with columns 'window_id', 'start_iloc', 'end_iloc',
        'start_date', 'end_date', and 'n_obs'.

    Raises
    ------
    ValueError
        If n_windows < 1, window_length < 1, or there are not enough
        observations to fit all requested windows.
    """
    n_obs = len(returns_index)
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    required_obs = n_windows * window_length
    if n_obs < required_obs:
        raise ValueError(
            f"Not enough observations for {n_windows} non-overlapping windows "
            f"of length {window_length}. Have {n_obs}, need {required_obs}."
        )

    slack = n_obs - required_obs
    slack_positions = np.linspace(0, slack, n_windows)

    starts = []
    for i, slack_pos in enumerate(slack_positions):
        start = int(round(i * window_length + float(slack_pos)))
        starts.append(start)

    # Guard against any rounding artifacts.
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1] + window_length)
    starts[-1] = min(starts[-1], n_obs - window_length)
    for i in range(len(starts) - 2, -1, -1):
        starts[i] = min(starts[i], starts[i + 1] - window_length)

    rows: list[dict] = []
    for i, start in enumerate(starts, start=1):
        end = start + window_length - 1
        rows.append(
            {
                "window_id": f"W{i}",
                "start_iloc": int(start),
                "end_iloc": int(end),
                "start_date": returns_index[start],
                "end_date": returns_index[end],
                "n_obs": int(window_length),
            }
        )

    windows_df = pd.DataFrame(rows)
    return windows_df
