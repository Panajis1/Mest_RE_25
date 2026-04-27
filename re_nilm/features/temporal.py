"""Calendar and dynamic temporal features for disaggregation models.

Replaces the identical _add_calendar_and_dynamic_features() that existed in both
ac_disaggregation.py and hp_model/disaggregation_functions.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, time_col: str = "dt_utc") -> pd.DataFrame:
    """Add sin/cos encoded hour, day-of-week, and month columns.

    Encoding: sin_x = sin(2π × x / period), cos_x = cos(2π × x / period).
    This creates smooth cyclic representations without ordinal artifacts.
    """
    out = df.copy()
    dt = pd.to_datetime(out[time_col])

    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)
    return out


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add lag columns for a given column.

    Args:
        df: DataFrame sorted by time.
        col: Column name to lag.
        lags: List of lag steps (in rows). Default: [1, 4, 8, 96].
    """
    if lags is None:
        lags = [1, 4, 8, 96]
    out = df.copy()
    for lag in lags:
        out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling mean and std columns for a given column.

    Args:
        df: DataFrame sorted by time.
        col: Column name to compute rolling statistics on.
        windows: List of window sizes (in rows). Default: [4, 8, 96].
    """
    if windows is None:
        windows = [4, 8, 96]
    out = df.copy()
    for w in windows:
        out[f"{col}_roll{w}_mean"] = out[col].rolling(w, min_periods=1).mean()
        out[f"{col}_roll{w}_std"] = out[col].rolling(w, min_periods=1).std()
    return out
