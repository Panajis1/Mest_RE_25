"""Weather-derived features for detection and disaggregation models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_radiation(weather_15min: pd.DataFrame, time_col: str = "dt_utc") -> pd.DataFrame:
    """Aggregate 15-min radiation to daily statistics.

    Returns DataFrame with columns [date, G_daily_kWh_m2, G_midday_W, rad_bucket].

    rad_bucket categories:
      0 = dark    (G_midday < 100 W/m²)
      1 = cloudy  (100 ≤ G_midday < 400 W/m²)
      2 = sunny   (G_midday ≥ 400 W/m²)
    """
    df = weather_15min.copy()
    df["_date"] = pd.to_datetime(df[time_col]).dt.date
    agg = df.groupby("_date").agg(
        G_daily_kWh_m2=("global_rad_W", lambda x: x.mean() * 24 / 1000),
        G_midday_W=("global_rad_W", "max"),
    ).reset_index().rename(columns={"_date": "date"})

    agg["rad_bucket"] = pd.cut(
        agg["G_midday_W"],
        bins=[-np.inf, 100, 400, np.inf],
        labels=[0, 1, 2],
    ).astype(float)
    return agg


def compute_temperature_bands(
    df: pd.DataFrame,
    temp_col: str = "t_2m_C",
    hot_thresh: float = 25.0,
    cold_thresh: float = 10.0,
) -> pd.DataFrame:
    """Add boolean columns is_hot and is_cold based on temperature thresholds."""
    out = df.copy()
    out["is_hot"] = out[temp_col] > hot_thresh
    out["is_cold"] = out[temp_col] < cold_thresh
    return out


def cooling_degree_days(temp: pd.Series, base: float = 18.0) -> pd.Series:
    """Daily CDD = max(T_mean - base, 0). Input must be a daily mean temperature series."""
    return (temp - base).clip(lower=0)


def heating_degree_days(temp: pd.Series, base: float = 18.0) -> pd.Series:
    """Daily HDD = max(base - T_mean, 0). Input must be a daily mean temperature series."""
    return (base - temp).clip(lower=0)
