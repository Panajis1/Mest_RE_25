"""Timezone handling and weather-to-meter alignment utilities."""

from __future__ import annotations

import pandas as pd


def normalize_ts(series: pd.Series) -> pd.Series:
    """Cast a datetime series to datetime64[us], tz-naive UTC.

    pandas 2.x preserves parquet timestamp precision (ms, us, ns) and requires
    exact dtype matches in merge_asof. This normalises both sides to the same
    unit before any temporal join.
    """
    s = pd.to_datetime(series, errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[us]")


def to_utc_naive(df: pd.DataFrame, col: str = "DT_UTC") -> pd.DataFrame:
    """Strip timezone from a datetime column, keeping values as UTC wall-clock."""
    series = pd.to_datetime(df[col], errors="coerce")
    if series.dt.tz is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df[col] = series
    return df


def align_weather_to_meter(
    meter_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    meter_time_col: str = "DT_UTC",
    weather_time_col: str = "dt_utc",
) -> pd.DataFrame:
    """Merge weather onto meter DataFrame using backward nearest-neighbour join.

    Both DataFrames must have tz-naive UTC timestamps. The meter column is used
    as the left key; the weather column is the right key.

    Returns:
        meter_df with weather columns appended; rows with no weather match are
        kept but weather values are NaN.
    """
    meter = meter_df.copy()
    weather = weather_df.copy()
    # Normalize to datetime64[us] — pandas 2.x merge_asof requires identical units
    meter[meter_time_col] = pd.to_datetime(meter[meter_time_col]).astype("datetime64[us]")
    weather[weather_time_col] = pd.to_datetime(weather[weather_time_col]).astype("datetime64[us]")
    meter = meter.sort_values(meter_time_col)
    weather = weather.sort_values(weather_time_col)

    merged = pd.merge_asof(
        meter,
        weather.rename(columns={weather_time_col: meter_time_col}),
        on=meter_time_col,
        direction="backward",
        tolerance=pd.Timedelta("1h"),
    )
    return merged


def daily_weather_from_15min(weather_df: pd.DataFrame, time_col: str = "dt_utc") -> pd.DataFrame:
    """Aggregate 15-min weather to daily statistics used by feature extractors.

    Returns DataFrame with columns [date, G_daily_kWh_m2, G_midday_W, t_mean_C, t_max_C, t_min_C].
    """
    df = weather_df.copy()
    df["_date"] = pd.to_datetime(df[time_col]).dt.date

    agg = df.groupby("_date").agg(
        G_daily_kWh_m2=("global_rad_W", lambda x: x.mean() * 24 / 1000),
        G_midday_W=("global_rad_W", "max"),
        t_mean_C=("t_2m_C", "mean"),
        t_max_C=("t_2m_C", "max"),
        t_min_C=("t_2m_C", "min"),
    ).reset_index().rename(columns={"_date": "date"})

    return agg
