"""Daily PV feature and indicator functions.

Copied from legacy model/pv_detection.py to make re_nilm self-contained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_weather(avg_meteo_15min: pd.DataFrame) -> pd.DataFrame:
    """Compute daily weather features from 15-min weather time series."""
    if avg_meteo_15min.empty:
        return pd.DataFrame()

    meteo = avg_meteo_15min.sort_index().reset_index().rename(
        columns={"index": "timestamp"}
    )
    meteo["date"] = meteo["timestamp"].dt.date
    meteo["hour"] = meteo["timestamp"].dt.hour
    meteo["is_midday"] = (meteo["hour"] >= 10) & (meteo["hour"] < 16)

    daily_weather = (
        meteo.groupby("date")["global_rad_W"].sum().to_frame("G_daily")
    )
    midday_weather = (
        meteo.loc[meteo["is_midday"]]
        .groupby("date")["global_rad_W"]
        .sum()
        .to_frame("G_midday")
    )
    daily_weather = daily_weather.join(midday_weather, how="left")

    daily_weather_index = pd.to_datetime(daily_weather.index)
    daily_weather = daily_weather.copy()
    daily_weather["month"] = daily_weather_index.to_period("M")

    def month_low_q(x):
        return x.quantile(0.2)

    def month_high_q(x):
        return x.quantile(0.8)

    low_q = daily_weather.groupby("month")["G_midday"].transform(month_low_q)
    high_q = daily_weather.groupby("month")["G_midday"].transform(month_high_q)

    def bucket_row(row, lq, hq):
        g_mid = row["G_midday"]
        if pd.isna(g_mid):
            return "unknown"
        if g_mid <= lq:
            return "low"
        if g_mid >= hq:
            return "high"
        return "medium"

    daily_weather["rad_bucket"] = [
        bucket_row(row, lq, hq)
        for (_, row), lq, hq in zip(
            daily_weather.iterrows(), low_q, high_q
        )
    ]

    daily_weather = daily_weather.reset_index().rename(columns={"index": "date"})
    return daily_weather


def build_customer_daily_features(
    re_data_with_meteo: pd.DataFrame,
    daily_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-customer daily and midday aggregates merged with weather."""
    if re_data_with_meteo.empty or daily_weather.empty:
        return pd.DataFrame()

    df = re_data_with_meteo.copy()
    df["date"] = df["DT_UTC"].dt.date
    df["hour"] = df["DT_UTC"].dt.hour

    midday_mask = (df["hour"] >= 10) & (df["hour"] < 16)

    daily_cust = (
        df.groupby(["ID", "date"])
        .agg(
            Prod_daily=("PROD_KWH", "sum"),
            Conso_daily=("CONSO_KWH", "sum"),
        )
        .reset_index()
    )
    daily_cust["Net_daily"] = daily_cust["Conso_daily"] - daily_cust["Prod_daily"]

    midday_cust = (
        df.loc[midday_mask]
        .groupby(["ID", "date"])
        .agg(
            Prod_midday=("PROD_KWH", "sum"),
            Conso_midday=("CONSO_KWH", "sum"),
        )
        .reset_index()
    )
    midday_cust["Net_midday"] = (
        midday_cust["Conso_midday"] - midday_cust["Prod_midday"]
    )

    daily_cust = daily_cust.merge(
        midday_cust, on=["ID", "date"], how="left"
    )

    daily_features = daily_cust.merge(
        daily_weather, on="date", how="left"
    )

    return daily_features


def compute_pv_indicators(
    daily_features: pd.DataFrame,
    re_data_with_meteo: pd.DataFrame,
) -> pd.DataFrame:
    """Compute customer-level PV indicators from daily and 15-min features."""
    if daily_features.empty or re_data_with_meteo.empty:
        return pd.DataFrame()

    hi = daily_features[daily_features["rad_bucket"] == "high"]
    lo = daily_features[daily_features["rad_bucket"] == "low"]

    hi_midday = (
        hi.groupby("ID")[["Prod_midday", "Net_midday"]].mean()
        .rename(
            columns={
                "Prod_midday": "Prod_midday_high",
                "Net_midday": "Net_midday_high",
            }
        )
    )
    lo_midday = (
        lo.groupby("ID")[["Prod_midday", "Net_midday"]].mean()
        .rename(
            columns={
                "Prod_midday": "Prod_midday_low",
                "Net_midday": "Net_midday_low",
            }
        )
    )

    delta = hi_midday.join(lo_midday, how="outer")
    delta["DeltaProd"] = (
        delta["Prod_midday_high"] - delta["Prod_midday_low"]
    )
    delta["DeltaNet"] = (
        delta["Net_midday_high"] - delta["Net_midday_low"]
    )

    def corr_and_beta(group: pd.DataFrame):
        g = group.dropna(subset=["global_rad_W"])
        if g["global_rad_W"].var() == 0 or g["PROD_KWH"].var() == 0:
            return pd.Series({"corr_prod_rad": np.nan, "beta_regression": 0.0})
        corr = g["PROD_KWH"].corr(g["global_rad_W"])
        x = g["global_rad_W"].to_numpy()
        y = g["PROD_KWH"].to_numpy()
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            beta = 0.0
        else:
            beta = ((x - x_mean) * (y - y_mean)).sum() / denom
        return pd.Series(
            {
                "corr_prod_rad": corr,
                "beta_regression": beta,
            }
        )

    try:
        corr_beta = (
            re_data_with_meteo.groupby("ID", group_keys=False)
            .apply(corr_and_beta, include_groups=False)
            .reset_index()
            .set_index("ID")
        )
    except TypeError:
        corr_beta = (
            re_data_with_meteo.groupby("ID", group_keys=False)
            .apply(corr_and_beta)
            .reset_index()
            .set_index("ID")
        )

    yearly = (
        re_data_with_meteo.groupby("ID")
        .agg(
            yearly_prod=("PROD_KWH", "sum"),
            yearly_cons=("CONSO_KWH", "sum"),
        )
    )

    indicators = delta.join(corr_beta, how="outer").join(yearly, how="outer")
    indicators = indicators.reset_index().rename(columns={"ID": "customer_id"})
    return indicators


def classify_pv_customers(
    pv_indicators: pd.DataFrame,
    corr_threshold: float = 0.3,
    delta_net_threshold: float = -0.1,
    min_yearly_prod: float = 1.0,
) -> pd.DataFrame:
    """Classify customers as PV/non-PV with a probability-like score."""
    if pv_indicators.empty:
        return pv_indicators

    df = pv_indicators.copy()

    has_pv = (
        (df["yearly_prod"].fillna(0.0) > min_yearly_prod)
        | (df["corr_prod_rad"].fillna(0.0) > corr_threshold)
        | (df["DeltaProd"].fillna(0.0) > 0.01)
        | (df["DeltaNet"].fillna(0.0) < delta_net_threshold)
    )

    df["has_pv"] = has_pv
    corr = df["corr_prod_rad"].fillna(0.0)
    df["has_pv_prob"] = np.clip((corr - 0.1) / 0.4, 0.0, 1.0)
    df.loc[~df["has_pv"], "has_pv_prob"] = 0.0
    return df
