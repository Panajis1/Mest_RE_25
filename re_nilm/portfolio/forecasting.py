"""PV generation forecast and net consumption calculation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_PV_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_PV_DIR) not in sys.path:
    sys.path.insert(0, str(_PV_DIR))

try:
    from pv_detection import forecast_pv_for_customers_streaming as _forecast_streaming
    _PV_AVAILABLE = True
except ImportError:
    _PV_AVAILABLE = False


def predict_customer_pv_15min(
    customer_ts: pd.DataFrame,
    pv_capacity_kwp: float,
    weather_df: pd.DataFrame,
    efficiency: float = 0.15,
) -> pd.DataFrame:
    """Predict 15-min PV generation for a single customer.

    Uses a simple irradiance-based model:
        pv_kw = capacity_kwp × (global_rad_W / 1000) × efficiency

    Args:
        customer_ts: Customer time series with [DT_UTC, CONSO_KWH, PROD_KWH].
        pv_capacity_kwp: Installed PV capacity in kWp.
        weather_df: Weather data with [dt_utc, global_rad_W].
        efficiency: Panel efficiency factor (default 0.15 = 15%).

    Returns:
        DataFrame with [dt_utc, customer_id, pv_forecast_kwh_15min, net_consumption_kwh_15min].
    """
    df = customer_ts.copy()
    df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
    df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")

    customer_id = str(df["ID"].iloc[0]) if "ID" in df.columns else "unknown"

    # Normalize to datetime64[us] — pandas 2.x merge_asof requires identical units
    df["DT_UTC"] = df["DT_UTC"].astype("datetime64[us]")
    wdf = weather_df.copy()
    wdf["dt_utc"] = wdf["dt_utc"].astype("datetime64[us]")
    wdf = wdf.sort_values("dt_utc")
    merged = pd.merge_asof(
        df.rename(columns={"DT_UTC": "dt_utc"}),
        wdf[["dt_utc", "global_rad_W"]],
        on="dt_utc",
        direction="backward",
        tolerance=pd.Timedelta("1h"),
    )

    # Power (kW) × 0.25 h = kWh per 15-min interval
    pv_kw = (merged["global_rad_W"].clip(lower=0) / 1000.0) * pv_capacity_kwp * efficiency
    pv_kwh = pv_kw * 0.25
    net_kwh = merged["CONSO_KWH"] - pv_kwh.clip(lower=0)

    return pd.DataFrame({
        "dt_utc": merged["dt_utc"],
        "customer_id": customer_id,
        "pv_forecast_kwh_15min": pv_kwh.clip(lower=0),
        "net_consumption_kwh_15min": net_kwh,
    })


def forecast_pv_for_customers(
    results: pd.DataFrame,
    customer_index: dict,
    weather_df: pd.DataFrame,
    efficiency: float = 0.15,
    min_capacity_kwp: float = 0.1,
) -> pd.DataFrame:
    """Predict 15-min PV generation for all PV-positive customers.

    Args:
        results: Per-customer results with [customer_id, pv_capacity_kwp].
        customer_index: Dict {customer_id → [parquet file paths]}.
        weather_df: Weather data [dt_utc, global_rad_W].
        efficiency: Panel efficiency factor.
        min_capacity_kwp: Skip customers below this capacity threshold.

    Returns:
        Concatenated DataFrame with [dt_utc, customer_id, pv_forecast_kwh_15min,
        net_consumption_kwh_15min] for all PV customers.
    """
    from re_nilm.pipeline.customer_index import load_customer_from_index

    pv_customers = results[
        results["pv_capacity_kwp"].notna() & (results["pv_capacity_kwp"] >= min_capacity_kwp)
    ]

    frames: List[pd.DataFrame] = []
    for _, row in pv_customers.iterrows():
        cid = str(row["customer_id"])
        cap = float(row["pv_capacity_kwp"])
        df = load_customer_from_index(cid, customer_index)
        if df.empty:
            continue
        try:
            fc = predict_customer_pv_15min(df, cap, weather_df, efficiency)
            frames.append(fc)
        except Exception:
            continue

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_net_consumption(
    pv_forecast: pd.DataFrame,
    aggregate_by: str = "dt_utc",
) -> pd.DataFrame:
    """Aggregate 15-min net consumption (CONSO - PV) across all PV customers.

    Args:
        pv_forecast: Output of forecast_pv_for_customers.
        aggregate_by: Column to group by (default: dt_utc for portfolio timeseries).

    Returns:
        DataFrame with [dt_utc, total_pv_kwh, total_net_consumption_kwh, n_customers].
    """
    if pv_forecast.empty:
        return pd.DataFrame()

    agg = (
        pv_forecast.groupby(aggregate_by)
        .agg(
            total_pv_kwh=("pv_forecast_kwh_15min", "sum"),
            total_net_consumption_kwh=("net_consumption_kwh_15min", "sum"),
            n_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    return agg
