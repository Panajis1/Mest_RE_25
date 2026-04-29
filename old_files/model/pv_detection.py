#%%

import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

# Add the repository root to sys.path to allow absolute imports from 'data'
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import data.envdata as meteo
import data.load_smart_meter as re_data

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def write_plotly_figures_to_dir(
    figs: Dict[str, Any],
    out_dir: Union[str, Path],
    *,
    fmt: str = "png",
    scale: float = 2.0,
) -> list:
    """
    Write Plotly figures to *out_dir* as static images (requires the
    ``kaleido`` package: ``pip install kaleido``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list = []
    for name, fig in figs.items():
        if fig is None:
            continue
        path = out / f"{name}.{fmt}"
        try:
            fig.write_image(str(path), scale=scale)
        except Exception as e:
            raise RuntimeError(
                "Plotly static image export failed. Install kaleido in your environment "
                "(e.g. pip install 'kaleido>=0.2,<1')."
            ) from e
        written.append(str(path))
    return written


#%%
def load_meteo_data():
    """
    Load MeteoSwiss data and return:
    - combined_meteo: stacked per-station records (UTC-aware timestamp column)
    - avg_meteo_15min: regional time series at 15-min resolution (tz-naive
      DatetimeIndex, ready for merge with RE parquet data)

    env_data() already resamples to 15-min and strips the timezone, so we
    only need to ensure the index type is correct.
    """
    combined_meteo, avg_meteo_15min = meteo.env_data()
    if avg_meteo_15min.empty:
        return combined_meteo, avg_meteo_15min

    if not isinstance(avg_meteo_15min.index, pd.DatetimeIndex):
        avg_meteo_15min = avg_meteo_15min.copy()
        avg_meteo_15min.index = pd.to_datetime(avg_meteo_15min.index)

    avg_meteo_15min = avg_meteo_15min.sort_index()
    return combined_meteo, avg_meteo_15min


def load_re_data(max_cons_kwh: float = 10_000):
    """
    Load Romande Energie smart meter data (all customers in parquet files)
    and return a filtered subset whose **total** ``CONSO_KWH`` over the
    available history is <= *max_cons_kwh* (kWh, not kW).

    For large datasets prefer :func:`stream_customer_summary` plus metadata
    filters (e.g. :func:`particuliers_customer_ids`) instead of loading all rows.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data" / "re_data" / "ETHZ"
    re_data_gen = re_data.load_all_data_generator(str(data_dir))
    re_data_df = pd.concat(re_data_gen, ignore_index=True)
    # Ensure timestamps are UTC, then strip tz so they are naive-UTC
    # (matching the tz-naive meteo index from env_data)
    dt = pd.to_datetime(re_data_df["DT_UTC"], utc=True)
    re_data_df["DT_UTC"] = dt.dt.tz_convert(None)

    customer_summary = (
        re_data_df.groupby("ID").agg(
            sum_cons_kwh=pd.NamedAgg(column="CONSO_KWH", aggfunc="sum"),
            sum_prod_kwh=pd.NamedAgg(column="PROD_KWH", aggfunc="sum"),
            dt_start=pd.NamedAgg(column="DT_UTC", aggfunc="min"),
            dt_end=pd.NamedAgg(column="DT_UTC", aggfunc="max"),
        )
        .sort_values(by="sum_cons_kwh", ascending=False)
        .reset_index()
    )

    ids = customer_summary[customer_summary["sum_cons_kwh"] <= max_cons_kwh][["ID"]]
    re_data_df_small = re_data_df[re_data_df["ID"].isin(ids["ID"])]
    return re_data_df, re_data_df_small, customer_summary


# ---------------------------------------------------------------------------
# Streaming helpers (shared by stream_* functions)
# ---------------------------------------------------------------------------

def _build_meteo_merge_frame(avg_meteo_15min: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["global_rad_W"]
    if "t_2m_C" in avg_meteo_15min.columns:
        merge_cols.append("t_2m_C")
    return (
        avg_meteo_15min[merge_cols]
        .rename_axis("DT_UTC")
        .reset_index()
    )


def _prepare_raw_file(
    fpath: str,
    meteo_for_merge: pd.DataFrame,
    target_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Load a single parquet file, convert timestamps, merge meteo, downcast."""
    df = pd.read_parquet(fpath)
    if df.empty:
        return df
    df["ID"] = df["ID"].astype(str)
    if target_ids is not None:
        df = df[df["ID"].isin(target_ids)]
        if df.empty:
            return df
    dt = pd.to_datetime(df["DT_UTC"], utc=True)
    df["DT_UTC"] = dt.dt.tz_convert(None)
    for col in ("CONSO_KWH", "PROD_KWH"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df = df.merge(meteo_for_merge, on="DT_UTC", how="left")
    return df


# ---------------------------------------------------------------------------
# Streaming customer summary (replaces load_re_data for large datasets)
# ---------------------------------------------------------------------------

def stream_customer_summary(
    data_dir: Optional[str] = None,
    max_cons_kwh: float = 10_000,
    allowed_ids: Optional[Set[str]] = None,
) -> tuple:
    """
    Scan all parquet files to compute per-customer summary statistics without
    ever holding more than one file in memory.

    ``sum_cons_kwh`` is the **sum of** ``CONSO_KWH`` **over all timestamps
    present** for that customer (full meter history in the files), not a
    calendar-year normalization. *max_cons_kwh* is in **kWh**.

    Parameters
    ----------
    allowed_ids
        If given, only these customer IDs are aggregated and returned; use
        with metadata (e.g. :func:`particuliers_customer_ids`) before applying
        the consumption cap.

    Returns
    -------
    customer_summary : DataFrame
        One row per customer with sum_cons_kwh, sum_prod_kwh, dt_start, dt_end.
    target_ids : set[str]
        Customer IDs with ``sum_cons_kwh`` <= *max_cons_kwh* (among *allowed_ids*
        when that set is provided).
    """
    if data_dir is None:
        data_dir = str(
            Path(__file__).resolve().parent.parent / "data" / "re_data" / "ETHZ"
        )

    if allowed_ids is not None and not allowed_ids:
        return pd.DataFrame(), set()

    files = re_data.get_parquet_files(data_dir)
    partial_summaries: list[pd.DataFrame] = []

    _iter = tqdm(files, desc="Scanning files") if _HAS_TQDM else files
    for fpath in _iter:
        try:
            df = pd.read_parquet(
                fpath, columns=["ID", "CONSO_KWH", "PROD_KWH", "DT_UTC"]
            )
        except Exception:
            continue
        if df.empty:
            continue
        df["ID"] = df["ID"].astype(str)
        if allowed_ids is not None:
            df = df.loc[df["ID"].isin(allowed_ids)]
            if df.empty:
                continue
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"])
        partial = df.groupby("ID").agg(
            sum_cons_kwh=pd.NamedAgg(column="CONSO_KWH", aggfunc="sum"),
            sum_prod_kwh=pd.NamedAgg(column="PROD_KWH", aggfunc="sum"),
            dt_start=pd.NamedAgg(column="DT_UTC", aggfunc="min"),
            dt_end=pd.NamedAgg(column="DT_UTC", aggfunc="max"),
        )
        partial_summaries.append(partial)
        del df

    if not partial_summaries:
        return pd.DataFrame(), set()

    combined = pd.concat(partial_summaries)
    customer_summary = (
        combined.groupby(level=0)
        .agg({
            "sum_cons_kwh": "sum",
            "sum_prod_kwh": "sum",
            "dt_start": "min",
            "dt_end": "max",
        })
        .sort_values("sum_cons_kwh", ascending=False)
        .reset_index()
        .rename(columns={"index": "ID"})
    )
    if "ID" not in customer_summary.columns:
        customer_summary = customer_summary.rename(
            columns={customer_summary.columns[0]: "ID"}
        )

    if allowed_ids is not None:
        customer_summary = customer_summary.loc[
            customer_summary["ID"].isin(allowed_ids)
        ].copy()

    target_ids = set(
        customer_summary.loc[
            customer_summary["sum_cons_kwh"] <= max_cons_kwh, "ID"
        ].astype(str)
    )
    return customer_summary, target_ids


def align_meteo_with_re_data(avg_meteo_15min: pd.DataFrame, re_data_df_small: pd.DataFrame):
    """
    Restrict meteorological data to the time span covered by the filtered RE data
    and merge global radiation onto the 15-min smart-meter records.
    """
    if avg_meteo_15min.empty or re_data_df_small.empty:
        return avg_meteo_15min, re_data_df_small

    first_dt = re_data_df_small["DT_UTC"].min()
    last_dt = re_data_df_small["DT_UTC"].max()

    # Restrict meteo series to overlap with RE data window
    meteo_window = avg_meteo_15min.loc[first_dt:last_dt].copy()

    # Merge global radiation and temperature onto smart-meter time series
    merge_cols = ["global_rad_W"]
    if "t_2m_C" in meteo_window.columns:
        merge_cols.append("t_2m_C")
    meteo_for_merge = (
        meteo_window[merge_cols]
        .rename_axis("DT_UTC")
        .reset_index()
    )

    re_data_with_meteo = re_data_df_small.merge(
        meteo_for_merge, on="DT_UTC", how="left"
    )

    return meteo_window, re_data_with_meteo


def compute_daily_weather(avg_meteo_15min: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily weather features (G_daily, G_midday, rad_bucket) from meteo
    data alone.  No customer data is needed -- call once and reuse across all
    streaming batches.
    """
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
    """
    Compute per-customer daily / midday aggregates and merge with pre-computed
    daily_weather (which supplies rad_bucket).
    """
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


def build_daily_features(
    avg_meteo_15min: pd.DataFrame, re_data_with_meteo: pd.DataFrame
):
    """
    Backward-compatible wrapper: compute daily weather + customer features.
    Prefer calling compute_daily_weather() and build_customer_daily_features()
    separately in streaming pipelines to avoid recomputing weather each time.
    """
    daily_weather = compute_daily_weather(avg_meteo_15min)
    daily_features = build_customer_daily_features(re_data_with_meteo, daily_weather)
    return daily_features, daily_weather


def compute_pv_indicators(
    daily_features: pd.DataFrame, re_data_with_meteo: pd.DataFrame
):
    """
    Compute per-customer PV indicators:
    - DeltaProd, DeltaNet (high vs low radiation days, midday)
    - corr_prod_rad (15-min correlation between PROD_KWH and global_rad_W)
    - beta_regression (slope of production vs radiation at 15-min resolution)
    """
    if daily_features.empty or re_data_with_meteo.empty:
        return pd.DataFrame()

    # High vs low radiation days
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

    # 15-min correlation and regression slope per customer
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
        # pandas < 2.2 does not have include_groups
        corr_beta = (
            re_data_with_meteo.groupby("ID", group_keys=False)
            .apply(corr_and_beta)
            .reset_index()
            .set_index("ID")
        )

    # Long-term aggregates
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
    """
    Add a simple PV / non-PV classification and probability-like score.
    """
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
    # Simple probability-like score based on correlation strength
    corr = df["corr_prod_rad"].fillna(0.0)
    df["has_pv_prob"] = np.clip((corr - 0.1) / 0.4, 0.0, 1.0)
    df.loc[~df["has_pv"], "has_pv_prob"] = 0.0

    return df


def plot_customer_timeseries(
    customer_id: str,
    re_data_with_meteo: pd.DataFrame,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    show: bool = True,
):
    """
    Plot time series of CONSO_KWH, PROD_KWH, and global_rad_W for a single customer.
    """
    df = re_data_with_meteo[re_data_with_meteo["ID"] == customer_id].copy()
    if df.empty:
        raise ValueError(f"No data found for customer {customer_id}")

    if start is not None:
        start_ts = pd.to_datetime(start)
        df = df[df["DT_UTC"] >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        df = df[df["DT_UTC"] <= end_ts]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["DT_UTC"],
            y=df["CONSO_KWH"],
            name="CONSO_KWH (import)",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["DT_UTC"],
            y=df["PROD_KWH"],
            name="PROD_KWH (export)",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["DT_UTC"],
            y=df["global_rad_W"],
            name="Global radiation (W/m²)",
            mode="lines",
            yaxis="y2",
            opacity=0.6,
        )
    )

    fig.update_layout(
        title=f"Customer {customer_id} – imports, exports, and radiation",
        xaxis_title="Time (UTC)",
        yaxis=dict(title="Energy (kWh per 15 min)"),
        yaxis2=dict(
            title="Global radiation (W/m²)",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    if show:
        fig.show()
    return fig


def plot_customer_capacity_validation(
    customer_id: str,
    re_data_with_meteo: pd.DataFrame,
    prob_summary: pd.DataFrame,
    show: bool = True
) -> go.Figure:
    """
    Plots a customer's Import/Export Power (kW) during their peak production week,
    overlaid with their statistically estimated PV Capacity (kWp) and 95% CI.
    """
    # 1. Fetch the estimated capacity metrics from your summary table
    cust_summary = prob_summary[prob_summary["customer_id"] == customer_id]
    if cust_summary.empty:
        raise ValueError(f"Customer {customer_id} not found in the probability summary table.")
        
    cap_kwp = cust_summary["pv_capacity_kwp"].values[0]
    ci_lower = cust_summary["pv_capacity_kwp_ci_lower"].values[0]
    ci_upper = cust_summary["pv_capacity_kwp_ci_upper"].values[0]
    # Optional diagnostics if available
    cap_reg = cust_summary.get("pv_capacity_kwp_regression_only", pd.Series([np.nan])).values[0]
    cap_floor = cust_summary.get("pv_capacity_kwp_floor", pd.Series([np.nan])).values[0]

    # 2. Fetch and prepare the raw time-series data
    df = re_data_with_meteo[re_data_with_meteo["ID"] == customer_id].copy()
    if df.empty:
        raise ValueError(f"No time-series data found for customer {customer_id}.")

    # CRITICAL: Convert 15-min Energy (kWh) to Instantaneous Power (kW)
    # 1 kWh in a 15 min window = a steady flow of 4 kW of power.
    df["Import_kW"] = df["CONSO_KWH"] * 4
    df["Export_kW"] = df["PROD_KWH"] * 4

    # 3. Find the "Best" Week to plot (the week with the highest single day of export)
    df['date'] = df['DT_UTC'].dt.date
    daily_export = df.groupby('date')['Export_kW'].sum()
    if daily_export.sum() == 0:
        print(f"Warning: Customer {customer_id} has absolutely zero export.")
        best_day = df['DT_UTC'].max()
    else:
        best_day = pd.to_datetime(daily_export.idxmax())
    
    # Create a 7-day window centered roughly around their best production day
    start_date = best_day - pd.Timedelta(days=3)
    end_date = best_day + pd.Timedelta(days=4)
    plot_df = df[(df['DT_UTC'] >= start_date) & (df['DT_UTC'] < end_date)]

    # 4. Build the Plotly Figure
    fig = go.Figure()

    # Plot Grid Import (Consumption)
    fig.add_trace(
        go.Scatter(
            x=plot_df['DT_UTC'], 
            y=plot_df['Import_kW'], 
            mode='lines', 
            name='Grid Import (kW)', 
            line=dict(color='red', width=1.5),
            opacity=0.8
        )
    )

    # Plot Solar Export (Production)
    fig.add_trace(
        go.Scatter(
            x=plot_df['DT_UTC'], 
            y=plot_df['Export_kW'], 
            mode='lines', 
            name='Solar Export (kW)', 
            line=dict(color='royalblue', width=1.5),
            fill='tozeroy', # Light shading under the export curve
            fillcolor='rgba(65, 105, 225, 0.2)'
        )
    )

    # Draw the hybrid Estimated Capacity (kWp) as a horizontal dashed ceiling
    fig.add_hline(
        y=cap_kwp, 
        line_dash="dash", 
        line_color="green", 
        line_width=2,
        annotation_text=f" Hybrid PV Capacity: {cap_kwp:.2f} kWp ",
        annotation_position="top left",
        annotation_font=dict(color="green", size=12)
    )

    # Optional: overlay regression-only and floor capacities if present
    if pd.notna(cap_reg):
        fig.add_hline(
            y=cap_reg,
            line_dash="dot",
            line_color="darkgreen",
            line_width=1.5,
            annotation_text=f" Regression-only: {cap_reg:.2f} kWp ",
            annotation_position="top right",
            annotation_font=dict(color="darkgreen", size=11),
        )
    if pd.notna(cap_floor) and cap_floor > 0:
        fig.add_hline(
            y=cap_floor,
            line_dash="dash",
            line_color="orange",
            line_width=1.5,
            annotation_text=f" Physical floor: {cap_floor:.2f} kWp ",
            annotation_position="bottom left",
            annotation_font=dict(color="orange", size=11),
        )

    # Draw the 95% Confidence Interval as a shaded horizontal band
    if pd.notna(ci_lower) and pd.notna(ci_upper):
        fig.add_hrect(
            y0=ci_lower, 
            y1=ci_upper, 
            line_width=0, 
            fillcolor="green", 
            opacity=0.15,
            annotation_text=" 95% CI ",
            annotation_position="bottom left",
        )

    # Clean up the layout
    fig.update_layout(
        title=f"Power Profile vs. Estimated PV Capacity<br><sup>Customer: {customer_id} | Showing Peak Week</sup>",
        xaxis_title="Time (UTC)",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
        template="plotly_white"
    )

    if show:
        fig.show()
    

def plot_customer_high_low_profile(
    customer_id: str,
    re_data_with_meteo: pd.DataFrame,
    daily_features: pd.DataFrame,
    show: bool = True,
):
    """
    Plot average daily profiles (15-min resolution) on high vs low radiation days
    for a given customer.
    """
    df = re_data_with_meteo[re_data_with_meteo["ID"] == customer_id].copy()
    if df.empty:
        raise ValueError(f"No data found for customer {customer_id}")

    df["date"] = df["DT_UTC"].dt.date
    # Map radiation bucket onto each timestamp
    bucket_map = (
        daily_features[["date", "rad_bucket"]]
        .drop_duplicates()
        .set_index("date")["rad_bucket"]
    )
    df["rad_bucket"] = df["date"].map(bucket_map)
    df = df[df["rad_bucket"].isin(["high", "low"])]
    if df.empty:
        raise ValueError(f"No high/low radiation days for customer {customer_id}")

    # Time-of-day in minutes since midnight for ordering
    df["minutes"] = (
        df["DT_UTC"].dt.hour * 60 + df["DT_UTC"].dt.minute
    )

    prof = (
        df.groupby(["rad_bucket", "minutes"])[["CONSO_KWH", "PROD_KWH"]]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    for bucket, color in [("high", "red"), ("low", "blue")]:
        sub = prof[prof["rad_bucket"] == bucket]
        x_vals = sub["minutes"] / 60.0
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=sub["CONSO_KWH"],
                name=f"Import ({bucket})",
                mode="lines",
                line=dict(color=color, dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=sub["PROD_KWH"],
                name=f"Export ({bucket})",
                mode="lines",
                line=dict(color=color, dash="dot"),
            )
        )

    fig.update_layout(
        title=f"Customer {customer_id} – high vs low radiation daily profiles",
        xaxis_title="Hour of day",
        yaxis_title="Energy (kWh per 15 min)",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    if show:
        fig.show()
    return fig


def plot_population_statistics(
    pv_indicators: pd.DataFrame,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
):
    """
    Create population-level Plotly figures for PV indicators.
    """
    if pv_indicators.empty:
        return {}

    figs = {}

    figs["corr_hist"] = px.histogram(
        pv_indicators,
        x="corr_prod_rad",
        nbins=40,
        title="Distribution of corr(PROD_KWH, global_rad_W)",
    )
    if show:
        figs["corr_hist"].show()

    figs["beta_hist"] = px.histogram(
        pv_indicators,
        x="beta_regression",
        nbins=40,
        title="Distribution of slope(PROD_KWH vs global_rad_W)",
    )
    if show:
        figs["beta_hist"].show()

    if "has_pv" in pv_indicators.columns:
        color_col = "has_pv"
    else:
        color_col = None

    figs["beta_vs_yearly_prod"] = px.scatter(
        pv_indicators,
        x="beta_regression",
        y="yearly_prod",
        color=color_col,
        hover_data=["customer_id"],
        title="Slope(PROD vs radiation) vs yearly production",
    )
    if show:
        figs["beta_vs_yearly_prod"].show()

    figs["delta_scatter"] = px.scatter(
        pv_indicators,
        x="DeltaProd",
        y="DeltaNet",
        color=color_col,
        hover_data=["customer_id"],
        title="DeltaProd vs DeltaNet (high – low radiation days, midday)",
    )
    if show:
        figs["delta_scatter"].show()

    if save_dir:
        write_plotly_figures_to_dir(figs, save_dir)

    return figs


def plot_capacity_vs_production_with_ci(
    prob_summary: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Scatter: estimated PV capacity (kWp) vs yearly production (kWh) with 95% CI error bars.
    """
    df = prob_summary.merge(
        pv_indicators[["customer_id", "yearly_prod"]], on="customer_id", how="left"
    )
    df = df.copy()
    df["error_upper"] = df["pv_capacity_kwp_ci_upper"] - df["pv_capacity_kwp"]
    df["error_lower"] = df["pv_capacity_kwp"] - df["pv_capacity_kwp_ci_lower"]
    df["error_upper"] = df["error_upper"].clip(lower=0).fillna(0)
    df["error_lower"] = df["error_lower"].clip(lower=0).fillna(0)

    color_col = "floor_to_reg_ratio" if "floor_to_reg_ratio" in df.columns else None

    fig = px.scatter(
        df,
        x="pv_capacity_kwp",
        y="yearly_prod",
        error_x="error_upper",
        error_x_minus="error_lower",
        color=color_col,
        hover_data=["customer_id"],
        title="Estimated PV Capacity (kWp) vs. Yearly Production with 95% Confidence Intervals",
        labels={
            "pv_capacity_kwp": "Estimated PV Capacity (kWp)",
            "yearly_prod": "Yearly Production (kWh)",
            "floor_to_reg_ratio": "Floor / Regression Capacity",
        },
    )
    if save_dir:
        write_plotly_figures_to_dir({"capacity_vs_production_with_ci": fig}, save_dir)
    if show:
        fig.show()
    return fig


def plot_capacity_vs_self_consumption(
    prob_summary: pd.DataFrame,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Scatter: estimated PV capacity (kWp) vs self-consumption share (0–1).
    Y-axis is logarithmic; values at or below zero are clipped to epsilon for display.
    """
    _eps = 1e-6
    df = prob_summary.dropna(subset=["pv_capacity_kwp", "sc_share_mean"]).copy()
    df["sc_share_plot"] = df["sc_share_mean"].clip(lower=_eps)
    fig = px.scatter(
        df,
        x="pv_capacity_kwp",
        y="sc_share_plot",
        hover_data=["customer_id", "sc_share_mean"],
        title="System Size vs. Self-Consumption Share",
        labels={
            "pv_capacity_kwp": "Estimated PV Capacity (kWp)",
            "sc_share_plot": f"Self-consumption share (log scale, ≥{_eps:g})",
        },
    )
    fig.update_yaxes(
        type="log",
        range=[np.log10(_eps), np.log10(1.0)],
    )
    if save_dir:
        write_plotly_figures_to_dir({"capacity_vs_self_consumption": fig}, save_dir)
    if show:
        fig.show()
    return fig


def plot_customer_heatmap(
    customer_id: str,
    re_data_with_meteo: pd.DataFrame,
    show: bool = True,
) -> go.Figure:
    """
    Net load (import - export) heatmap by hour of day and month for one customer.
    Red = net import, blue = net export; zero at midpoint ("solar belly").
    """
    df = re_data_with_meteo[re_data_with_meteo["ID"] == customer_id].copy()
    if df.empty:
        raise ValueError(f"No data found for customer {customer_id}")

    df["Net_KWH"] = df["CONSO_KWH"] - df["PROD_KWH"]
    df["month"] = df["DT_UTC"].dt.month
    df["hour"] = df["DT_UTC"].dt.hour

    hourly_monthly_avg = (
        df.groupby(["month", "hour"])["Net_KWH"].mean().reset_index()
    )
    heatmap_data = hourly_monthly_avg.pivot(
        index="hour", columns="month", values="Net_KWH"
    )
    heatmap_data = heatmap_data.reindex(columns=range(1, 13))
    heatmap_data = heatmap_data.reindex(index=range(24))

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    fig = px.imshow(
        heatmap_data,
        labels=dict(
            x="Month of Year",
            y="Hour of Day",
            color="Net Load (kWh)",
        ),
        x=month_names,
        y=list(heatmap_data.index),
        title=f"Net Load Heatmap: Customer {customer_id}",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
    )
    fig.update_yaxes(autorange="reversed")
    if show:
        fig.show()
    return fig


def _fit_simple_slope(x: np.ndarray, y: np.ndarray):
    """
    Fit a simple linear regression y = alpha + beta x and return (beta, se_beta).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return 0.0, np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0, np.nan
    beta = ((x - x_mean) * (y - y_mean)).sum() / denom
    alpha = y_mean - beta * x_mean
    residuals = y - (alpha + beta * x)
    dof = max(n - 2, 1)
    sigma2 = (residuals ** 2).sum() / dof
    se_beta = float(np.sqrt(sigma2 / denom))
    return float(beta), se_beta


def _compute_demand_radiation_correction(df: pd.DataFrame) -> float:
    """
    Estimate the demand-radiation confound from nighttime demand-temperature
    sensitivity and midday radiation-temperature correlation.

    Returns beta_demand_rad: the portion of the Net_KWH regression slope
    attributable to demand response (positive means demand *decreases* with
    radiation, i.e. correction should be subtracted from -beta_net).
    """
    if "t_2m_C" not in df.columns:
        return 0.0

    night_mask = (df["hour"] >= 22) | (df["hour"] < 5)
    night = df[night_mask].dropna(subset=["t_2m_C", "CONSO_KWH"])
    if night.shape[0] < 30:
        return 0.0

    beta_demand_temp, _ = _fit_simple_slope(
        night["t_2m_C"].to_numpy(),
        night["CONSO_KWH"].to_numpy(),
    )

    midday_mask = (df["hour"] >= 10) & (df["hour"] < 16)
    midday = df[midday_mask].dropna(subset=["t_2m_C", "global_rad_W"])
    if midday.shape[0] < 30:
        return 0.0

    rad = midday["global_rad_W"].to_numpy()
    temp = midday["t_2m_C"].to_numpy()
    if rad.std() < 1e-6 or temp.std() < 1e-6:
        return 0.0

    beta_temp_vs_rad, _ = _fit_simple_slope(rad, temp)

    return float(beta_demand_temp * beta_temp_vs_rad)


def _capacity_and_sc_from_data(
    cust_df: pd.DataFrame, cust_days: pd.DataFrame
) -> dict:
    """
    Compute PV capacity proxy and self-consumption share for a single customer.

    v2 changes:
      - Floor: export-only (no sc_base_kW addition), 95th percentile, STC
        normalized via concurrent irradiance at near-peak export moments.
      - Combiner: regression-primary; floor serves only as a lower-bound clamp.
      - Optional demand-radiation correction from nighttime temperature model.
    """
    _empty = {
        "beta_net": 0.0,
        "beta_net_se": np.nan,
        "beta_export": 0.0,
        "beta_export_se": np.nan,
        "demand_rad_correction": 0.0,
        "pv_capacity_proxy": 0.0,
        "pv_capacity_regression_kwp": 0.0,
        "pv_capacity_floor_kwp": 0.0,
        "pv_capacity_hybrid_kwp": 0.0,
        "sc_share": np.nan,
    }
    if cust_df.empty or cust_days.empty:
        return _empty

    df = cust_df.dropna(subset=["global_rad_W"]).copy()
    if df.empty:
        return _empty

    df["date"] = df["DT_UTC"].dt.date
    df["hour"] = df["DT_UTC"].dt.hour
    if "rad_bucket" in cust_days.columns:
        bucket_map = (
            cust_days[["date", "rad_bucket"]]
            .dropna()
            .drop_duplicates()
            .set_index("date")["rad_bucket"]
        )
        df["rad_bucket"] = df["date"].map(bucket_map)
    else:
        df["rad_bucket"] = np.nan

    df["Net_KWH"] = df["CONSO_KWH"] - df["PROD_KWH"]

    # --- Regression capacity ---
    full_beta_net, full_beta_net_se = _fit_simple_slope(
        df["global_rad_W"].to_numpy(),
        df["Net_KWH"].to_numpy(),
    )

    midday_mask = (df["hour"] >= 10) & (df["hour"] < 16)
    sunny_mask = df["rad_bucket"] == "high"
    reg_df = df[midday_mask & sunny_mask].copy()
    if reg_df.empty:
        reg_beta_net, reg_beta_net_se = full_beta_net, full_beta_net_se
    else:
        reg_beta_net, reg_beta_net_se = _fit_simple_slope(
            reg_df["global_rad_W"].to_numpy(),
            reg_df["Net_KWH"].to_numpy(),
        )

    beta_net = reg_beta_net
    beta_net_se = reg_beta_net_se

    # Export slope vs radiation
    df_export = df[df["PROD_KWH"] > 0]
    if df_export.empty:
        beta_export, beta_export_se = 0.0, np.nan
    else:
        beta_export, beta_export_se = _fit_simple_slope(
            df_export["global_rad_W"].to_numpy(),
            df_export["PROD_KWH"].to_numpy(),
        )

    # Within-customer demand-radiation correction (section 2c of plan)
    demand_rad_correction = _compute_demand_radiation_correction(df)

    raw_pv_slope = max(0.0, -beta_net)
    corrected_pv_slope = max(0.0, raw_pv_slope - demand_rad_correction)
    regression_capacity_kwp = corrected_pv_slope * STC_FACTOR

    # --- Physical floor: export-only, no sc_base_kW (section 2a) ---
    export_kW = df["PROD_KWH"] * 4.0

    export_vals = export_kW.to_numpy()
    export_vals = export_vals[np.isfinite(export_vals)]
    if export_vals.size > 0:
        export_peak_kW = float(np.nanpercentile(export_vals, 95.0))
    else:
        export_peak_kW = 0.0

    # Reference irradiance at near-peak export moments
    if export_peak_kW > 0.0:
        peak_threshold = export_peak_kW * 0.9
        peak_mask = export_kW >= peak_threshold
        if peak_mask.any():
            g_ref_vals = df.loc[peak_mask, "global_rad_W"].to_numpy()
            g_ref_vals = g_ref_vals[np.isfinite(g_ref_vals)]
            if g_ref_vals.size > 0:
                g_ref = float(np.nanmean(g_ref_vals))
            else:
                g_ref = float(df["global_rad_W"].quantile(0.95))
        else:
            g_ref = float(df["global_rad_W"].quantile(0.95))
    else:
        g_ref = np.nan

    if export_peak_kW > 0.0 and np.isfinite(g_ref) and g_ref > 0.0:
        floor_capacity_kwp = float(export_peak_kW * 1000.0 / g_ref)
    else:
        floor_capacity_kwp = 0.0

    # --- Combiner: regression-primary, floor as lower clamp (section 2b) ---
    if regression_capacity_kwp >= floor_capacity_kwp:
        hybrid_capacity_kwp = regression_capacity_kwp
    elif regression_capacity_kwp >= floor_capacity_kwp * 0.7:
        hybrid_capacity_kwp = regression_capacity_kwp
    else:
        hybrid_capacity_kwp = floor_capacity_kwp

    pv_capacity_proxy = hybrid_capacity_kwp / STC_FACTOR

    # Self-consumption share
    days = cust_days.copy()
    hi = days[days["rad_bucket"] == "high"]
    lo = days[days["rad_bucket"] == "low"]

    if hi.empty or lo.empty:
        sc_share = np.nan
    else:
        base_import = lo["Conso_midday"].mean()
        sunny_import = hi["Conso_midday"].mean()
        if pd.isna(base_import) or pd.isna(sunny_import):
            sc_share = np.nan
        else:
            s_imp = max(0.0, base_import - sunny_import)
            exported_midday = hi["Prod_midday"].mean()
            denom = s_imp + max(exported_midday, 0.0)
            if denom <= 0:
                sc_share = np.nan
            else:
                sc_share = float(s_imp / denom)

    return {
        "beta_net": float(full_beta_net),
        "beta_net_se": float(full_beta_net_se),
        "beta_export": beta_export,
        "beta_export_se": beta_export_se,
        "demand_rad_correction": demand_rad_correction,
        "pv_capacity_proxy": pv_capacity_proxy,
        "pv_capacity_regression_kwp": float(regression_capacity_kwp),
        "pv_capacity_floor_kwp": float(floor_capacity_kwp),
        "pv_capacity_hybrid_kwp": float(hybrid_capacity_kwp),
        "sc_share": sc_share,
    }


def _bootstrap_capacity_and_sc(
    cust_df: pd.DataFrame,
    cust_days: pd.DataFrame,
    n_bootstrap: int = 200,
    rng: Optional[np.random.Generator] = None,
    stratify_by_month: bool = True,
    block_size: int = 1,
):
    """
    Block-bootstrap by day to obtain distributions of capacity proxy and
    self-consumption share for a single customer.
    Uses precomputed date->row indices and iloc instead of merge for speed.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_bootstrap <= 0:
        return None, None

    # Precompute date -> row indices (avoids 2 merges per bootstrap iteration)
    date_to_idx_df = cust_df.groupby("date", sort=False).indices
    date_to_idx_days = cust_days.groupby("date", sort=False).indices

    # Restrict to dates present in both 15-min and daily data
    unique_dates = np.array([
        d for d in cust_days["date"].dropna().unique()
        if d in date_to_idx_df and d in date_to_idx_days
    ])
    if unique_dates.size < 3:
        return None, None

    # Optional: month information for stratified sampling (from restricted dates).
    # Pre-sort once so we don't re-sort inside the bootstrap loop.
    unique_dates_sorted = np.array(sorted(unique_dates))
    if stratify_by_month:
        month_df = pd.DataFrame({"date": unique_dates})
        month_df["month"] = pd.to_datetime(month_df["date"]).dt.to_period("M")
        month_to_dates = {
            m: np.array(sorted(grp["date"].values))
            for m, grp in month_df.groupby("month")
        }
    else:
        month_to_dates = None

    def _sample_dates_once() -> np.ndarray:
        """
        Sample a list of dates with replacement, optionally stratified by month
        and optionally in multi-day blocks. Uses pre-sorted month/unique arrays.
        """
        if stratify_by_month and month_to_dates is not None:
            sampled_chunks = []
            for m, dates_m_sorted in month_to_dates.items():
                n_days_m = len(dates_m_sorted)
                if n_days_m == 0:
                    continue

                if block_size <= 1 or n_days_m < block_size:
                    sampled_m = rng.choice(dates_m_sorted, size=n_days_m, replace=True)
                else:
                    if n_days_m <= block_size:
                        sampled_m = rng.choice(dates_m_sorted, size=n_days_m, replace=True)
                    else:
                        blocks = [
                            dates_m_sorted[i : i + block_size]
                            for i in range(0, n_days_m - block_size + 1)
                        ]
                        n_blocks = int(np.ceil(n_days_m / block_size))
                        idx = rng.integers(0, len(blocks), size=n_blocks)
                        sampled_blocks = [blocks[i] for i in idx]
                        sampled_m = np.concatenate(sampled_blocks)
                        if sampled_m.size > n_days_m:
                            sampled_m = sampled_m[:n_days_m]
                sampled_chunks.append(sampled_m)

            if not sampled_chunks:
                return unique_dates
            return np.concatenate(sampled_chunks)
        else:
            dates_sorted = unique_dates_sorted
            n_days = len(dates_sorted)
            if block_size <= 1 or n_days < block_size:
                return rng.choice(dates_sorted, size=n_days, replace=True)
            if n_days <= block_size:
                return rng.choice(dates_sorted, size=n_days, replace=True)
            blocks = [
                dates_sorted[i : i + block_size]
                for i in range(0, n_days - block_size + 1)
            ]
            n_blocks = int(np.ceil(n_days / block_size))
            idx = rng.integers(0, len(blocks), size=n_blocks)
            sampled_blocks = [blocks[i] for i in idx]
            sampled = np.concatenate(sampled_blocks)
            if sampled.size > n_days:
                sampled = sampled[:n_days]
            return sampled

    cap_samples: list[float] = []
    sc_samples: list[float] = []

    for _ in range(n_bootstrap):
        sampled_dates = _sample_dates_once()
        # Index-based resampling: duplicate dates => duplicated row indices
        idx_df = np.concatenate([date_to_idx_df[d] for d in sampled_dates])
        idx_days = np.concatenate([date_to_idx_days[d] for d in sampled_dates])
        boot_df = cust_df.iloc[idx_df].reset_index(drop=True)
        boot_days = cust_days.iloc[idx_days].reset_index(drop=True)

        res = _capacity_and_sc_from_data(boot_df, boot_days)
        cap_samples.append(res["pv_capacity_proxy"])
        sc_samples.append(res["sc_share"])

    return np.array(cap_samples), np.array(sc_samples)


# (kWh/15min)/(W/m²) to kWp: ×4 for 15min→power, ×1000 for STC irradiance
STC_FACTOR = 4000.0


def _process_single_customer(
    cid: str,
    has_pv_prob: float,
    cust_df: pd.DataFrame,
    cust_days: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> Optional[dict]:
    """
    Worker function to process a single customer.
    Designed to be run in parallel on separate CPU cores (picklable, no shared state).
    """
    if cust_df is None or cust_days is None or cust_df.empty or cust_days.empty:
        return None

    cust_df = cust_df.copy()
    cust_df["date"] = cust_df["DT_UTC"].dt.date

    base_res = _capacity_and_sc_from_data(cust_df, cust_days)
    rng = np.random.default_rng(seed)
    cap_boot, sc_boot = _bootstrap_capacity_and_sc(
        cust_df,
        cust_days,
        n_bootstrap=n_bootstrap,
        rng=rng,
        stratify_by_month=True,
        block_size=1,
    )

    if cap_boot is not None and cap_boot.size > 0:
        cap_ci_low, cap_ci_high = np.nanpercentile(cap_boot, [2.5, 97.5])
    else:
        cap_ci_low = cap_ci_high = np.nan

    if sc_boot is not None and sc_boot.size > 0:
        sc_ci_low, sc_ci_high = np.nanpercentile(sc_boot, [2.5, 97.5])
    else:
        sc_ci_low = sc_ci_high = np.nan

    # Interpret cap_boot and pv_capacity_proxy as hybrid slope samples (kWh/15min)/(W/m²)
    pv_capacity_proxy_hybrid = float(base_res["pv_capacity_proxy"])
    hybrid_kwp = pv_capacity_proxy_hybrid * STC_FACTOR
    regression_kwp = float(base_res.get("pv_capacity_regression_kwp", hybrid_kwp))
    floor_kwp = float(base_res.get("pv_capacity_floor_kwp", 0.0))
    floor_to_reg_ratio = (
        floor_kwp / regression_kwp if regression_kwp > 0 else np.nan
    )

    return {
        "customer_id": cid,
        "has_pv_prob": float(has_pv_prob),
        "pv_capacity_mean": pv_capacity_proxy_hybrid,
        "pv_capacity_ci_lower": float(cap_ci_low),
        "pv_capacity_ci_upper": float(cap_ci_high),
        "pv_capacity_kwp": hybrid_kwp,
        "pv_capacity_kwp_ci_lower": float(cap_ci_low) * STC_FACTOR,
        "pv_capacity_kwp_ci_upper": float(cap_ci_high) * STC_FACTOR,
        "pv_capacity_kwp_regression_only": regression_kwp,
        "pv_capacity_kwp_floor": floor_kwp,
        "floor_to_reg_ratio": floor_to_reg_ratio,
        "demand_rad_correction": float(base_res.get("demand_rad_correction", 0.0)),
        "sc_share_mean": float(base_res["sc_share"])
        if base_res["sc_share"] is not None
        else np.nan,
        "sc_share_ci_lower": float(sc_ci_low),
        "sc_share_ci_upper": float(sc_ci_high),
    }


def compute_probabilistic_capacity_parallel(
    re_data_with_meteo: pd.DataFrame,
    daily_features: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    n_bootstrap: int = 200,
    random_state: int = 0,
    max_workers: Optional[int] = 4,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Parallel version of compute_probabilistic_capacity using ThreadPoolExecutor.
    Threads share memory (no fork duplication); the bootstrap inner loop is
    NumPy-heavy and releases the GIL, so threads still provide parallelism.
    """
    if re_data_with_meteo.empty or daily_features.empty or pv_indicators.empty:
        return pd.DataFrame()

    # GroupBy views (do not copy data; customer subsets are materialised on get_group)
    grouped_re_data = re_data_with_meteo.groupby("ID")
    grouped_daily = daily_features.groupby("ID")

    base_rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    customer_tasks = [row for _, row in pv_indicators.iterrows()]
    total_customers = len(customer_tasks)
    print(f"Processing {total_customers} customers in batches of {batch_size}...")

    for i in range(0, total_customers, batch_size):
        batch = customer_tasks[i : i + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_map: dict[concurrent.futures.Future, str] = {}
            for row in batch:
                cid = row["customer_id"]
                has_pv_prob = float(row.get("has_pv_prob", 1.0))
                seed = base_rng.integers(0, 1_000_000)

                try:
                    cust_df = grouped_re_data.get_group(cid)
                    cust_days = grouped_daily.get_group(cid)
                except KeyError:
                    # Customer has no data in one of the groups; skip.
                    continue

                future = executor.submit(
                    _process_single_customer,
                    cid,
                    has_pv_prob,
                    cust_df,
                    cust_days,
                    n_bootstrap,
                    seed,
                )
                futures_map[future] = cid

            done = concurrent.futures.as_completed(futures_map)
            if _HAS_TQDM:
                done = tqdm(done, total=len(futures_map), desc=f"Batch {i // batch_size + 1}")

            for future in done:
                try:
                    res = future.result()
                    if res is not None:
                        rows.append(res)
                except Exception as e:
                    cid = futures_map[future]
                    print(f"Customer {cid} failed: {e}")

        # Exiting the with-block tears down worker processes and frees their RAM before next batch

    return pd.DataFrame(rows)


def compute_probabilistic_capacity(
    re_data_with_meteo: pd.DataFrame,
    daily_features: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    n_bootstrap: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    For each customer, estimate PV capacity proxy and self-consumption share
    with confidence intervals via bootstrap.
    Uses pre-grouped dicts for O(1) customer lookup instead of scanning the full DataFrame.
    """
    if re_data_with_meteo.empty or daily_features.empty or pv_indicators.empty:
        return pd.DataFrame()

    # Pre-group by ID so we don't scan the whole DataFrame per customer (O(N*C) -> O(N+C))
    dict_re_data = dict(tuple(re_data_with_meteo.groupby("ID")))
    dict_daily = dict(tuple(daily_features.groupby("ID")))

    rng = np.random.default_rng(random_state)
    rows = []

    for _, row in pv_indicators.iterrows():
        cid = row["customer_id"]
        cust_df = dict_re_data.get(cid)
        cust_days = dict_daily.get(cid)
        if cust_df is None or cust_days is None:
            continue
        cust_df = cust_df.copy()
        cust_df["date"] = cust_df["DT_UTC"].dt.date
        if cust_df.empty or cust_days.empty:
            continue

        base_res = _capacity_and_sc_from_data(cust_df, cust_days)
        cap_boot, sc_boot = _bootstrap_capacity_and_sc(
            cust_df,
            cust_days,
            n_bootstrap=n_bootstrap,
            rng=np.random.default_rng(rng.integers(0, 1_000_000)),
            stratify_by_month=True,
            block_size=1,
        )

        if cap_boot is not None and cap_boot.size > 0:
            cap_ci_low, cap_ci_high = np.nanpercentile(cap_boot, [2.5, 97.5])
        else:
            cap_ci_low = cap_ci_high = np.nan

        if sc_boot is not None and sc_boot.size > 0:
            sc_ci_low, sc_ci_high = np.nanpercentile(sc_boot, [2.5, 97.5])
        else:
            sc_ci_low = sc_ci_high = np.nan

        pv_capacity_proxy_hybrid = float(base_res["pv_capacity_proxy"])
        hybrid_kwp = pv_capacity_proxy_hybrid * STC_FACTOR
        regression_kwp = float(base_res.get("pv_capacity_regression_kwp", hybrid_kwp))
        floor_kwp = float(base_res.get("pv_capacity_floor_kwp", 0.0))
        floor_to_reg_ratio = (
            floor_kwp / regression_kwp if regression_kwp > 0 else np.nan
        )

        rows.append(
            {
                "customer_id": cid,
                "has_pv_prob": float(row.get("has_pv_prob", 1.0)),
                "pv_capacity_mean": pv_capacity_proxy_hybrid,
                "pv_capacity_ci_lower": float(cap_ci_low),
                "pv_capacity_ci_upper": float(cap_ci_high),
                "pv_capacity_kwp": hybrid_kwp,
                "pv_capacity_kwp_ci_lower": float(cap_ci_low) * STC_FACTOR,
                "pv_capacity_kwp_ci_upper": float(cap_ci_high) * STC_FACTOR,
                "pv_capacity_kwp_regression_only": regression_kwp,
                "pv_capacity_kwp_floor": floor_kwp,
                "floor_to_reg_ratio": floor_to_reg_ratio,
                "demand_rad_correction": float(base_res.get("demand_rad_correction", 0.0)),
                "sc_share_mean": float(base_res["sc_share"])
                if base_res["sc_share"] is not None
                else np.nan,
                "sc_share_ci_lower": float(sc_ci_low),
                "sc_share_ci_upper": float(sc_ci_high),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PV forecasting (15-min, per-customer, laptop-friendly)
# ---------------------------------------------------------------------------

def _safe_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _year_time_index_15min(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=45)
    return pd.date_range(start=start, end=end, freq="15min")


def _estimate_customer_pr_monthly(
    cust_df: pd.DataFrame,
    pv_capacity_kwp: float,
    default_pr: float = 0.85,
    rad_min_w: float = 50.0,
) -> tuple[float, dict]:
    """
    Estimate a per-customer performance ratio (PR) and month multipliers.

    This is designed to be robust when PROD/CONSO semantics are imperfect by
    leaning on net-load slope vs irradiance and using export only when it is
    consistent with radiation.
    """
    pv_capacity_kwp = _safe_float(pv_capacity_kwp, default=0.0)
    if pv_capacity_kwp <= 0 or cust_df is None or cust_df.empty:
        return float(default_pr), {m: 1.0 for m in range(1, 13)}

    req_cols = {"DT_UTC", "CONSO_KWH", "PROD_KWH", "global_rad_W"}
    if not req_cols.issubset(set(cust_df.columns)):
        return float(default_pr), {m: 1.0 for m in range(1, 13)}

    df = cust_df.dropna(subset=["DT_UTC", "global_rad_W"]).copy()
    if df.empty:
        return float(default_pr), {m: 1.0 for m in range(1, 13)}

    df["hour"] = df["DT_UTC"].dt.hour.astype("int16")
    df["month"] = df["DT_UTC"].dt.month.astype("int8")
    rad = df["global_rad_W"].astype("float32")

    # Focus on daylight-ish observations where PV signal exists.
    daylight_mask = (rad >= rad_min_w) & (df["hour"] >= 6) & (df["hour"] <= 20)
    dld = df.loc[daylight_mask].copy()
    if dld.shape[0] < 50:
        return float(default_pr), {m: 1.0 for m in range(1, 13)}

    # Net-load-informed slope (kWh/15min)/(W/m²).
    dld["Net_KWH"] = (dld["CONSO_KWH"].astype("float32") - dld["PROD_KWH"].astype("float32"))
    beta_net, _ = _fit_simple_slope(
        dld["global_rad_W"].to_numpy(),
        dld["Net_KWH"].to_numpy(),
    )
    pv_slope_from_net = max(0.0, -float(beta_net))
    pr_from_net = (pv_slope_from_net * STC_FACTOR) / pv_capacity_kwp if pv_capacity_kwp > 0 else np.nan

    # Export-based slope if export is consistent with radiation.
    prod = dld["PROD_KWH"].astype("float32")
    has_prod = np.isfinite(prod.to_numpy()).any()
    pr_from_prod = np.nan
    if has_prod:
        g = dld.dropna(subset=["PROD_KWH", "global_rad_W"])
        if g.shape[0] >= 50:
            prod_corr = g["PROD_KWH"].corr(g["global_rad_W"])
            pos_rate = float((g["PROD_KWH"] > 0).mean())
            if (prod_corr is not None) and np.isfinite(prod_corr) and prod_corr > 0.25 and pos_rate > 0.01:
                beta_prod, _ = _fit_simple_slope(
                    g["global_rad_W"].to_numpy(),
                    g["PROD_KWH"].to_numpy(),
                )
                pv_slope_from_prod = max(0.0, float(beta_prod))
                pr_from_prod = (pv_slope_from_prod * STC_FACTOR) / pv_capacity_kwp

    def _clip_pr(v: float) -> float:
        if v is None or not np.isfinite(v):
            return np.nan
        # Allow mild >1 due to measurement/model mismatch, but cap hard.
        return float(np.clip(v, 0.05, 1.30))

    pr_from_net = _clip_pr(pr_from_net)
    pr_from_prod = _clip_pr(pr_from_prod)

    if np.isfinite(pr_from_net) and np.isfinite(pr_from_prod):
        pr_base = 0.6 * pr_from_net + 0.4 * pr_from_prod
    elif np.isfinite(pr_from_net):
        pr_base = pr_from_net
    elif np.isfinite(pr_from_prod):
        pr_base = pr_from_prod
    else:
        pr_base = float(default_pr)

    # Month multipliers: estimate relative PR by month using net-load slope.
    pr_by_month: dict[int, float] = {m: 1.0 for m in range(1, 13)}
    month_prs = {}
    for m, grp in dld.groupby("month", sort=True):
        if grp.shape[0] < 50:
            continue
        grp = grp.copy()
        grp["Net_KWH"] = grp["CONSO_KWH"].astype("float32") - grp["PROD_KWH"].astype("float32")
        b_m, _ = _fit_simple_slope(grp["global_rad_W"].to_numpy(), grp["Net_KWH"].to_numpy())
        slope_m = max(0.0, -float(b_m))
        pr_m = (slope_m * STC_FACTOR) / pv_capacity_kwp if pv_capacity_kwp > 0 else np.nan
        pr_m = _clip_pr(pr_m)
        if np.isfinite(pr_m):
            month_prs[int(m)] = pr_m

    if month_prs:
        # Normalize to mean=1.0 so pr_base controls the global scale.
        vals = np.array(list(month_prs.values()), dtype="float64")
        mean_val = float(np.nanmean(vals)) if vals.size else np.nan
        if np.isfinite(mean_val) and mean_val > 0:
            for m in range(1, 13):
                if m in month_prs:
                    pr_by_month[m] = float(np.clip(month_prs[m] / mean_val, 0.6, 1.4))

    return float(pr_base), pr_by_month


def predict_customer_pv_15min(
    customer_id: str,
    pv_capacity_kwp: float,
    avg_meteo_15min: pd.DataFrame,
    forecast_year: int,
    cust_history_with_meteo: Optional[pd.DataFrame] = None,
    default_pr: float = 0.85,
) -> pd.DataFrame:
    """
    Produce a full-year 15-min PV energy forecast (kWh per 15 min) for one customer.

    Model:
      pv_kwh_15min = pv_capacity_kwp * PR(customer) * PR_month(month) * (rad_Wm2 / 1000) * 0.25

    Uses net-load-informed calibration if customer history is provided.
    """
    pv_capacity_kwp = _safe_float(pv_capacity_kwp, default=0.0)
    if pv_capacity_kwp <= 0 or avg_meteo_15min is None or avg_meteo_15min.empty:
        return pd.DataFrame(columns=["customer_id", "DT_UTC", "pv_forecast_kwh_15min", "global_rad_W", "pv_capacity_kwp"])

    if "global_rad_W" not in avg_meteo_15min.columns:
        raise ValueError("avg_meteo_15min must contain column 'global_rad_W'.")

    pr_base, pr_by_month = _estimate_customer_pr_monthly(
        cust_history_with_meteo if cust_history_with_meteo is not None else pd.DataFrame(),
        pv_capacity_kwp=pv_capacity_kwp,
        default_pr=default_pr,
    )

    idx = _year_time_index_15min(int(forecast_year))
    met = avg_meteo_15min.reindex(idx)[["global_rad_W"]].copy()
    met["month"] = met.index.month.astype("int8")
    met["pr_month"] = met["month"].map(pr_by_month).astype("float32").fillna(1.0)

    rad = met["global_rad_W"].astype("float32")
    rad = rad.clip(lower=0.0)

    # capacity (kWp) * (rad/1000) -> kW at PR=1. Multiply by 0.25 h for 15-min energy.
    pv_kwh_15min = (pv_capacity_kwp * float(pr_base)) * met["pr_month"] * (rad / 1000.0) * 0.25
    pv_kwh_15min = pv_kwh_15min.astype("float32").clip(lower=0.0)

    out = pd.DataFrame(
        {
            "customer_id": str(customer_id),
            "DT_UTC": met.index,
            "pv_forecast_kwh_15min": pv_kwh_15min.to_numpy(),
            "global_rad_W": rad.to_numpy(),
            "pv_capacity_kwp": np.float32(pv_capacity_kwp),
        }
    )
    return out


def _forecast_open_parquet_writer(output_path: str):
    """
    Open a pyarrow ParquetWriter for incremental writes.
    Returns (writer, pa) so callers can create tables without re-importing.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # We don't know the schema until we see the first batch.
    return None, pa, pq


def forecast_pv_for_customers_streaming(
    data_dir: str,
    cust_file_index: dict,
    prob_summary: pd.DataFrame,
    avg_meteo_15min: pd.DataFrame,
    forecast_year: int,
    output_path: str,
    batch_customers: int = 25,
    default_pr: float = 0.85,
    min_capacity_kwp: float = 0.1,
) -> str:
    """
    Stream through customers, produce 15-min PV forecasts for *forecast_year*,
    and write a single parquet file at *output_path*.

    Memory profile: keeps at most one customer's history and a small batch of
    forecast frames in RAM.
    """
    if prob_summary is None or prob_summary.empty:
        raise ValueError("prob_summary is empty; cannot forecast.")
    if "customer_id" not in prob_summary.columns or "pv_capacity_kwp" not in prob_summary.columns:
        raise ValueError("prob_summary must contain 'customer_id' and 'pv_capacity_kwp'.")

    df = prob_summary[["customer_id", "pv_capacity_kwp"]].copy()
    df["pv_capacity_kwp"] = pd.to_numeric(df["pv_capacity_kwp"], errors="coerce")
    df = df.dropna(subset=["customer_id", "pv_capacity_kwp"])
    df = df[df["pv_capacity_kwp"] >= float(min_capacity_kwp)]
    if df.empty:
        raise ValueError("No customers with capacity above threshold to forecast.")

    # Deterministic order for reproducibility.
    df = df.sort_values("customer_id")

    # Write incrementally to a single parquet file (keeps RAM bounded).
    writer = None
    pa = pq = None
    try:
        writer, pa, pq = _forecast_open_parquet_writer(output_path)
    except Exception as e:
        raise RuntimeError(
            "Parquet writing requires pyarrow. "
            "Install/repair pyarrow in the active environment."
        ) from e

    n_since_flush = 0
    for _, row in df.iterrows():
        cid = str(row["customer_id"])
        cap = float(row["pv_capacity_kwp"])

        hist = load_single_customer(cid, data_dir, cust_file_index, avg_meteo_15min)
        try:
            pred = predict_customer_pv_15min(
                customer_id=cid,
                pv_capacity_kwp=cap,
                avg_meteo_15min=avg_meteo_15min,
                forecast_year=int(forecast_year),
                cust_history_with_meteo=hist if hist is not None and not hist.empty else None,
                default_pr=default_pr,
            )
        finally:
            del hist

        if pred is None or pred.empty:
            continue

        table = pa.Table.from_pandas(pred, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

        n_since_flush += 1
        # Keep a small, tunable cadence for progress without storing batches.
        if int(batch_customers) > 0 and n_since_flush >= int(batch_customers):
            n_since_flush = 0

    if writer is not None:
        writer.close()

    return str(output_path)


def validate_portfolio_forecast(
    forecast_parquet_path: str,
    prob_summary: pd.DataFrame,
    max_specific_yield_kwh_kwp: float = 1800.0,
) -> dict:
    """
    Portfolio-level plausibility checks (lightweight).
    Reads parquet once and compares aggregate yield against installed capacity.
    """
    if prob_summary is None or prob_summary.empty:
        return {"ok": False, "reason": "empty_prob_summary"}
    if "pv_capacity_kwp" not in prob_summary.columns:
        return {"ok": False, "reason": "missing_capacity_column"}

    total_cap = float(pd.to_numeric(prob_summary["pv_capacity_kwp"], errors="coerce").fillna(0.0).sum())
    if total_cap <= 0:
        return {"ok": False, "reason": "zero_total_capacity"}

    df = pd.read_parquet(forecast_parquet_path, columns=["pv_forecast_kwh_15min"])
    total_kwh = float(np.nansum(pd.to_numeric(df["pv_forecast_kwh_15min"], errors="coerce").to_numpy(dtype="float64")))
    specific = total_kwh / total_cap

    if specific > float(max_specific_yield_kwh_kwp):
        return {
            "ok": False,
            "reason": "portfolio_specific_yield_too_high",
            "portfolio_specific_yield_kwh_kwp": specific,
            "total_capacity_kwp": total_cap,
            "total_forecast_kwh": total_kwh,
        }

    return {
        "ok": True,
        "portfolio_specific_yield_kwh_kwp": specific,
        "total_capacity_kwp": total_cap,
        "total_forecast_kwh": total_kwh,
    }


def validate_customer_forecast(
    forecast_df: pd.DataFrame,
    pv_capacity_kwp: float,
    max_specific_yield_kwh_kwp: float = 1800.0,
) -> dict:
    """
    Lightweight plausibility checks for a single customer's forecast.
    """
    if forecast_df is None or forecast_df.empty:
        return {"ok": False, "reason": "empty_forecast"}
    if "pv_forecast_kwh_15min" not in forecast_df.columns:
        return {"ok": False, "reason": "missing_column_pv_forecast_kwh_15min"}
    pv = pd.to_numeric(forecast_df["pv_forecast_kwh_15min"], errors="coerce")
    if (pv < -1e-6).any():
        return {"ok": False, "reason": "negative_values"}

    cap = _safe_float(pv_capacity_kwp, default=0.0)
    if cap > 0:
        annual_kwh = float(np.nansum(pv.to_numpy(dtype="float64")))
        specific = annual_kwh / cap
        if specific > float(max_specific_yield_kwh_kwp):
            return {"ok": False, "reason": "specific_yield_too_high", "specific_yield_kwh_kwp": specific}

    return {"ok": True}


# ---------------------------------------------------------------------------
# Portfolio-level aggregation
# ---------------------------------------------------------------------------

def _capacity_weighted_sc_aggregate(sc_df: pd.DataFrame) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    Capacity-weighted mean of ``sc_share_mean`` and conservative / independence CIs.

    Expects columns: ``pv_capacity_kwp``, ``sc_share_mean``,
    ``sc_share_ci_lower``, ``sc_share_ci_upper``.
    """
    if sc_df.empty:
        return float(np.nan), (float(np.nan), float(np.nan)), (float(np.nan), float(np.nan))
    weights = sc_df["pv_capacity_kwp"]
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return float(np.nan), (float(np.nan), float(np.nan)), (float(np.nan), float(np.nan))

    agg_sc = float((weights * sc_df["sc_share_mean"]).sum() / w_sum)
    sc_lo = sc_df["sc_share_ci_lower"].fillna(sc_df["sc_share_mean"])
    sc_hi = sc_df["sc_share_ci_upper"].fillna(sc_df["sc_share_mean"])
    agg_sc_ci_conservative = (
        float(np.clip((weights * sc_lo).sum() / w_sum, 0, 1)),
        float(np.clip((weights * sc_hi).sum() / w_sum, 0, 1)),
    )
    sc_sigma_i = (sc_hi - sc_lo) / (2.0 * 1.96)
    sc_sigma_agg = float(np.sqrt(((weights ** 2) * (sc_sigma_i ** 2)).sum()) / w_sum)
    agg_sc_ci_independence = (
        float(np.clip(agg_sc - 1.96 * sc_sigma_agg, 0, 1)),
        float(np.clip(agg_sc + 1.96 * sc_sigma_agg, 0, 1)),
    )
    return agg_sc, agg_sc_ci_conservative, agg_sc_ci_independence


def aggregate_portfolio_estimates(
    prob_summary: pd.DataFrame,
    pv_indicators: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Aggregate per-customer PV capacity and self-consumption estimates to
    portfolio level, with two uncertainty bounds per metric:

    * **Conservative** (perfect-correlation): sum of individual CI endpoints.
    * **Independence** (zero-correlation): propagate via sqrt-sum-of-variances.

    The true portfolio CI lies between these two bounds because customer
    errors are partially correlated (shared regional radiation, same STC
    factor) but not perfectly so (individual behaviour, local conditions).

    Parameters
    ----------
    prob_summary : DataFrame
        Output of ``compute_probabilistic_capacity[_parallel]``.
    pv_indicators : DataFrame, optional
        If provided, yearly production/consumption totals are included.

    Returns
    -------
    dict  with keys documented inline.

    Self-consumption aggregates use capacity weights among hybrid-PV-positive
    customers. When ``sc_share_mean`` is present, ``aggregate_sc_share_nonzero_sc``
    repeats the same weighting over rows with ``sc_share_mean > 0`` only
    (``n_customers_nonzero_sc`` is the count of those customers).
    """
    if prob_summary.empty:
        return {}

    df = prob_summary.copy()

    # -- Filter to rows with valid hybrid capacity --
    valid = df["pv_capacity_kwp"].notna() & (df["pv_capacity_kwp"] > 0)
    df_valid = df[valid]
    n = len(df_valid)
    if n == 0:
        return {}

    # ---- Total PV capacity (point estimates) ----
    total_hybrid = float(df_valid["pv_capacity_kwp"].sum())
    total_regression = float(df_valid["pv_capacity_kwp_regression_only"].fillna(0).sum())
    total_floor = float(df_valid["pv_capacity_kwp_floor"].fillna(0).sum())

    # ---- Per-customer sigma from bootstrap CI (assume ~normal) ----
    ci_lo = df_valid["pv_capacity_kwp_ci_lower"].fillna(df_valid["pv_capacity_kwp"])
    ci_hi = df_valid["pv_capacity_kwp_ci_upper"].fillna(df_valid["pv_capacity_kwp"])
    sigma_i = (ci_hi - ci_lo) / (2.0 * 1.96)

    # Conservative bound: sum of endpoints (perfect correlation)
    cap_ci_conservative = (float(ci_lo.sum()), float(ci_hi.sum()))

    # Independence bound: sqrt-sum-of-variances
    sigma_agg = float(np.sqrt((sigma_i ** 2).sum()))
    cap_ci_independence = (
        max(0.0, total_hybrid - 1.96 * sigma_agg),
        total_hybrid + 1.96 * sigma_agg,
    )

    # ---- Aggregate self-consumption share (capacity-weighted mean) ----
    sc_valid = df_valid.dropna(subset=["sc_share_mean"])
    agg_sc, agg_sc_ci_conservative, agg_sc_ci_independence = _capacity_weighted_sc_aggregate(
        sc_valid
    )

    sc_nonzero = sc_valid[sc_valid["sc_share_mean"] > 0]
    n_nonzero_sc = int(len(sc_nonzero))
    agg_sc_nz, agg_sc_ci_con_nz, agg_sc_ci_ind_nz = _capacity_weighted_sc_aggregate(sc_nonzero)

    # ---- Diagnostic ratio ----
    portfolio_f2r = total_floor / total_regression if total_regression > 0 else np.nan

    # ---- Distribution statistics ----
    caps = df_valid["pv_capacity_kwp"]
    capacity_stats = {
        "mean": float(caps.mean()),
        "median": float(caps.median()),
        "std": float(caps.std()),
        "min": float(caps.min()),
        "max": float(caps.max()),
    }

    # ---- Optional yearly totals from pv_indicators ----
    yearly_totals = {}
    if pv_indicators is not None and not pv_indicators.empty:
        matched = pv_indicators[
            pv_indicators["customer_id"].isin(df_valid["customer_id"])
        ]
        yearly_totals["total_yearly_prod_kwh"] = float(
            matched["yearly_prod"].fillna(0).sum()
        )
        yearly_totals["total_yearly_cons_kwh"] = float(
            matched["yearly_cons"].fillna(0).sum()
        )

    result = {
        "n_customers": n,
        "total_hybrid_kwp": total_hybrid,
        "total_regression_kwp": total_regression,
        "total_floor_kwp": total_floor,
        "ci_conservative": cap_ci_conservative,
        "ci_independence": cap_ci_independence,
        "aggregate_sc_share": agg_sc,
        "aggregate_sc_ci_conservative": agg_sc_ci_conservative,
        "aggregate_sc_ci_independence": agg_sc_ci_independence,
        "aggregate_sc_share_nonzero_sc": agg_sc_nz,
        "aggregate_sc_ci_conservative_nonzero_sc": agg_sc_ci_con_nz,
        "aggregate_sc_ci_independence_nonzero_sc": agg_sc_ci_ind_nz,
        "n_customers_nonzero_sc": n_nonzero_sc,
        "portfolio_floor_to_reg_ratio": portfolio_f2r,
        "capacity_stats": capacity_stats,
        **yearly_totals,
    }
    return result


# ---------------------------------------------------------------------------
# Yield validation (section 4a of plan)
# ---------------------------------------------------------------------------

def validate_yield_plausibility(
    prob_summary: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    yield_low: float = 800.0,
    yield_high: float = 1200.0,
) -> pd.DataFrame:
    """
    Per-customer yield plausibility check for Vaud (~800-1200 kWh/kWp/year).

    Adds columns to prob_summary:
      - estimated_total_gen_kwh: total generation inferred from export + SC
      - specific_yield_kwh_kwp: generation / capacity
      - yield_flag: 'plausible', 'low', 'high', or 'no_data'
    """
    df = prob_summary.copy()
    ind = pv_indicators[["customer_id", "yearly_prod", "yearly_cons"]].copy()
    ind = ind.rename(columns={"yearly_prod": "yearly_export_kwh"})
    df = df.merge(ind, on="customer_id", how="left")

    sc = df["sc_share_mean"].fillna(0.0).clip(0.0, 0.7)
    yearly_export = df["yearly_export_kwh"].fillna(0.0)
    denom = (1.0 - sc).replace(0.0, np.nan)
    df["estimated_total_gen_kwh"] = yearly_export / denom

    cap = df["pv_capacity_kwp"].replace(0.0, np.nan)
    df["specific_yield_kwh_kwp"] = df["estimated_total_gen_kwh"] / cap

    def _flag(row):
        sy = row["specific_yield_kwh_kwp"]
        if pd.isna(sy) or pd.isna(row["pv_capacity_kwp"]) or row["pv_capacity_kwp"] <= 0:
            return "no_data"
        if sy < yield_low:
            return "low"
        if sy > yield_high:
            return "high"
        return "plausible"

    df["yield_flag"] = df.apply(_flag, axis=1)
    return df


# ---------------------------------------------------------------------------
# Segment plausibility (section 3b of plan)
# ---------------------------------------------------------------------------

def load_customer_metadata(data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load the Romande Energie customer metadata parquet (contains segment info).

    Typical columns include ``ID`` and ``TYPE_PARTENAIRE_LIBELLE`` (e.g. value
    ``"Particuliers"``). Returns None if the file is missing or unreadable.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data" / "re_data" / "ETHZ")
    meta_path = Path(data_dir) / "metadata"
    if not meta_path.exists():
        return None
    try:
        meta = pd.read_parquet(meta_path)
        if "ID" in meta.columns:
            meta["ID"] = meta["ID"].astype(str)
        return meta
    except Exception:
        return None


PARTICULIERS_PARTNER_LABEL = "Particuliers"


def particuliers_customer_ids(metadata: pd.DataFrame) -> Set[str]:
    """
    Return customer IDs whose partner type is Romande Energie *Particuliers*.

    Uses column ``TYPE_PARTENAIRE_LIBELLE`` and exact label *Particuliers*
    (see ``PARTICULIERS_PARTNER_LABEL``, aligned with ``data/fast_load_smart_meter.py``).
    """
    if metadata is None or metadata.empty:
        raise ValueError("metadata is empty or None; cannot resolve Particuliers IDs.")
    if "ID" not in metadata.columns:
        raise ValueError(
            f"metadata has no 'ID' column; columns={sorted(map(str, metadata.columns))}"
        )
    col = "TYPE_PARTENAIRE_LIBELLE"
    if col not in metadata.columns:
        raise ValueError(
            f"metadata has no {col!r} column (needed for Particuliers filter); "
            f"columns={sorted(map(str, metadata.columns))}"
        )
    sub = metadata.loc[metadata[col] == PARTICULIERS_PARTNER_LABEL, "ID"].astype(str)
    return set(sub.unique())


def eligible_pv_customer_ids(
    customer_summary: pd.DataFrame,
    particulier_ids: Set[str],
    max_cons_kwh: float = 100_000,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Restrict *customer_summary* to Particuliers with total consumption <= cap.

    Returns a filtered copy of *customer_summary* and the set of eligible IDs.
    """
    if customer_summary is None or customer_summary.empty:
        return customer_summary, set()
    if "ID" not in customer_summary.columns or "sum_cons_kwh" not in customer_summary.columns:
        raise ValueError("customer_summary must contain 'ID' and 'sum_cons_kwh'.")
    cons_ok = customer_summary["sum_cons_kwh"] <= float(max_cons_kwh)
    ids = set(customer_summary.loc[cons_ok, "ID"].astype(str)) & set(particulier_ids)
    filtered = customer_summary.loc[customer_summary["ID"].astype(str).isin(ids)].copy()
    return filtered, ids


def compute_segment_stats(
    prob_summary: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    min_segment_size: int = 30,
) -> pd.DataFrame:
    """
    Per-segment capacity distribution stats for outlier detection.

    If metadata is None or segment_col missing, uses a single "all" segment.
    Small segments (< min_segment_size) are merged into "other".
    """
    df = prob_summary.copy()
    df = df[df["pv_capacity_kwp"].notna() & (df["pv_capacity_kwp"] > 0)]

    if metadata is not None and segment_col in metadata.columns:
        id_col = "ID" if "ID" in metadata.columns else metadata.columns[0]
        seg_map = metadata.set_index(id_col)[segment_col]
        df["segment"] = df["customer_id"].map(seg_map).fillna("unknown")
    else:
        df["segment"] = "all"

    seg_counts = df["segment"].value_counts()
    small_segs = seg_counts[seg_counts < min_segment_size].index
    df.loc[df["segment"].isin(small_segs), "segment"] = "other"

    stats_rows = []
    for seg, grp in df.groupby("segment"):
        caps = grp["pv_capacity_kwp"]
        q1 = float(caps.quantile(0.25))
        q3 = float(caps.quantile(0.75))
        iqr = q3 - q1
        stats_rows.append({
            "segment": seg,
            "n_customers": len(grp),
            "cap_mean": float(caps.mean()),
            "cap_median": float(caps.median()),
            "cap_q1": q1,
            "cap_q3": q3,
            "cap_iqr": iqr,
            "cap_lower_fence": max(0.0, q1 - 2.0 * iqr),
            "cap_upper_fence": q3 + 2.0 * iqr,
        })

    return pd.DataFrame(stats_rows)


def flag_implausible_estimates(
    prob_summary: pd.DataFrame,
    segment_stats: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    yield_bounds: tuple = (800.0, 1200.0),
) -> pd.DataFrame:
    """
    Flag customers with implausible capacity estimates using segment IQR fences
    and yield bounds.

    Returns prob_summary with added columns:
      - segment: customer segment
      - capacity_flag: 'ok', 'too_low', 'too_high'
    """
    df = prob_summary.copy()

    if metadata is not None and segment_col in metadata.columns:
        id_col = "ID" if "ID" in metadata.columns else metadata.columns[0]
        seg_map = metadata.set_index(id_col)[segment_col]
        df["segment"] = df["customer_id"].map(seg_map).fillna("unknown")
    else:
        df["segment"] = "all"

    small_segs = set(df["segment"].unique()) - set(segment_stats["segment"].unique())
    df.loc[df["segment"].isin(small_segs), "segment"] = "other"

    fence_map = segment_stats.set_index("segment")

    def _flag_cap(row):
        seg = row["segment"]
        cap = row.get("pv_capacity_kwp", 0.0)
        if pd.isna(cap) or cap <= 0:
            return "no_data"
        if seg not in fence_map.index:
            return "ok"
        lower = fence_map.loc[seg, "cap_lower_fence"]
        upper = fence_map.loc[seg, "cap_upper_fence"]
        if cap < lower:
            return "too_low"
        if cap > upper:
            return "too_high"
        return "ok"

    df["capacity_flag"] = df.apply(_flag_cap, axis=1)
    return df


# ---------------------------------------------------------------------------
# Evaluation framework (section 4 of plan)
# ---------------------------------------------------------------------------

def evaluate_portfolio(
    prob_summary: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    yield_bounds: tuple = (800.0, 1200.0),
) -> dict:
    """
    Comprehensive portfolio evaluation:
      1. Yield distribution and plausibility
      2. Estimator agreement (floor vs regression)
      3. Cross-segment consistency
      4. Flagged customer lists

    Returns dict with summary stats, DataFrames, and diagnostic info.
    """
    result: dict = {}

    # 1. Yield validation
    yield_df = validate_yield_plausibility(
        prob_summary, pv_indicators,
        yield_low=yield_bounds[0], yield_high=yield_bounds[1],
    )
    yield_counts = yield_df["yield_flag"].value_counts().to_dict()
    sy = yield_df["specific_yield_kwh_kwp"].dropna()
    result["yield_stats"] = {
        "median": float(sy.median()) if not sy.empty else np.nan,
        "mean": float(sy.mean()) if not sy.empty else np.nan,
        "std": float(sy.std()) if not sy.empty else np.nan,
        "flag_counts": yield_counts,
    }
    result["yield_df"] = yield_df

    # 2. Estimator agreement
    valid = yield_df[
        (yield_df["pv_capacity_kwp_regression_only"] > 0)
        & (yield_df["pv_capacity_kwp_floor"] > 0)
    ].copy()
    if not valid.empty:
        ratio = valid["pv_capacity_kwp_floor"] / valid["pv_capacity_kwp_regression_only"]
        result["estimator_agreement"] = {
            "floor_to_reg_ratio_median": float(ratio.median()),
            "floor_to_reg_ratio_mean": float(ratio.mean()),
            "floor_to_reg_ratio_std": float(ratio.std()),
            "pct_floor_gt_regression": float((ratio > 1.0).mean() * 100),
            "pct_within_30pct": float(((ratio >= 0.7) & (ratio <= 1.3)).mean() * 100),
        }
    else:
        result["estimator_agreement"] = {}

    # 3. Segment analysis
    seg_stats = compute_segment_stats(
        prob_summary, metadata=metadata, segment_col=segment_col,
    )
    result["segment_stats"] = seg_stats

    flagged_df = flag_implausible_estimates(
        yield_df, seg_stats, metadata=metadata, segment_col=segment_col,
    )
    cap_flag_counts = flagged_df["capacity_flag"].value_counts().to_dict()
    result["capacity_flag_counts"] = cap_flag_counts
    result["flagged_df"] = flagged_df

    # 4. Cross-segment consistency
    if seg_stats.shape[0] > 1:
        seg_summary = seg_stats[["segment", "n_customers", "cap_median", "cap_mean"]].copy()
        result["cross_segment"] = seg_summary
    else:
        result["cross_segment"] = seg_stats

    # 5. Aggregate portfolio check
    total_cap = prob_summary["pv_capacity_kwp"].sum()
    total_export = pv_indicators["yearly_prod"].sum() if "yearly_prod" in pv_indicators.columns else np.nan
    result["portfolio_check"] = {
        "total_capacity_kwp": float(total_cap),
        "total_yearly_export_kwh": float(total_export),
        "implied_yield_kwh_kwp": float(total_export / total_cap) if total_cap > 0 else np.nan,
    }

    return result


def print_evaluation_report(evaluation: dict) -> None:
    """Print a human-readable evaluation report to stdout."""
    print("=" * 70)
    print("PV CAPACITY ESTIMATION – EVALUATION REPORT")
    print("=" * 70)

    ys = evaluation.get("yield_stats", {})
    print(f"\n--- Yield Plausibility ---")
    print(f"  Specific yield: median = {ys.get('median', float('nan')):.0f}, "
          f"mean = {ys.get('mean', float('nan')):.0f}, "
          f"std = {ys.get('std', float('nan')):.0f} kWh/kWp")
    for flag, count in ys.get("flag_counts", {}).items():
        print(f"  {flag}: {count} customers")

    ea = evaluation.get("estimator_agreement", {})
    if ea:
        print(f"\n--- Estimator Agreement (floor vs regression) ---")
        print(f"  Floor/Reg ratio: median = {ea.get('floor_to_reg_ratio_median', float('nan')):.2f}, "
              f"mean = {ea.get('floor_to_reg_ratio_mean', float('nan')):.2f}")
        print(f"  Floor > Regression: {ea.get('pct_floor_gt_regression', float('nan')):.1f}%")
        print(f"  Within ±30%: {ea.get('pct_within_30pct', float('nan')):.1f}%")

    print(f"\n--- Capacity Flags ---")
    for flag, count in evaluation.get("capacity_flag_counts", {}).items():
        print(f"  {flag}: {count}")

    cs = evaluation.get("cross_segment")
    if cs is not None and not cs.empty:
        print(f"\n--- Cross-Segment Consistency ---")
        print(cs.to_string(index=False))

    pc = evaluation.get("portfolio_check", {})
    if pc:
        print(f"\n--- Portfolio-Level Check ---")
        print(f"  Total capacity: {pc.get('total_capacity_kwp', float('nan')):.0f} kWp")
        print(f"  Total yearly export: {pc.get('total_yearly_export_kwh', float('nan')):.0f} kWh")
        print(f"  Implied yield: {pc.get('implied_yield_kwh_kwp', float('nan')):.0f} kWh/kWp")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Streaming PV indicator computation
# ---------------------------------------------------------------------------

def stream_pv_indicators(
    data_dir: str,
    target_ids: set,
    avg_meteo_15min: pd.DataFrame,
    daily_weather: pd.DataFrame,
    cust_file_index: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Stream through parquet files to compute PV indicators for *target_ids*
    without loading the full dataset into memory.

    Single-file customers (98%+) are processed immediately per file.
    Multi-file customers are accumulated and processed at the end.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the parquet files.
    target_ids : set[str]
        Customer IDs to process (e.g. from stream_customer_summary).
    avg_meteo_15min : DataFrame
        Regional meteo time series at 15-min resolution.
    daily_weather : DataFrame
        Pre-computed daily weather with rad_bucket (from compute_daily_weather).
    cust_file_index : dict, optional
        {customer_id: [file_paths]} mapping.  Built automatically if not given.

    Returns
    -------
    pv_indicators : DataFrame
    """
    if cust_file_index is None:
        cust_file_index = build_customer_file_index(data_dir)

    meteo_for_merge = _build_meteo_merge_frame(avg_meteo_15min)

    single_file_cids = {
        c for c in target_ids
        if len(cust_file_index.get(c, [])) == 1
    }
    multi_file_cids = {
        c for c in target_ids
        if len(cust_file_index.get(c, [])) > 1
    }

    all_indicator_dfs: list[pd.DataFrame] = []
    multi_file_accum: dict[str, list[pd.DataFrame]] = {
        c: [] for c in multi_file_cids
    }

    files = re_data.get_parquet_files(data_dir)
    _iter = tqdm(files, desc="Computing indicators") if _HAS_TQDM else files

    for fpath in _iter:
        df = _prepare_raw_file(fpath, meteo_for_merge, target_ids)
        if df.empty:
            continue

        file_ids = set(df["ID"].unique())

        singles_here = file_ids & single_file_cids
        if singles_here:
            batch_df = df[df["ID"].isin(singles_here)]
            batch_daily = build_customer_daily_features(batch_df, daily_weather)
            if not batch_daily.empty:
                batch_ind = compute_pv_indicators(batch_daily, batch_df)
                if not batch_ind.empty:
                    all_indicator_dfs.append(batch_ind)

        multis_here = file_ids & multi_file_cids
        for cid in multis_here:
            chunk = df[df["ID"] == cid]
            if not chunk.empty:
                multi_file_accum[cid].append(chunk.copy())

        del df

    if multi_file_cids:
        multi_chunks = []
        for cid, chunks in multi_file_accum.items():
            if chunks:
                multi_chunks.append(pd.concat(chunks, ignore_index=True))
        del multi_file_accum

        if multi_chunks:
            multi_df = pd.concat(multi_chunks, ignore_index=True)
            del multi_chunks
            multi_daily = build_customer_daily_features(multi_df, daily_weather)
            if not multi_daily.empty:
                multi_ind = compute_pv_indicators(multi_daily, multi_df)
                if not multi_ind.empty:
                    all_indicator_dfs.append(multi_ind)
            del multi_df

    if all_indicator_dfs:
        return pd.concat(all_indicator_dfs, ignore_index=True)
    return pd.DataFrame()


def load_single_customer(
    customer_id: str,
    data_dir: str,
    cust_file_index: dict,
    avg_meteo_15min: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load one customer's 15-min data with meteo merged (for plotting).
    Only reads the file(s) that contain this customer.
    """
    files = cust_file_index.get(customer_id, [])
    if not files:
        return pd.DataFrame()

    meteo_for_merge = _build_meteo_merge_frame(avg_meteo_15min)
    chunks = []
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            df["ID"] = df["ID"].astype(str)
            chunk = df.loc[df["ID"] == customer_id].copy()
            del df
            if chunk.empty:
                continue
            dt = pd.to_datetime(chunk["DT_UTC"], utc=True)
            chunk["DT_UTC"] = dt.dt.tz_convert(None)
            for col in ("CONSO_KWH", "PROD_KWH"):
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype("float32")
            chunk = chunk.merge(meteo_for_merge, on="DT_UTC", how="left")
            chunks.append(chunk)
        except Exception:
            continue

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("DT_UTC")


def stream_portfolio_aggregate_load(
    data_dir: str,
    cust_file_index: dict,
    avg_meteo_15min: pd.DataFrame,
    pv_customer_ids: set,
    resolution: str = "D",
) -> pd.DataFrame:
    """
    Compute aggregate (summed) load profile across PV customers by streaming
    through files one at a time.  Returns a DataFrame indexed by timestamp
    with columns [Import_kW, Export_kW, Net_kW].
    """
    meteo_for_merge = _build_meteo_merge_frame(avg_meteo_15min)
    files = re_data.get_parquet_files(data_dir)

    agg_series: Optional[pd.DataFrame] = None

    _iter = tqdm(files, desc="Aggregating load") if _HAS_TQDM else files
    for fpath in _iter:
        df = _prepare_raw_file(fpath, meteo_for_merge, pv_customer_ids)
        if df.empty:
            continue

        df["Import_kW"] = df["CONSO_KWH"] * 4.0
        df["Export_kW"] = df["PROD_KWH"] * 4.0

        file_agg = (
            df.set_index("DT_UTC")
            .groupby(level=0)[["Import_kW", "Export_kW"]]
            .sum()
        )
        if agg_series is None:
            agg_series = file_agg
        else:
            agg_series = agg_series.add(file_agg, fill_value=0)
        del df

    if agg_series is None:
        return pd.DataFrame()

    agg = agg_series.resample(resolution).mean()
    agg["Net_kW"] = agg["Import_kW"] - agg["Export_kW"]
    return agg


# ---------------------------------------------------------------------------
# Streaming data loader (section 5b of plan)
# ---------------------------------------------------------------------------

def build_customer_file_index(
    data_dir: str,
    restrict_to_ids: Optional[Set[str]] = None,
) -> dict:
    """
    Build {customer_id: [file_paths]} index by scanning ID columns.
    Most customers appear in 1 file (median=1, max~3).

    If *restrict_to_ids* is set, only those customer IDs are kept in the map
    (files are still scanned, but the dict stays small for downstream steps).
    """
    from collections import defaultdict
    files = re_data.get_parquet_files(data_dir)
    cust_to_files: dict[str, list] = defaultdict(list)
    restrict = restrict_to_ids
    for fpath in files:
        try:
            df = pd.read_parquet(fpath, columns=["ID"])
            for cid in df["ID"].astype(str).unique():
                if restrict is not None and cid not in restrict:
                    continue
                cust_to_files[cid].append(fpath)
        except Exception:
            continue
    return dict(cust_to_files)


def process_customers_streaming(
    data_dir: str,
    avg_meteo_15min: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    n_bootstrap: int = 200,
    max_workers: int = 4,
    batch_size: int = 500,
    random_state: int = 0,
    daily_weather: Optional[pd.DataFrame] = None,
    cust_file_index: Optional[dict] = None,
    autosave_enabled: bool = False,
    autosave_path: Optional[str] = None,
    autosave_every_customers: int = 500,
    resume_from_autosave: bool = False,
    resume_from_file: Optional[str] = None,
    overwrite_autosave: bool = True,
) -> pd.DataFrame:
    """
    Stream-process customers file-by-file instead of loading all into RAM.

    Two-pass approach:
      1. Build ID->files index (lightweight, columns=["ID"] only)
      2. Process file-by-file; single-file customers immediately,
         multi-file customers accumulated and processed after last file.
    """
    if cust_file_index is None:
        cust_file_index = build_customer_file_index(data_dir)
    if not cust_file_index:
        return pd.DataFrame()

    if daily_weather is None:
        daily_weather = compute_daily_weather(avg_meteo_15min)

    indicator_ids = set(pv_indicators["customer_id"].unique())
    relevant_cids = [c for c in cust_file_index if c in indicator_ids]

    prob_lookup = pv_indicators.set_index("customer_id")
    base_rng = np.random.default_rng(random_state)

    single_file_cids = [c for c in relevant_cids if len(cust_file_index[c]) == 1]
    multi_file_cids = [c for c in relevant_cids if len(cust_file_index[c]) > 1]

    # -------------------------------
    # Autosave / resume configuration
    # -------------------------------
    if autosave_path is None:
        autosave_path = str(Path(data_dir) / "capacity_autosave.parquet")
    autosave_path = str(autosave_path)
    autosave_state_path = autosave_path + ".state.json"

    processed_ids: set[str] = set()
    all_rows: list[dict] = []
    last_autosave_n: int = 0

    if autosave_enabled and not resume_from_autosave and overwrite_autosave:
        # Start fresh and overwrite any previous autosave artifacts.
        try:
            Path(autosave_path).unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8 compatibility: missing_ok not available
            p = Path(autosave_path)
            if p.exists():
                p.unlink()
        try:
            Path(autosave_state_path).unlink(missing_ok=True)
        except TypeError:
            p = Path(autosave_state_path)
            if p.exists():
                p.unlink()

    if autosave_enabled and resume_from_autosave:
        p = Path(autosave_path)
        parquet_processed_ids: set[str] = set()
        if p.exists():
            try:
                prev = pd.read_parquet(p)
                if not prev.empty and "customer_id" in prev.columns:
                    processed_ids = set(prev["customer_id"].astype(str).unique())
                    all_rows = prev.to_dict(orient="records")
            except Exception:
                # If autosave can't be read, we fall back to starting from scratch.
                processed_ids = set()
                all_rows = []
        sp = Path(autosave_state_path)
        if sp.exists():
            try:
                state = json.loads(sp.read_text())
                state_ids = set(state.get("processed_ids", list(processed_ids)))
                if parquet_processed_ids and len(parquet_processed_ids) >= len(state_ids):
                    processed_ids = parquet_processed_ids
                else:
                    processed_ids = state_ids
                last_autosave_n = max(
                    int(state.get("processed_n", 0)),
                    len(processed_ids),
                )
            except Exception:
                pass

    total = len(relevant_cids)
    processed = len(processed_ids)
    if last_autosave_n <= 0:
        last_autosave_n = processed

    meteo_for_merge = _build_meteo_merge_frame(avg_meteo_15min)

    def _maybe_autosave():
        if not autosave_enabled:
            return
        if autosave_every_customers <= 0:
            return
        if processed <= 0:
            return
        # Save when we advanced by at least autosave_every_customers since last save.
        nonlocal last_autosave_n
        if processed < (last_autosave_n + autosave_every_customers):
            return
        try:
            df_out = pd.DataFrame(all_rows)
            df_out.to_parquet(autosave_path, index=False)
            state = {
                "processed_ids": sorted(list(processed_ids)),
                "processed_n": int(processed),
            }
            Path(autosave_state_path).write_text(json.dumps(state))
            last_autosave_n = processed
        except Exception:
            # Autosave failure should not kill the run.
            pass

    def _merge_and_features(raw_df: pd.DataFrame) -> tuple:
        """Merge meteo and build daily features for a batch of customers."""
        raw_df = raw_df.copy()
        dt = pd.to_datetime(raw_df["DT_UTC"], utc=True)
        raw_df["DT_UTC"] = dt.dt.tz_convert(None)
        for col in ("CONSO_KWH", "PROD_KWH"):
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].astype("float32")
        merged = raw_df.merge(meteo_for_merge, on="DT_UTC", how="left")
        daily = build_customer_daily_features(merged, daily_weather)
        return merged, daily

    requested_workers = int(max_workers) if max_workers is not None else 1
    if requested_workers < 1:
        requested_workers = 1
    cpu_total = os.cpu_count() or 1
    effective_max_workers = min(requested_workers, max(1, cpu_total - 1), 4)
    effective_batch_size = max(1, int(batch_size) if batch_size is not None else 1)

    def _process_one_customer(
        cid: str,
        grouped_re,
        grouped_daily,
        seed: int,
    ):
        if cid not in prob_lookup.index:
            return None
        try:
            cust_df = grouped_re.get_group(cid)
            cust_days = grouped_daily.get_group(cid)
        except KeyError:
            return None
        has_pv_prob = float(prob_lookup.loc[cid].get("has_pv_prob", 1.0))
        return _process_single_customer(
            cid, has_pv_prob, cust_df, cust_days, n_bootstrap, seed,
        )

    def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _process_batch(cids: list, merged: pd.DataFrame, daily: pd.DataFrame):
        grouped_re = merged.groupby("ID")
        grouped_daily = daily.groupby("ID")
        batch_rows = []
        for cid_chunk in _chunk_list(list(cids), effective_batch_size):
            seeds = {
                cid: int(base_rng.integers(0, 1_000_000))
                for cid in cid_chunk
            }
            if effective_max_workers <= 1 or len(cid_chunk) == 1:
                for cid in cid_chunk:
                    res = _process_one_customer(cid, grouped_re, grouped_daily, seeds[cid])
                    if res is not None:
                        batch_rows.append(res)
                continue

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(effective_max_workers, len(cid_chunk))
            ) as executor:
                futures_map = {
                    executor.submit(
                        _process_one_customer,
                        cid,
                        grouped_re,
                        grouped_daily,
                        seeds[cid],
                    ): cid
                    for cid in cid_chunk
                }
                for future in concurrent.futures.as_completed(futures_map):
                    try:
                        res = future.result()
                        if res is not None:
                            batch_rows.append(res)
                    except Exception as e:
                        print(f"\nCustomer {futures_map[future]} failed: {e}")
        return batch_rows

    print(
        "Streaming capacity settings: "
        f"workers={effective_max_workers} "
        f"(requested={requested_workers}, cpu={cpu_total}), "
        f"batch_size={effective_batch_size}, "
        f"autosave_every={autosave_every_customers}, "
        f"n_bootstrap={n_bootstrap}"
    )

    # Process single-file customers in batches by source file
    file_to_single = {}
    for cid in single_file_cids:
        fpath = cust_file_index[cid][0]
        file_to_single.setdefault(fpath, []).append(cid)

    # Deterministic processing order (important for resume_from_file)
    file_items = sorted(file_to_single.items(), key=lambda x: x[0])
    if resume_from_file is not None:
        # Skip until we reach the specified file path.
        # Accept either basename or full path.
        resume_key = resume_from_file
        started = False
        filtered_items = []
        for fpath, cids_in_file in file_items:
            if started:
                filtered_items.append((fpath, cids_in_file))
                continue
            if fpath == resume_key or Path(fpath).name == Path(resume_key).name:
                started = True
                filtered_items.append((fpath, cids_in_file))
        file_items = filtered_items if started else file_items

    for fpath, cids_in_file in file_items:
        try:
            raw = re_data.load_customer_data(fpath)
            if raw.empty:
                continue
            raw["ID"] = raw["ID"].astype(str)
            # Skip customers already processed (resume)
            cids_in_file = [c for c in cids_in_file if c not in processed_ids]
            if not cids_in_file:
                continue
            raw = raw[raw["ID"].isin(cids_in_file)]
            if raw.empty:
                continue
            merged, daily = _merge_and_features(raw)
            rows = _process_batch(cids_in_file, merged, daily)
            all_rows.extend(rows)
            for r in rows:
                cid = str(r.get("customer_id"))
                if cid:
                    processed_ids.add(cid)
            processed = len(processed_ids)
            _maybe_autosave()
            if _HAS_TQDM:
                print(f"\r  Streaming: {processed}/{total} customers processed", end="", flush=True)
        except Exception as e:
            print(f"\nError processing file {fpath}: {e}")

    # Process multi-file customers (load 2-3 files per customer, resume-friendly)
    if multi_file_cids:
        for cid in sorted(multi_file_cids):
            cid = str(cid)
            if cid in processed_ids:
                continue
            try:
                file_list = cust_file_index.get(cid, [])
                if not file_list:
                    continue
                chunks = []
                for fpath in file_list:
                    raw = re_data.load_customer_data(fpath)
                    if raw.empty:
                        continue
                    raw["ID"] = raw["ID"].astype(str)
                    chunk = raw[raw["ID"] == cid]
                    if not chunk.empty:
                        chunks.append(chunk)
                if not chunks:
                    continue
                combined = pd.concat(chunks, ignore_index=True)
                merged, daily = _merge_and_features(combined)
                rows = _process_batch([cid], merged, daily)
                all_rows.extend(rows)
                for r in rows:
                    ccid = str(r.get("customer_id"))
                    if ccid:
                        processed_ids.add(ccid)
                processed = len(processed_ids)
                _maybe_autosave()
            except Exception as e:
                print(f"\nError processing multi-file customer {cid}: {e}")

    if _HAS_TQDM:
        print(f"\r  Streaming: {processed}/{total} customers processed. Done.")

    out = pd.DataFrame(all_rows)
    if autosave_enabled:
        try:
            out.to_parquet(autosave_path, index=False)
            state = {
                "processed_ids": sorted(list(processed_ids)),
                "processed_n": int(processed),
                "completed": True,
            }
            Path(autosave_state_path).write_text(json.dumps(state))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Portfolio-level plotting
# ---------------------------------------------------------------------------

def plot_portfolio_aggregate_load(
    re_data_with_meteo: pd.DataFrame,
    prob_summary: pd.DataFrame,
    resolution: str = "D",
    show: bool = True,
) -> go.Figure:
    """
    Plot the aggregate (summed) load profile across all PV customers in
    *prob_summary*, resampled to *resolution* (default daily).

    Three traces: Aggregate Import (kW), Aggregate Export (kW), Net (kW).
    """
    pv_ids = set(prob_summary["customer_id"].dropna())
    df = re_data_with_meteo[re_data_with_meteo["ID"].isin(pv_ids)].copy()
    if df.empty:
        raise ValueError("No matching customer data found in re_data_with_meteo.")

    df["Import_kW"] = df["CONSO_KWH"] * 4.0
    df["Export_kW"] = df["PROD_KWH"] * 4.0

    df = df.set_index("DT_UTC")
    agg = df.groupby(df.index)[["Import_kW", "Export_kW"]].sum()
    agg = agg.resample(resolution).mean()
    agg["Net_kW"] = agg["Import_kW"] - agg["Export_kW"]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["Import_kW"],
        mode="lines", name="Aggregate Import (kW)",
        line=dict(color="crimson", width=1.5),
    ))
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["Export_kW"],
        mode="lines", name="Aggregate Export (kW)",
        line=dict(color="royalblue", width=1.5),
        fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
    ))
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["Net_kW"],
        mode="lines", name="Net Load (kW)",
        line=dict(color="grey", width=1, dash="dot"),
    ))

    res_label = {"D": "Daily", "W": "Weekly", "h": "Hourly"}.get(resolution, resolution)
    fig.update_layout(
        title=f"Portfolio Aggregate Load Profile ({res_label} avg, {len(pv_ids)} customers)",
        xaxis_title="Time (UTC)",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
        template="plotly_white",
    )
    if show:
        fig.show()
    return fig


def plot_portfolio_aggregate_load_streaming(
    data_dir: str,
    cust_file_index: dict,
    avg_meteo_15min: pd.DataFrame,
    prob_summary: pd.DataFrame,
    resolution: str = "D",
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Streaming version of plot_portfolio_aggregate_load: loads one file at a
    time so the full dataset never needs to be in memory.
    """
    pv_ids = set(prob_summary["customer_id"].dropna())
    agg = stream_portfolio_aggregate_load(
        data_dir, cust_file_index, avg_meteo_15min, pv_ids, resolution,
    )
    if agg.empty:
        raise ValueError("No matching customer data found.")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["Import_kW"],
        mode="lines", name="Aggregate Import (kW)",
        line=dict(color="crimson", width=1.5),
    ))
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["Export_kW"],
        mode="lines", name="Aggregate Export (kW)",
        line=dict(color="royalblue", width=1.5),
        fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
    ))
    fig.add_trace(go.Scattergl(
        x=agg.index, y=agg["Net_kW"],
        mode="lines", name="Net Load (kW)",
        line=dict(color="grey", width=1, dash="dot"),
    ))

    res_label = {"D": "Daily", "W": "Weekly", "h": "Hourly"}.get(resolution, resolution)
    fig.update_layout(
        title=f"Portfolio Aggregate Load Profile ({res_label} avg, {len(pv_ids)} customers)",
        xaxis_title="Time (UTC)",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
        template="plotly_white",
    )
    if save_dir:
        write_plotly_figures_to_dir({"portfolio_aggregate_load": fig}, save_dir)
    if show:
        fig.show()
    return fig


def plot_portfolio_pv_capacity(
    prob_summary: pd.DataFrame,
    portfolio_agg: dict,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Multi-figure portfolio capacity visualisation.

    Returns a dict with keys:
      * ``capacity_histogram`` -- distribution of individual kWp estimates
      * ``total_capacity_bar`` -- hybrid portfolio total kWp with conservative
                                  (robust) 95% CI
      * ``sc_share_bar``              -- aggregate self-consumption share with CIs
      * ``sc_share_bar_nonzero_sc``   -- same, excluding customers with ``sc_share_mean`` 0
    """
    figs: dict[str, go.Figure] = {}

    # ================================================================
    # Panel A: Individual capacity histogram
    # ================================================================
    df = prob_summary.dropna(subset=["pv_capacity_kwp"])
    df = df[df["pv_capacity_kwp"] > 0].copy()

    dominated_by = np.where(
        df["floor_to_reg_ratio"].fillna(0) > 1.0,
        "Floor-dominated",
        "Regression-dominated",
    )

    fig_hist = go.Figure()
    for label, color in [("Regression-dominated", "teal"), ("Floor-dominated", "orange")]:
        mask = dominated_by == label
        if mask.any():
            fig_hist.add_trace(go.Histogram(
                x=df.loc[mask, "pv_capacity_kwp"],
                name=label,
                marker_color=color,
                opacity=0.75,
                nbinsx=30,
            ))

    mean_kwp = portfolio_agg["capacity_stats"]["mean"]
    median_kwp = portfolio_agg["capacity_stats"]["median"]
    fig_hist.add_vline(x=mean_kwp, line_dash="dash", line_color="black",
                       annotation_text=f"Mean: {mean_kwp:.1f} kWp")
    fig_hist.add_vline(x=median_kwp, line_dash="dot", line_color="grey",
                       annotation_text=f"Median: {median_kwp:.1f} kWp",
                       annotation_position="bottom right")

    fig_hist.update_layout(
        title="Distribution of Individual PV Capacity Estimates",
        xaxis_title="Estimated PV Capacity (kWp)",
        yaxis_title="Number of Customers",
        barmode="stack",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figs["capacity_histogram"] = fig_hist

    # ================================================================
    # Panel B: Hybrid portfolio total with conservative (robust) CI only
    # ================================================================
    total_h = portfolio_agg["total_hybrid_kwp"]
    ci_con = portfolio_agg["ci_conservative"]

    fig_bar = go.Figure()

    err_plus = ci_con[1] - total_h
    err_minus = total_h - ci_con[0]

    fig_bar.add_trace(go.Bar(
        x=["Hybrid forecast"],
        y=[total_h],
        width=0.42,
        marker=dict(color="#2E7D32", line=dict(color="white", width=1)),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[err_plus],
            arrayminus=[err_minus],
            color="#1B5E20",
            thickness=2.2,
            width=8,
        ),
        text=[f"{total_h:.0f} kWp"],
        textposition="outside",
        textfont=dict(size=13, color="#1B1B1B"),
        name="Portfolio total",
        showlegend=False,
    ))

    fig_bar.update_layout(
        title=dict(
            text=(
                f"<b>Total portfolio PV capacity</b>: {total_h:,.0f} kWp"
                f"<br><sup>{portfolio_agg['n_customers']:,} customers · "
                f"Robust 95% CI [{ci_con[0]:,.0f}, {ci_con[1]:,.0f}] kWp</sup>"
            ),
            font=dict(family="Arial, sans-serif", size=15, color="#1B1B1B"),
        ),
        yaxis=dict(
            title=dict(text="Total capacity (kWp)", font=dict(size=13)),
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            showline=True,
            linecolor="rgba(0,0,0,0.25)",
            mirror=False,
            tickfont=dict(size=11),
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=12),
            showline=True,
            linecolor="rgba(0,0,0,0.25)",
        ),
        bargap=0.55,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=56, r=28, t=88, b=56),
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        showlegend=False,
    )
    figs["total_capacity_bar"] = fig_bar

    # ================================================================
    # Panel C: Aggregate self-consumption share
    # ================================================================
    agg_sc = portfolio_agg.get("aggregate_sc_share", np.nan)
    if not np.isnan(agg_sc):
        sc_ci_con = portfolio_agg["aggregate_sc_ci_conservative"]
        sc_ci_ind = portfolio_agg["aggregate_sc_ci_independence"]

        fig_sc = go.Figure()

        fig_sc.add_trace(go.Bar(
            x=["Self-Consumption Share"],
            y=[agg_sc],
            marker_color="mediumpurple",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[sc_ci_ind[1] - agg_sc],
                arrayminus=[agg_sc - sc_ci_ind[0]],
                color="indigo",
                thickness=2,
                width=8,
            ),
            text=[f"{agg_sc:.1%}"],
            textposition="outside",
            name="Capacity-weighted mean",
            width=0.4,
        ))

        fig_sc.add_trace(go.Scatter(
            x=["Self-Consumption Share", "Self-Consumption Share"],
            y=[sc_ci_con[0], sc_ci_con[1]],
            mode="markers+lines",
            marker=dict(symbol="line-ew-open", size=14, color="indigo", line_width=2),
            line=dict(color="indigo", width=1.5, dash="dash"),
            name=f"Conservative 95% CI [{sc_ci_con[0]:.1%}, {sc_ci_con[1]:.1%}]",
        ))

        fig_sc.update_layout(
            title=(
                f"Aggregate Self-Consumption Share: {agg_sc:.1%}"
                f"<br><sup>Capacity-weighted mean across portfolio</sup>"
            ),
            yaxis_title="Self-Consumption Share",
            yaxis_range=[0, min(1.0, sc_ci_con[1] + 0.15)],
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        figs["sc_share_bar"] = fig_sc

    # ================================================================
    # Panel D: Aggregate SC excluding customers with zero point estimate
    # ================================================================
    agg_sc_nz = portfolio_agg.get("aggregate_sc_share_nonzero_sc", np.nan)
    if not np.isnan(agg_sc_nz):
        sc_ci_con_nz = portfolio_agg["aggregate_sc_ci_conservative_nonzero_sc"]
        sc_ci_ind_nz = portfolio_agg["aggregate_sc_ci_independence_nonzero_sc"]
        n_nz = portfolio_agg.get("n_customers_nonzero_sc", 0)
        n_all = portfolio_agg.get("n_customers", 0)

        fig_sc_nz = go.Figure()
        fig_sc_nz.add_trace(go.Bar(
            x=["Self-Consumption Share"],
            y=[agg_sc_nz],
            marker_color="mediumpurple",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[sc_ci_ind_nz[1] - agg_sc_nz],
                arrayminus=[agg_sc_nz - sc_ci_ind_nz[0]],
                color="indigo",
                thickness=2,
                width=8,
            ),
            text=[f"{agg_sc_nz:.1%}"],
            textposition="outside",
            name="Capacity-weighted mean",
            width=0.4,
        ))
        fig_sc_nz.add_trace(go.Scatter(
            x=["Self-Consumption Share", "Self-Consumption Share"],
            y=[sc_ci_con_nz[0], sc_ci_con_nz[1]],
            mode="markers+lines",
            marker=dict(symbol="line-ew-open", size=14, color="indigo", line_width=2),
            line=dict(color="indigo", width=1.5, dash="dash"),
            name=f"Conservative 95% CI [{sc_ci_con_nz[0]:.1%}, {sc_ci_con_nz[1]:.1%}]",
        ))
        fig_sc_nz.update_layout(
            title=(
                f"Aggregate Self-Consumption Share: {agg_sc_nz:.1%}"
                f"<br><sup>Excluding customers with zero estimated self-consumption "
                f"({n_nz:,} of {n_all:,} hybrid-PV customers)</sup>"
            ),
            yaxis_title="Self-Consumption Share",
            yaxis_range=[0, min(1.0, sc_ci_con_nz[1] + 0.15)],
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        figs["sc_share_bar_nonzero_sc"] = fig_sc_nz

    if save_dir:
        write_plotly_figures_to_dir(figs, save_dir)
    if show:
        for fig in figs.values():
            fig.show()

    return figs


# ---------------------------------------------------------------------------
# Evaluation dashboard plots (section 4 of plan)
# ---------------------------------------------------------------------------

def plot_evaluation_dashboard(
    evaluation: dict,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Multi-panel evaluation dashboard:
      1. Yield distribution histogram
      2. Estimator agreement scatter (floor vs regression)
      3. Cross-segment capacity box plot
    """
    figs: dict[str, go.Figure] = {}

    # --- Panel 1: Yield distribution ---
    yield_df = evaluation.get("yield_df")
    if yield_df is not None and not yield_df.empty:
        sy = yield_df.dropna(subset=["specific_yield_kwh_kwp"])
        if not sy.empty:
            fig_yield = go.Figure()
            for flag, color in [
                ("plausible", "seagreen"), ("low", "orange"), ("high", "crimson"), ("no_data", "grey")
            ]:
                mask = sy["yield_flag"] == flag
                if mask.any():
                    fig_yield.add_trace(go.Histogram(
                        x=sy.loc[mask, "specific_yield_kwh_kwp"],
                        name=flag,
                        marker_color=color,
                        opacity=0.75,
                        nbinsx=40,
                    ))

            ys = evaluation.get("yield_stats", {})
            fig_yield.add_vline(
                x=ys.get("median", 0), line_dash="dash", line_color="black",
                annotation_text=f"Median: {ys.get('median', 0):.0f}",
            )
            fig_yield.add_vrect(x0=800, x1=1200, fillcolor="green", opacity=0.08,
                                line_width=0, annotation_text="Plausible range")

            fig_yield.update_layout(
                title="Specific Yield Distribution (kWh/kWp/year)",
                xaxis_title="Specific Yield (kWh/kWp)",
                yaxis_title="Number of Customers",
                barmode="stack",
                template="plotly_white",
            )
            figs["yield_distribution"] = fig_yield

    # --- Panel 2: Estimator agreement scatter ---
    if yield_df is not None and not yield_df.empty:
        valid = yield_df[
            (yield_df["pv_capacity_kwp_regression_only"] > 0)
            & (yield_df["pv_capacity_kwp_floor"] > 0)
        ]
        if not valid.empty:
            fig_agree = px.scatter(
                valid,
                x="pv_capacity_kwp_regression_only",
                y="pv_capacity_kwp_floor",
                color="yield_flag",
                hover_data=["customer_id"],
                title="Estimator Agreement: Floor vs. Regression Capacity",
                labels={
                    "pv_capacity_kwp_regression_only": "Regression Capacity (kWp)",
                    "pv_capacity_kwp_floor": "Export Floor Capacity (kWp)",
                },
            )
            max_val = max(valid["pv_capacity_kwp_regression_only"].quantile(0.99),
                          valid["pv_capacity_kwp_floor"].quantile(0.99))
            fig_agree.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(color="grey", dash="dash"),
                name="1:1 line", showlegend=True,
            ))
            fig_agree.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val * 0.7],
                mode="lines", line=dict(color="lightgrey", dash="dot"),
                name="0.7x (expected floor/reg)", showlegend=True,
            ))
            fig_agree.update_layout(template="plotly_white")
            figs["estimator_agreement"] = fig_agree

    # --- Panel 3: Cross-segment box plot ---
    flagged_df = evaluation.get("flagged_df")
    if flagged_df is not None and "segment" in flagged_df.columns:
        seg_data = flagged_df[
            flagged_df["pv_capacity_kwp"].notna() & (flagged_df["pv_capacity_kwp"] > 0)
        ]
        if not seg_data.empty and seg_data["segment"].nunique() > 1:
            fig_seg = px.box(
                seg_data,
                x="segment",
                y="pv_capacity_kwp",
                color="segment",
                title="PV Capacity Distribution by Customer Segment",
                labels={
                    "pv_capacity_kwp": "Estimated PV Capacity (kWp)",
                    "segment": "Segment",
                },
            )
            fig_seg.update_layout(
                template="plotly_white",
                showlegend=False,
                yaxis_range=[0, seg_data["pv_capacity_kwp"].quantile(0.98) * 1.1],
            )
            figs["cross_segment"] = fig_seg

    if save_dir:
        write_plotly_figures_to_dir(figs, save_dir)
    if show:
        for fig in figs.values():
            fig.show()

    return figs


#%%

def main():
    data_dir = str(
        Path(__file__).resolve().parent.parent / "data" / "re_data" / "ETHZ"
    )

    # Phase 0: Setup
    print("Loading meteo data...")
    combined_meteo, avg_meteo_15min = load_meteo_data()
    print("Computing daily weather features...")
    daily_weather = compute_daily_weather(avg_meteo_15min)
    print("Loading customer metadata (Particuliers)...")
    metadata = load_customer_metadata(data_dir)
    particulier_ids = particuliers_customer_ids(metadata)
    print(f"  {len(particulier_ids):,} Particuliers IDs in metadata")

    # Phase 1: Streaming summary + indicators
    print("Streaming customer summary (Particuliers, <= 100 MWh total CONSO)...")
    customer_summary, target_ids = stream_customer_summary(
        data_dir,
        max_cons_kwh=100_000,
        allowed_ids=particulier_ids,
    )
    print(f"  {len(customer_summary)} Particuliers in scan, {len(target_ids)} within consumption cap")

    print("Building customer-file index (restricted to eligible IDs)...")
    cust_file_index = build_customer_file_index(data_dir, restrict_to_ids=target_ids)
    print(f"  {len(cust_file_index)} customers indexed")

    print("Streaming PV indicators...")
    pv_indicators = stream_pv_indicators(
        data_dir, target_ids, avg_meteo_15min, daily_weather, cust_file_index,
    )
    pv_indicators = classify_pv_customers(pv_indicators)

    valid_mask = (pv_indicators["DeltaProd"] >= 0) & ~(
        (pv_indicators["yearly_prod"] > 5000)
        & (pv_indicators["corr_prod_rad"] < 0.4)
    )
    pv_indicators_clean = pv_indicators[valid_mask].copy()
    dropped = len(pv_indicators) - len(pv_indicators_clean)
    if dropped > 0:
        print(f"  Dropped {dropped} anomalous customers")

    # Phase 2: Streaming capacity estimation
    print("Streaming capacity estimation...")
    prob_summary = process_customers_streaming(
        data_dir,
        avg_meteo_15min,
        pv_indicators_clean,
        n_bootstrap=200,
        max_workers=3,
        batch_size=100,
        daily_weather=daily_weather,
        cust_file_index=cust_file_index,
    )

    print("Probabilistic capacity summary head:")
    print(prob_summary.head())

    # Portfolio-level aggregation
    print("Aggregating portfolio estimates...")
    portfolio_agg = aggregate_portfolio_estimates(prob_summary, pv_indicators_clean)
    if portfolio_agg:
        print(f"  Total hybrid capacity : {portfolio_agg['total_hybrid_kwp']:.1f} kWp")
        print(f"  Independence 95% CI   : [{portfolio_agg['ci_independence'][0]:.1f}, {portfolio_agg['ci_independence'][1]:.1f}] kWp")
        print(f"  Conservative 95% CI   : [{portfolio_agg['ci_conservative'][0]:.1f}, {portfolio_agg['ci_conservative'][1]:.1f}] kWp")
        print(f"  Aggregate SC share    : {portfolio_agg['aggregate_sc_share']:.1%}")

    # Evaluation framework
    print("Running evaluation framework...")
    evaluation = evaluate_portfolio(
        prob_summary,
        pv_indicators_clean,
        metadata=metadata,
        segment_col="TYPE_PARTENAIRE_LIBELLE",
    )
    print_evaluation_report(evaluation)

    # Plotting
    plot_population_statistics(pv_indicators_clean, show=True)
    plot_capacity_vs_production_with_ci(prob_summary, pv_indicators_clean, show=True)
    plot_capacity_vs_self_consumption(prob_summary, show=True)

    if not prob_summary.empty:
        plot_portfolio_aggregate_load_streaming(
            data_dir, cust_file_index, avg_meteo_15min,
            prob_summary, show=True,
        )
        if portfolio_agg:
            plot_portfolio_pv_capacity(prob_summary, portfolio_agg, show=True)

        plot_evaluation_dashboard(evaluation, show=True)

        massive_customer_id = prob_summary.loc[
            prob_summary["pv_capacity_kwp"].idxmax(), "customer_id"
        ]
        cust_data = load_single_customer(
            massive_customer_id, data_dir, cust_file_index, avg_meteo_15min,
        )
        if not cust_data.empty:
            plot_customer_capacity_validation(
                customer_id=massive_customer_id,
                re_data_with_meteo=cust_data,
                prob_summary=prob_summary,
            )

# MANDATORY MULTIPROCESSING GUARD
if __name__ == "__main__":
    main()

#%%



# %%
# %%

# Plot their profile

# %%

def plot_yearly_customer_capacity(
    customer_id: str,
    re_data_with_meteo: pd.DataFrame,
    prob_summary: pd.DataFrame,
    show: bool = True
):
    """
    Plots a customer's Import/Export Power (kW) for the ENTIRE YEAR,
    overlaid with their estimated PV Capacity and CI.
    Uses WebGL (Scattergl) to prevent browser crashing with large data.
    """
    # 1. Fetch capacity metrics
    cust_summary = prob_summary[prob_summary["customer_id"] == customer_id]
    if cust_summary.empty:
        print(f"Skipping {customer_id}: Not found in summary table.")
        return

    cap_kwp = cust_summary["pv_capacity_kwp"].values[0]
    ci_lower = cust_summary["pv_capacity_kwp_ci_lower"].values[0]
    ci_upper = cust_summary["pv_capacity_kwp_ci_upper"].values[0]
    cap_reg = cust_summary.get("pv_capacity_kwp_regression_only", pd.Series([np.nan])).values[0]
    cap_floor = cust_summary.get("pv_capacity_kwp_floor", pd.Series([np.nan])).values[0]

    # 2. Fetch data and convert Energy (kWh) to Power (kW)
    df = re_data_with_meteo[re_data_with_meteo["ID"] == customer_id].copy()
    if df.empty:
        return
        
    df["Import_kW"] = df["CONSO_KWH"] * 4
    df["Export_kW"] = df["PROD_KWH"] * 4

    # 3. Build Figure using WebGL
    fig = go.Figure()

    # Plot Grid Import (Consumption)
    fig.add_trace(
        go.Scattergl(
            x=df['DT_UTC'], 
            y=df['Import_kW'], 
            mode='lines', 
            name='Grid Import (kW)', 
            line=dict(color='red', width=1),
            opacity=0.6 
        )
    )

    # Plot Solar Export (Production)
    fig.add_trace(
        go.Scattergl(
            x=df['DT_UTC'], 
            y=df['Export_kW'], 
            mode='lines', 
            name='Solar Export (kW)', 
            line=dict(color='royalblue', width=1),
            opacity=0.6
        )
    )

    # Draw the hybrid Estimated Capacity (kWp) as a horizontal dashed ceiling
    fig.add_hline(
        y=cap_kwp, 
        line_dash="dash", 
        line_color="green", 
        line_width=2.5,
        annotation_text=f" Hybrid PV Capacity: {cap_kwp:.2f} kWp ",
        annotation_position="top left",
        annotation_font=dict(color="green", size=14), # FIXED: removed bgcolor from here
        annotation_bgcolor="white"                    # FIXED: added it directly to the hline
    )

    # Draw the 95% Confidence Interval
    if pd.notna(ci_lower) and pd.notna(ci_upper):
        fig.add_hrect(
            y0=ci_lower, 
            y1=ci_upper, 
            line_width=0, 
            fillcolor="green", 
            opacity=0.2,
            annotation_text=" 95% CI ",
            annotation_position="bottom left"
        )

    # Optional: overlay regression-only and floor capacities if present
    if pd.notna(cap_reg):
        fig.add_hline(
            y=cap_reg,
            line_dash="dot",
            line_color="darkgreen",
            line_width=1.5,
            annotation_text=f" Regression-only: {cap_reg:.2f} kWp ",
            annotation_position="top right",
            annotation_font=dict(color="darkgreen", size=11),
        )
    if pd.notna(cap_floor) and cap_floor > 0:
        fig.add_hline(
            y=cap_floor,
            line_dash="dash",
            line_color="orange",
            line_width=1.5,
            annotation_text=f" Physical floor: {cap_floor:.2f} kWp ",
            annotation_position="bottom left",
            annotation_font=dict(color="orange", size=11),
        )

    # Clean up layout
    fig.update_layout(
        title=f"Full Year Profile vs. Estimated PV Capacity<br><sup>Customer: {customer_id}</sup>",
        xaxis_title="Time (UTC)",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
        template="plotly_white"
    )

    if show:
        fig.show()

# %%
