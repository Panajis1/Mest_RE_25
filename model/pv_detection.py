#%%

import concurrent.futures
import sys
from pathlib import Path
from typing import Optional, Union

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


def load_re_data():
    """
    Load Romande Energie smart meter data and compute a filtered subset
    of customers with annual consumption <= 10 MWh.
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

    # Focus on customers with <= 10 MWh total consumption
    ids = customer_summary[customer_summary["sum_cons_kwh"] <= 10000][["ID"]]
    re_data_df_small = re_data_df[re_data_df["ID"].isin(ids["ID"])]
    return re_data_df, re_data_df_small, customer_summary


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


def build_daily_features(
    avg_meteo_15min: pd.DataFrame, re_data_with_meteo: pd.DataFrame
):
    """
    Compute daily and midday aggregates for weather and smart-meter data and
    merge them into a tidy (customer, date) table, including a radiation bucket.
    """
    if avg_meteo_15min.empty or re_data_with_meteo.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- Weather daily features ---
    meteo = avg_meteo_15min.copy()
    meteo = meteo.sort_index()
    meteo = meteo.reset_index().rename(columns={"index": "timestamp"})
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

    # Add monthly radiation buckets (low / medium / high)
    daily_weather_index = pd.to_datetime(daily_weather.index)
    daily_weather = daily_weather.copy()
    daily_weather["month"] = daily_weather_index.to_period("M")

    # Compute month-wise quantiles on G_midday; handle missing G_midday robustly
    def month_low_q(x):
        return x.quantile(0.2)

    def month_high_q(x):
        return x.quantile(0.8)

    low_q = (
        daily_weather.groupby("month")["G_midday"]
        .transform(month_low_q)
    )
    high_q = (
        daily_weather.groupby("month")["G_midday"]
        .transform(month_high_q)
    )

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

    # Keep a clean index for merging
    daily_weather = daily_weather.reset_index().rename(columns={"index": "date"})

    # --- Customer daily features ---
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

    # Merge in daily weather features
    daily_features = daily_cust.merge(
        daily_weather, on="date", how="left"
    )

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


def plot_population_statistics(pv_indicators: pd.DataFrame, show: bool = True):
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

    return figs


def plot_capacity_vs_production_with_ci(
    prob_summary: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    show: bool = True,
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
    if show:
        fig.show()
    return fig


def plot_capacity_vs_self_consumption(
    prob_summary: pd.DataFrame, show: bool = True
) -> go.Figure:
    """
    Scatter: estimated PV capacity (kWp) vs self-consumption share (0–1).
    """
    df = prob_summary.dropna(subset=["pv_capacity_kwp", "sc_share_mean"])
    fig = px.scatter(
        df,
        x="pv_capacity_kwp",
        y="sc_share_mean",
        hover_data=["customer_id"],
        title="System Size vs. Self-Consumption Share",
        labels={
            "pv_capacity_kwp": "Estimated PV Capacity (kWp)",
            "sc_share_mean": "Self-Consumption Share (0.0 - 1.0)",
        },
    )
    fig.update_yaxes(range=[0, 1])
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
    Parallel version of compute_probabilistic_capacity using ProcessPoolExecutor.
    Processes customers in batches to limit peak RAM usage, and fetches
    per-customer data on the fly via groupby.get_group.
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

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
# Portfolio-level aggregation
# ---------------------------------------------------------------------------

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
    weights = sc_valid["pv_capacity_kwp"]
    w_sum = weights.sum()

    if w_sum > 0 and not sc_valid.empty:
        agg_sc = float((weights * sc_valid["sc_share_mean"]).sum() / w_sum)

        # SC CI -- conservative (capacity-weighted mean of CI endpoints)
        sc_lo = sc_valid["sc_share_ci_lower"].fillna(sc_valid["sc_share_mean"])
        sc_hi = sc_valid["sc_share_ci_upper"].fillna(sc_valid["sc_share_mean"])
        agg_sc_ci_conservative = (
            float(np.clip((weights * sc_lo).sum() / w_sum, 0, 1)),
            float(np.clip((weights * sc_hi).sum() / w_sum, 0, 1)),
        )

        # SC CI -- independence (delta method for weighted mean)
        sc_sigma_i = (sc_hi - sc_lo) / (2.0 * 1.96)
        # Var(weighted_mean) = sum(w_i^2 * sigma_i^2) / (sum(w_i))^2
        sc_sigma_agg = float(np.sqrt(((weights ** 2) * (sc_sigma_i ** 2)).sum()) / w_sum)
        agg_sc_ci_independence = (
            float(np.clip(agg_sc - 1.96 * sc_sigma_agg, 0, 1)),
            float(np.clip(agg_sc + 1.96 * sc_sigma_agg, 0, 1)),
        )
    else:
        agg_sc = np.nan
        agg_sc_ci_conservative = (np.nan, np.nan)
        agg_sc_ci_independence = (np.nan, np.nan)

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
    Returns DataFrame with at least columns [ID, customer_type] or None.
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
# Streaming data loader (section 5b of plan)
# ---------------------------------------------------------------------------

def _build_customer_file_index(data_dir: str) -> dict:
    """
    First pass: build {customer_id: [file_paths]} index.
    Most customers appear in 1 file (median=1, max~3).
    """
    from collections import defaultdict
    files = re_data.get_parquet_files(data_dir)
    cust_to_files: dict[str, list] = defaultdict(list)
    for fpath in files:
        try:
            df = pd.read_parquet(fpath, columns=["ID"])
            for cid in df["ID"].astype(str).unique():
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
) -> pd.DataFrame:
    """
    Stream-process customers file-by-file instead of loading all into RAM.

    Two-pass approach:
      1. Build ID→files index (lightweight, columns=["ID"] only)
      2. Process file-by-file; single-file customers immediately,
         multi-file customers accumulated and processed after last file.
    """
    cust_file_index = _build_customer_file_index(data_dir)
    if not cust_file_index:
        return pd.DataFrame()

    indicator_ids = set(pv_indicators["customer_id"].unique())
    relevant_cids = [c for c in cust_file_index if c in indicator_ids]

    prob_lookup = pv_indicators.set_index("customer_id")
    base_rng = np.random.default_rng(random_state)

    single_file_cids = [c for c in relevant_cids if len(cust_file_index[c]) == 1]
    multi_file_cids = [c for c in relevant_cids if len(cust_file_index[c]) > 1]

    all_rows: list[dict] = []
    total = len(relevant_cids)
    processed = 0

    # Prepare meteo merge frame
    merge_cols = ["global_rad_W"]
    if "t_2m_C" in avg_meteo_15min.columns:
        merge_cols.append("t_2m_C")
    meteo_for_merge = (
        avg_meteo_15min[merge_cols]
        .rename_axis("DT_UTC")
        .reset_index()
    )

    def _merge_and_features(raw_df: pd.DataFrame) -> tuple:
        """Merge meteo and build daily features for a batch of customers."""
        raw_df = raw_df.copy()
        dt = pd.to_datetime(raw_df["DT_UTC"], utc=True)
        raw_df["DT_UTC"] = dt.dt.tz_convert(None)
        merged = raw_df.merge(meteo_for_merge, on="DT_UTC", how="left")
        daily, _ = build_daily_features(avg_meteo_15min, merged)
        return merged, daily

    def _process_batch(cids: list, merged: pd.DataFrame, daily: pd.DataFrame):
        grouped_re = merged.groupby("ID")
        grouped_daily = daily.groupby("ID")
        batch_rows = []
        for cid in cids:
            try:
                cust_df = grouped_re.get_group(cid)
                cust_days = grouped_daily.get_group(cid)
            except KeyError:
                continue
            if cid not in prob_lookup.index:
                continue
            has_pv_prob = float(prob_lookup.loc[cid].get("has_pv_prob", 1.0))
            seed = int(base_rng.integers(0, 1_000_000))
            res = _process_single_customer(
                cid, has_pv_prob, cust_df, cust_days, n_bootstrap, seed,
            )
            if res is not None:
                batch_rows.append(res)
        return batch_rows

    # Process single-file customers in batches by source file
    file_to_single = {}
    for cid in single_file_cids:
        fpath = cust_file_index[cid][0]
        file_to_single.setdefault(fpath, []).append(cid)

    for fpath, cids_in_file in file_to_single.items():
        try:
            raw = re_data.load_customer_data(fpath)
            if raw.empty:
                continue
            raw["ID"] = raw["ID"].astype(str)
            raw = raw[raw["ID"].isin(cids_in_file)]
            if raw.empty:
                continue
            merged, daily = _merge_and_features(raw)
            rows = _process_batch(cids_in_file, merged, daily)
            all_rows.extend(rows)
            processed += len(cids_in_file)
            if _HAS_TQDM:
                print(f"\r  Streaming: {processed}/{total} customers processed", end="", flush=True)
        except Exception as e:
            print(f"\nError processing file {fpath}: {e}")

    # Process multi-file customers: accumulate across files, then process
    if multi_file_cids:
        multi_accum: dict[str, list] = {c: [] for c in multi_file_cids}
        all_multi_files = set()
        for c in multi_file_cids:
            all_multi_files.update(cust_file_index[c])

        for fpath in all_multi_files:
            try:
                raw = re_data.load_customer_data(fpath)
                if raw.empty:
                    continue
                raw["ID"] = raw["ID"].astype(str)
                for cid in multi_file_cids:
                    chunk = raw[raw["ID"] == cid]
                    if not chunk.empty:
                        multi_accum[cid].append(chunk)
            except Exception:
                continue

        for cid, chunks in multi_accum.items():
            if not chunks:
                continue
            combined = pd.concat(chunks, ignore_index=True)
            try:
                merged, daily = _merge_and_features(combined)
                rows = _process_batch([cid], merged, daily)
                all_rows.extend(rows)
            except Exception as e:
                print(f"\nError processing multi-file customer {cid}: {e}")
            processed += 1

    if _HAS_TQDM:
        print(f"\r  Streaming: {processed}/{total} customers processed. Done.")

    return pd.DataFrame(all_rows)


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


def plot_portfolio_pv_capacity(
    prob_summary: pd.DataFrame,
    portfolio_agg: dict,
    show: bool = True,
) -> dict:
    """
    Multi-figure portfolio capacity visualisation.

    Returns a dict with keys:
      * ``capacity_histogram`` -- distribution of individual kWp estimates
      * ``total_capacity_bar`` -- total portfolio kWp with dual CI bands
                                  and regression/floor breakdown
      * ``sc_share_bar``       -- aggregate self-consumption share with CIs
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
    # Panel B: Total portfolio capacity with CI bands and breakdown
    # ================================================================
    total_h = portfolio_agg["total_hybrid_kwp"]
    total_r = portfolio_agg["total_regression_kwp"]
    total_f = portfolio_agg["total_floor_kwp"]
    ci_con = portfolio_agg["ci_conservative"]
    ci_ind = portfolio_agg["ci_independence"]

    fig_bar = go.Figure()

    categories = ["Hybrid (portfolio)", "Regression-only", "Physical Floor"]
    values = [total_h, total_r, total_f]
    colors = ["seagreen", "teal", "orange"]

    # Error bars only on the hybrid bar (independence CI)
    err_plus = [ci_ind[1] - total_h, 0, 0]
    err_minus = [total_h - ci_ind[0], 0, 0]

    fig_bar.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        error_y=dict(
            type="data",
            symmetric=False,
            array=err_plus,
            arrayminus=err_minus,
            color="darkgreen",
            thickness=2,
            width=6,
        ),
        text=[f"{v:.0f} kWp" for v in values],
        textposition="outside",
        name="Point estimate",
    ))

    # Add the conservative CI as a wider semi-transparent error overlay
    fig_bar.add_trace(go.Scatter(
        x=["Hybrid (portfolio)", "Hybrid (portfolio)"],
        y=[ci_con[0], ci_con[1]],
        mode="markers+lines",
        marker=dict(symbol="line-ew-open", size=14, color="darkgreen", line_width=2),
        line=dict(color="darkgreen", width=1.5, dash="dash"),
        name=f"Conservative 95% CI [{ci_con[0]:.0f}, {ci_con[1]:.0f}]",
        showlegend=True,
    ))

    fig_bar.update_layout(
        title=(
            f"Total Portfolio PV Capacity: {total_h:.0f} kWp"
            f"<br><sup>{portfolio_agg['n_customers']} customers | "
            f"Indep. CI [{ci_ind[0]:.0f}, {ci_ind[1]:.0f}] | "
            f"Conserv. CI [{ci_con[0]:.0f}, {ci_con[1]:.0f}]</sup>"
        ),
        yaxis_title="Total Capacity (kWp)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
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

    if show:
        for fig in figs.values():
            fig.show()

    return figs


#%%

def main():
    print("Loading data...")
    combined_meteo, avg_meteo_15min = load_meteo_data()
    re_data_df, re_data_df_small, customer_summary = load_re_data()
    # Filter to only the biggest 100 customers by total consumption
    

    # Downcast to save RAM
    print("Downcasting data types...")
    re_data_df_small["CONSO_KWH"] = pd.to_numeric(
        re_data_df_small["CONSO_KWH"], downcast="float"
    )
    re_data_df_small["PROD_KWH"] = pd.to_numeric(
        re_data_df_small["PROD_KWH"], downcast="float"
    )
    if not avg_meteo_15min.empty and "global_rad_W" in avg_meteo_15min.columns:
        avg_meteo_15min["global_rad_W"] = pd.to_numeric(
            avg_meteo_15min["global_rad_W"], downcast="float"
        )

    print("Aligning data and building features...")
    meteo_window, re_data_with_meteo = align_meteo_with_re_data(
        avg_meteo_15min, re_data_df_small
    )
    daily_features, daily_weather = build_daily_features(
        meteo_window, re_data_with_meteo
    )
    
    print("Calculating PV indicators...")
    pv_indicators = compute_pv_indicators(daily_features, re_data_with_meteo)
    pv_indicators = classify_pv_customers(pv_indicators)

    # Drop anomalies before bootstrapping
    valid_mask = (pv_indicators["DeltaProd"] >= 0) & ~(
        (pv_indicators["yearly_prod"] > 5000) & (pv_indicators["corr_prod_rad"] < 0.4)
    )
    pv_indicators_clean = pv_indicators[valid_mask].copy()
    dropped = len(pv_indicators) - len(pv_indicators_clean)
    if dropped > 0:
        print(f"Dropped {dropped} anomalous customers before bootstrapping.")

    # Run Parallel Bootstrap
    prob_summary = compute_probabilistic_capacity_parallel(
        re_data_with_meteo,
        daily_features,
        pv_indicators_clean,
        n_bootstrap=200,
        max_workers=3, # Safely limited to 4
        batch_size=100
    )

    print("Probabilistic capacity summary head:")
    print(prob_summary.head())

    # --- Portfolio-level aggregation ---
    print("Aggregating portfolio estimates...")
    portfolio_agg = aggregate_portfolio_estimates(prob_summary, pv_indicators_clean)
    if portfolio_agg:
        print(f"  Total hybrid capacity : {portfolio_agg['total_hybrid_kwp']:.1f} kWp")
        print(f"  Independence 95% CI   : [{portfolio_agg['ci_independence'][0]:.1f}, {portfolio_agg['ci_independence'][1]:.1f}] kWp")
        print(f"  Conservative 95% CI   : [{portfolio_agg['ci_conservative'][0]:.1f}, {portfolio_agg['ci_conservative'][1]:.1f}] kWp")
        print(f"  Aggregate SC share    : {portfolio_agg['aggregate_sc_share']:.1%}")

    # --- Evaluation framework ---
    print("Running evaluation framework...")
    metadata = load_customer_metadata()
    evaluation = evaluate_portfolio(
        prob_summary, pv_indicators_clean, metadata=metadata,
    )
    print_evaluation_report(evaluation)

    # --- Plotting ---
    plot_population_statistics(pv_indicators_clean, show=True)
    plot_capacity_vs_production_with_ci(prob_summary, pv_indicators_clean, show=True)
    plot_capacity_vs_self_consumption(prob_summary, show=True)

    if not prob_summary.empty:
        plot_portfolio_aggregate_load(re_data_with_meteo, prob_summary, show=True)
        if portfolio_agg:
            plot_portfolio_pv_capacity(prob_summary, portfolio_agg, show=True)

        plot_evaluation_dashboard(evaluation, show=True)

        massive_customer_id = prob_summary.loc[prob_summary["pv_capacity_kwp"].idxmax(), "customer_id"]
        plot_customer_capacity_validation(
            customer_id=massive_customer_id, 
            re_data_with_meteo=re_data_with_meteo, 
            prob_summary=prob_summary
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
