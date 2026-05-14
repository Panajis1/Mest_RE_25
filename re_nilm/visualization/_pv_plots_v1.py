"""In-package visualization helpers replacing legacy pv_detection plot wrappers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as _exc:  # pragma: no cover
    px = None
    go = None
    make_subplots = None
    _PLOTLY_IMPORT_ERROR = _exc
else:
    _PLOTLY_IMPORT_ERROR = None

# Shared palette — matches the matplotlib battery summary plots.
_RED = "#c9252b"
_BLACK = "#111111"


def _require_plotly():
    if px is None or go is None:
        raise ImportError(f"plotly not importable: {_PLOTLY_IMPORT_ERROR}")


def plot_customer_timeseries(
    customer_ts: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
):
    _require_plotly()
    df = customer_ts.copy()
    df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
    df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")
    fig = go.Figure()
    if "CONSO_KWH" in df.columns:
        fig.add_trace(go.Scatter(x=df["DT_UTC"], y=df["CONSO_KWH"], mode="lines", name="CONSO_KWH"))
    if "PROD_KWH" in df.columns:
        fig.add_trace(go.Scatter(x=df["DT_UTC"], y=df["PROD_KWH"], mode="lines", name="PROD_KWH"))
    if weather_df is not None and "dt_utc" in weather_df.columns and "global_rad_W" in weather_df.columns:
        w = weather_df.copy()
        w["dt_utc"] = pd.to_datetime(w["dt_utc"], errors="coerce")
        w = w.dropna(subset=["dt_utc"]).sort_values("dt_utc")
        fig.add_trace(go.Scatter(x=w["dt_utc"], y=w["global_rad_W"], mode="lines", name="global_rad_W", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="global_rad_W"))
    fig.update_layout(title=title or "Customer Timeseries", xaxis_title="UTC", yaxis_title="kWh/15min")
    return fig


def plot_customer_capacity_validation(
    customer_ts: pd.DataFrame,
    pv_capacity_kwp: float,
    weather_df: pd.DataFrame,
    title: Optional[str] = None,
):
    _require_plotly()
    df = customer_ts.copy()
    df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
    df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")

    w = weather_df.copy()
    w["dt_utc"] = pd.to_datetime(w["dt_utc"], errors="coerce")
    w = w.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    merged = pd.merge_asof(
        df.rename(columns={"DT_UTC": "dt_utc"}),
        w[["dt_utc", "global_rad_W"]],
        on="dt_utc",
        direction="nearest",
        tolerance=pd.Timedelta("16min"),
    )
    pred_kwh = ((merged["global_rad_W"].clip(lower=0) / 1000.0) * pv_capacity_kwp * 0.15) * 0.25
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["dt_utc"], y=merged.get("PROD_KWH", pd.Series(dtype=float)), mode="lines", name="Measured PROD_KWH"))
    fig.add_trace(go.Scatter(x=merged["dt_utc"], y=pred_kwh, mode="lines", name="Modelled PV"))
    fig.update_layout(title=title or "Capacity Validation", xaxis_title="UTC", yaxis_title="kWh/15min")
    return fig


def plot_customer_high_low_profile(
    customer_ts: pd.DataFrame,
    weather_df: pd.DataFrame,
    title: Optional[str] = None,
):
    _require_plotly()
    df = customer_ts.copy()
    df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
    df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")
    w = weather_df.copy()
    w["dt_utc"] = pd.to_datetime(w["dt_utc"], errors="coerce")
    w = w.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    merged = pd.merge_asof(
        df.rename(columns={"DT_UTC": "dt_utc"}),
        w[["dt_utc", "global_rad_W"]],
        on="dt_utc",
        direction="nearest",
        tolerance=pd.Timedelta("16min"),
    )
    merged["date"] = merged["dt_utc"].dt.date
    day_rad = merged.groupby("date")["global_rad_W"].mean().dropna()
    if day_rad.empty:
        return go.Figure()
    lo, hi = day_rad.quantile(0.25), day_rad.quantile(0.75)
    high_days = set(day_rad[day_rad >= hi].index)
    low_days = set(day_rad[day_rad <= lo].index)
    merged["hour"] = merged["dt_utc"].dt.hour
    high_prof = merged[merged["date"].isin(high_days)].groupby("hour")["CONSO_KWH"].mean()
    low_prof = merged[merged["date"].isin(low_days)].groupby("hour")["CONSO_KWH"].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=high_prof.index, y=high_prof.values, mode="lines+markers", name="High-radiation days"))
    fig.add_trace(go.Scatter(x=low_prof.index, y=low_prof.values, mode="lines+markers", name="Low-radiation days"))
    fig.update_layout(title=title or "High/Low Radiation Profiles", xaxis_title="Hour", yaxis_title="CONSO_KWH")
    return fig


def plot_population_statistics(
    results: pd.DataFrame,
    title: str = "Portfolio PV Statistics",
):
    _require_plotly()
    df = results.copy()
    cols = [c for c in ["pv_capacity_kwp", "sc_share"] if c in df.columns]
    if not cols:
        return go.Figure()
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
    for i, c in enumerate(cols, start=1):
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=c,
                nbinsx=40,
                marker=dict(color=_RED, line=dict(color=_BLACK, width=0.5)),
            ),
            row=1,
            col=i,
        )
    fig.update_layout(title=title, showlegend=False)
    return fig


def plot_capacity_vs_production_with_ci(
    results: pd.DataFrame,
    title: str = "PV Capacity vs Annual Production",
):
    _require_plotly()
    df = results.copy()
    y_col = "yearly_prod" if "yearly_prod" in df.columns else "yearly_prod_kwh"
    if "pv_capacity_kwp" not in df.columns or y_col not in df.columns:
        return go.Figure()
    err_plus = None
    err_minus = None
    if "pv_ci_upper" in df.columns and "pv_ci_lower" in df.columns:
        err_plus = (df["pv_ci_upper"] - df["pv_capacity_kwp"]).clip(lower=0)
        err_minus = (df["pv_capacity_kwp"] - df["pv_ci_lower"]).clip(lower=0)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["pv_capacity_kwp"],
                y=df[y_col],
                mode="markers",
                marker=dict(color=_RED, line=dict(color=_BLACK, width=0.3), opacity=0.55),
                error_x=dict(
                    type="data",
                    array=err_plus,
                    arrayminus=err_minus,
                    visible=err_plus is not None,
                    color=_BLACK,
                ),
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="pv_capacity_kwp", yaxis_title=y_col)
    return fig


def plot_customer_heatmap(
    customer_ts: pd.DataFrame,
    col: str = "CONSO_KWH",
    title: Optional[str] = None,
):
    _require_plotly()
    df = customer_ts.copy()
    df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
    df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")
    df["date"] = df["DT_UTC"].dt.date
    df["quarter"] = df["DT_UTC"].dt.hour * 4 + (df["DT_UTC"].dt.minute // 15)
    pvt = df.pivot_table(index="date", columns="quarter", values=col, aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(z=pvt.values, x=pvt.columns, y=pvt.index))
    fig.update_layout(title=title or f"{col} Heatmap", xaxis_title="Quarter of day", yaxis_title="Date")
    return fig


def plot_evaluation_dashboard(
    results: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    title: str = "Pipeline Evaluation Dashboard",
):
    _require_plotly()
    y_col = "yearly_prod" if "yearly_prod" in results.columns else "yearly_prod_kwh"
    if y_col not in results.columns and "yearly_prod" in pv_indicators.columns:
        merged = results.merge(pv_indicators[["customer_id", "yearly_prod"]], on="customer_id", how="left")
        y_col = "yearly_prod"
    else:
        merged = results.copy()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Capacity distribution", "SC share distribution", "Capacity vs production", "Flags"],
    )
    if "pv_capacity_kwp" in merged.columns:
        fig.add_trace(go.Histogram(x=merged["pv_capacity_kwp"], nbinsx=40, name="capacity"), row=1, col=1)
    if "sc_share" in merged.columns:
        fig.add_trace(go.Histogram(x=merged["sc_share"], nbinsx=40, name="sc_share"), row=1, col=2)
    if "pv_capacity_kwp" in merged.columns and y_col in merged.columns:
        fig.add_trace(go.Scatter(x=merged["pv_capacity_kwp"], y=merged[y_col], mode="markers", name="cap_vs_prod"), row=2, col=1)
    if "capacity_flag" in merged.columns:
        counts = merged["capacity_flag"].fillna("unknown").value_counts()
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, name="capacity_flag"), row=2, col=2)
    elif "yield_flag" in merged.columns:
        counts = merged["yield_flag"].fillna("unknown").value_counts()
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, name="yield_flag"), row=2, col=2)
    fig.update_layout(title=title, showlegend=False)
    return fig
