"""Per-customer visualizations — delegates to pv_detection.py plot functions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_PV_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_PV_DIR) not in sys.path:
    sys.path.insert(0, str(_PV_DIR))

try:
    from pv_detection import (
        plot_customer_timeseries as _plot_timeseries,
        plot_customer_capacity_validation as _plot_capacity_validation,
        plot_customer_high_low_profile as _plot_high_low,
        plot_customer_heatmap as _plot_heatmap,
    )
    _PV_AVAILABLE = True
except ImportError:
    _PV_AVAILABLE = False


def _require_pv():
    if not _PV_AVAILABLE:
        raise ImportError("pv_detection.py not importable — cannot render customer plots")


def plot_customer_timeseries(
    customer_ts: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
):
    """Plot 15-min consumption and production time series for a single customer.

    Args:
        customer_ts: DataFrame with [DT_UTC, CONSO_KWH, PROD_KWH].
        weather_df: Optional weather data for radiation overlay.
        title: Plot title; defaults to customer ID.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_timeseries(customer_ts, weather_df=weather_df, title=title)


def plot_customer_capacity_validation(
    customer_ts: pd.DataFrame,
    pv_capacity_kwp: float,
    weather_df: pd.DataFrame,
    title: Optional[str] = None,
):
    """Plot measured vs. modelled PV generation to validate capacity estimate.

    Args:
        customer_ts: DataFrame with [DT_UTC, CONSO_KWH, PROD_KWH].
        pv_capacity_kwp: Estimated installed capacity.
        weather_df: Weather data with [dt_utc, global_rad_W].
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_capacity_validation(customer_ts, pv_capacity_kwp, weather_df, title=title)


def plot_customer_high_low_profile(
    customer_ts: pd.DataFrame,
    weather_df: pd.DataFrame,
    title: Optional[str] = None,
):
    """Compare average load profiles on high-radiation vs. low-radiation days.

    Args:
        customer_ts: DataFrame with [DT_UTC, CONSO_KWH, PROD_KWH].
        weather_df: Weather data with [dt_utc, global_rad_W].
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_high_low(customer_ts, weather_df, title=title)


def plot_customer_heatmap(
    customer_ts: pd.DataFrame,
    col: str = "CONSO_KWH",
    title: Optional[str] = None,
):
    """Plot a year × time-of-day heatmap for a customer's load or production.

    Args:
        customer_ts: DataFrame with [DT_UTC] and the selected column.
        col: Column to plot (default: CONSO_KWH).
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_heatmap(customer_ts, col=col, title=title)
