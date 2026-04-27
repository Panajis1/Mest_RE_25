"""Portfolio-level visualizations — delegates to pv_detection.py plot functions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PV_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_PV_DIR) not in sys.path:
    sys.path.insert(0, str(_PV_DIR))

try:
    from pv_detection import (
        plot_population_statistics as _plot_population_statistics,
        plot_capacity_vs_production_with_ci as _plot_capacity_ci,
        plot_portfolio_aggregate_load as _plot_aggregate_load,
        plot_evaluation_dashboard as _plot_dashboard,
    )
    _PV_AVAILABLE = True
except ImportError:
    _PV_AVAILABLE = False

try:
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def _require_pv():
    if not _PV_AVAILABLE:
        raise ImportError("pv_detection.py not importable — cannot render portfolio plots")


def plot_population_statistics(
    results: pd.DataFrame,
    title: str = "Portfolio PV Statistics",
):
    """Histograms and summary stats for the detected PV population.

    Args:
        results: Per-customer results with [pv_capacity_kwp, sc_share, ...].
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_population_statistics(results, title=title)


def plot_capacity_vs_production_with_ci(
    results: pd.DataFrame,
    title: str = "PV Capacity vs Annual Production",
):
    """Scatter of estimated capacity vs annual production with CI error bars.

    Args:
        results: Per-customer results with PV capacity and CI columns.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_capacity_ci(results, title=title)


def plot_portfolio_aggregate_load(
    net_consumption: pd.DataFrame,
    title: str = "Portfolio Net Consumption",
):
    """Plot aggregated net consumption time series for the portfolio.

    Args:
        net_consumption: Output of re_nilm.portfolio.forecasting.build_net_consumption.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_aggregate_load(net_consumption, title=title)


def plot_evaluation_dashboard(
    results: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    title: str = "Pipeline Evaluation Dashboard",
):
    """Multi-panel evaluation dashboard: yield distribution, flags, segment breakdown.

    Args:
        results: Per-customer results with capacity and flag columns.
        pv_indicators: Per-customer annual PV indicators.
        title: Dashboard title.

    Returns:
        Plotly Figure.
    """
    _require_pv()
    return _plot_dashboard(results, pv_indicators, title=title)


def write_plotly_figures_to_dir(
    figures: List,
    output_dir: Path,
    fmt: str = "png",
    width: int = 1200,
    height: int = 700,
) -> None:
    """Export a list of Plotly figures to a directory as PNG or HTML.

    Args:
        figures: List of (filename_stem, figure) tuples.
        output_dir: Directory to write files to (created if needed).
        fmt: 'png' (requires kaleido) or 'html'.
        width, height: Image dimensions for PNG export.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly not installed — cannot export figures")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, fig in figures:
        if fmt == "html":
            fig.write_html(str(out / f"{name}.html"))
        else:
            fig.write_image(str(out / f"{name}.png"), width=width, height=height)
