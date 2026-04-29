"""Portfolio-level visualizations — delegates to pv_detection.py plot functions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import pandas as pd

from re_nilm.visualization._pv_plots_v1 import (
    plot_capacity_vs_production_with_ci as _plot_capacity_ci,
    plot_evaluation_dashboard as _plot_dashboard,
    plot_population_statistics as _plot_population_statistics,
    plot_portfolio_aggregate_load as _plot_aggregate_load,
)

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


_APPLIANCE_SPECS = [
    ("PV", "has_pv", "prob_pv"),
    ("AC", "has_ac", "prob_ac"),
    ("Heat pump", "has_hp", "prob_hp"),
    ("Battery", "has_battery", "prob_battery"),
    ("EV", "has_ev", "prob_ev"),
]


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly not installed - cannot create portfolio figures")


def _bool_series(series: pd.Series) -> pd.Series:
    """Convert common boolean-like result columns without treating 'False' as truthy."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    return series.astype("string").str.lower().fillna("").isin({"true", "1", "yes", "y"})


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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
    return _plot_aggregate_load(net_consumption, title=title)


def plot_appliance_adoption_shares(
    results: pd.DataFrame,
    title: str = "Portfolio Appliance Adoption",
):
    """Bar chart of appliance adoption shares with Wilson 95% confidence intervals."""
    _require_plotly()

    rows = []
    for label, flag_col, _ in _APPLIANCE_SPECS:
        if flag_col not in results.columns:
            continue
        valid = results[flag_col].notna()
        n = int(valid.sum())
        k = int(_bool_series(results.loc[valid, flag_col]).sum()) if n else 0
        lower, upper = _wilson_ci(k, n)
        share = k / n if n else 0.0
        rows.append({
            "appliance": label,
            "share_pct": share * 100.0,
            "ci_low_pct": lower * 100.0,
            "ci_high_pct": upper * 100.0,
            "positive": k,
            "n": n,
        })

    if not rows:
        raise ValueError("No appliance detection columns found in results")

    df = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(
        x=df["appliance"],
        y=df["share_pct"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=df["ci_high_pct"] - df["share_pct"],
            arrayminus=df["share_pct"] - df["ci_low_pct"],
            thickness=1.5,
        ),
        customdata=df[["positive", "n", "ci_low_pct", "ci_high_pct"]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Share: %{y:.1f}%<br>"
            "Customers: %{customdata[0]:,.0f} / %{customdata[1]:,.0f}<br>"
            "95% CI: %{customdata[2]:.1f}%-%{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Appliance",
        yaxis_title="Share of customers (%)",
        yaxis=dict(range=[0, max(10.0, min(100.0, df["ci_high_pct"].max() * 1.15))]),
    )
    return fig


def plot_pv_installed_capacity_summary(
    results: pd.DataFrame,
    title: str = "Aggregated Installed PV Capacity",
):
    """Portfolio-level PV installed capacity bar with conservative CI bounds."""
    _require_plotly()
    if "pv_capacity_kwp" not in results.columns:
        raise ValueError("results must include pv_capacity_kwp")

    cap = pd.to_numeric(results["pv_capacity_kwp"], errors="coerce")
    valid = cap.notna() & (cap > 0)
    if not valid.any():
        raise ValueError("No positive PV capacity estimates found")

    cap_valid = cap.loc[valid]
    total_kwp = float(cap_valid.sum())
    lower_kwp = (
        float(pd.to_numeric(results.loc[valid, "pv_ci_lower"], errors="coerce").fillna(cap_valid).sum())
        if "pv_ci_lower" in results.columns else total_kwp
    )
    upper_kwp = (
        float(pd.to_numeric(results.loc[valid, "pv_ci_upper"], errors="coerce").fillna(cap_valid).sum())
        if "pv_ci_upper" in results.columns else total_kwp
    )
    n_capacity = int(valid.sum())
    n_detected = (
        int(_bool_series(results["has_pv"]).sum())
        if "has_pv" in results.columns else n_capacity
    )
    stats = {
        "mean_kwp": float(cap_valid.mean()),
        "median_kwp": float(cap_valid.median()),
        "p25_kwp": float(cap_valid.quantile(0.25)),
        "p75_kwp": float(cap_valid.quantile(0.75)),
    }

    fig = go.Figure(go.Bar(
        x=["PV"],
        y=[total_kwp / 1000.0],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[max(0.0, upper_kwp - total_kwp) / 1000.0],
            arrayminus=[max(0.0, total_kwp - lower_kwp) / 1000.0],
            thickness=1.5,
        ),
        customdata=[[
            n_detected,
            n_capacity,
            lower_kwp / 1000.0,
            upper_kwp / 1000.0,
            stats["mean_kwp"],
            stats["median_kwp"],
            stats["p25_kwp"],
            stats["p75_kwp"],
        ]],
        hovertemplate=(
            "<b>PV portfolio</b><br>"
            "Total: %{y:.2f} MWp<br>"
            "95% CI: %{customdata[2]:.2f}-%{customdata[3]:.2f} MWp<br>"
            "PV detected: %{customdata[0]:,.0f}<br>"
            "Capacity estimates: %{customdata[1]:,.0f}<br>"
            "Mean: %{customdata[4]:.1f} kWp<br>"
            "Median: %{customdata[5]:.1f} kWp<br>"
            "IQR: %{customdata[6]:.1f}-%{customdata[7]:.1f} kWp"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Technology",
        yaxis_title="Installed capacity (MWp)",
        annotations=[dict(
            text=(
                f"PV detected: {n_detected:,}<br>"
                f"Capacity estimates: {n_capacity:,}<br>"
                f"Mean / median: {stats['mean_kwp']:.1f} / {stats['median_kwp']:.1f} kWp"
            ),
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
        )],
    )
    return fig


def plot_pv_capacity_distribution(
    results: pd.DataFrame,
    x_max_kwp: float = 100.0,
    title: str = "PV Capacity Distribution",
):
    """Histogram of PV capacities with display range capped at 0-x_max_kwp."""
    _require_plotly()
    if "pv_capacity_kwp" not in results.columns:
        raise ValueError("results must include pv_capacity_kwp")

    cap = pd.to_numeric(results["pv_capacity_kwp"], errors="coerce")
    cap = cap[cap > 0].dropna()
    if cap.empty:
        raise ValueError("No positive PV capacity estimates found")

    n_over = int((cap > x_max_kwp).sum())
    fig = go.Figure(go.Histogram(
        x=cap,
        xbins=dict(start=0, end=x_max_kwp, size=2),
        hovertemplate="Capacity bin: %{x:.1f} kWp<br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="PV capacity (kWp)",
        yaxis_title="Number of customers",
        xaxis=dict(range=[0, x_max_kwp]),
        annotations=[dict(
            text=f"{n_over:,} customers above {x_max_kwp:.0f} kWp" if n_over else f"No customers above {x_max_kwp:.0f} kWp",
            x=0.98,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
        )],
    )
    return fig


def plot_appliance_probability_distributions(
    results: pd.DataFrame,
    title: str = "Appliance Detection Probability Distributions",
):
    """Small-multiple histograms of detector probability outputs."""
    _require_plotly()
    available = [(label, prob_col) for label, _, prob_col in _APPLIANCE_SPECS if prob_col in results.columns]
    if not available:
        raise ValueError("No appliance probability columns found in results")

    fig = make_subplots(rows=1, cols=len(available), subplot_titles=[label for label, _ in available])
    for idx, (label, prob_col) in enumerate(available, start=1):
        prob = pd.to_numeric(results[prob_col], errors="coerce").dropna().clip(0, 1)
        fig.add_trace(
            go.Histogram(
                x=prob,
                nbinsx=30,
                name=label,
                showlegend=False,
                hovertemplate="Probability: %{x:.2f}<br>Customers: %{y}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(range=[0, 1], title_text="Probability", row=1, col=idx)
        fig.update_yaxes(title_text="Customers" if idx == 1 else "", row=1, col=idx)

    fig.update_layout(title=title, bargap=0.05)
    return fig


def plot_appliance_cooccurrence_heatmap(
    results: pd.DataFrame,
    title: str = "Appliance Co-adoption Rates",
):
    """Heatmap of pairwise co-occurrence share across the customer portfolio."""
    _require_plotly()
    available = [(label, flag_col) for label, flag_col, _ in _APPLIANCE_SPECS if flag_col in results.columns]
    if len(available) < 2:
        raise ValueError("At least two appliance flag columns are required")

    labels = [label for label, _ in available]
    flags = {label: _bool_series(results[col]) for label, col in available}
    n = len(results)
    z = []
    text = []
    for row_label in labels:
        z_row = []
        text_row = []
        for col_label in labels:
            both = int((flags[row_label] & flags[col_label]).sum())
            share = both / n * 100.0 if n else 0.0
            z_row.append(share)
            text_row.append(f"{both:,} customers<br>{share:.1f}% of portfolio")
        z.append(z_row)
        text.append(text_row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="Blues",
        zmin=0,
        zmax=max(max(row) for row in z) if z else 1,
        text=text,
        hovertemplate="<b>%{y} + %{x}</b><br>%{text}<extra></extra>",
        colorbar=dict(title="% customers"),
    ))
    fig.update_layout(title=title, xaxis_title="Appliance", yaxis_title="Appliance")
    return fig


def plot_technology_portfolio_summaries(
    results: pd.DataFrame,
    title: str = "Technology-specific Portfolio Summaries",
):
    """Compact summary panels for HP type, battery capacity, and EV usage."""
    _require_plotly()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["HP type", "Battery capacity", "EV annual energy"],
    )

    has_any = False
    if "hp_type" in results.columns:
        hp_counts = results["hp_type"].fillna("missing").astype(str).value_counts()
        fig.add_trace(
            go.Bar(
                x=hp_counts.index,
                y=hp_counts.values,
                name="HP type",
                showlegend=False,
                hovertemplate="%{x}: %{y:,} customers<extra></extra>",
            ),
            row=1,
            col=1,
        )
        has_any = True

    if "battery_capacity_kwh" in results.columns:
        batt = pd.to_numeric(results["battery_capacity_kwh"], errors="coerce")
        batt = batt[batt > 0].dropna()
        if not batt.empty:
            fig.add_trace(
                go.Histogram(
                    x=batt,
                    nbinsx=30,
                    name="Battery capacity",
                    showlegend=False,
                    hovertemplate="Capacity: %{x:.1f} kWh<br>Customers: %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            has_any = True

    if "ev_yearly_mwh" in results.columns:
        ev = pd.to_numeric(results["ev_yearly_mwh"], errors="coerce")
        ev = ev[ev > 0].dropna()
        if not ev.empty:
            fig.add_trace(
                go.Histogram(
                    x=ev,
                    nbinsx=30,
                    name="EV annual energy",
                    showlegend=False,
                    hovertemplate="Energy: %{x:.2f} MWh/year<br>Customers: %{y}<extra></extra>",
                ),
                row=1,
                col=3,
            )
            has_any = True

    if not has_any:
        raise ValueError("No HP, battery, or EV summary columns found")

    fig.update_xaxes(title_text="Type", row=1, col=1)
    fig.update_yaxes(title_text="Customers", row=1, col=1)
    fig.update_xaxes(title_text="kWh", row=1, col=2)
    fig.update_xaxes(title_text="MWh/year", row=1, col=3)
    fig.update_layout(title=title, bargap=0.05)
    return fig


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
