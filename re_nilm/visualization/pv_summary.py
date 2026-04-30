"""Matplotlib 3-panel PV detection summary PNG.

Mirrors the AC Detection Results layout: pie chart of has_pv share, aggregated
installed capacity bar with summed CI, aggregated self-consumption bar with
capacity-weighted mean + capacity-weighted CI. Built in matplotlib (not plotly)
because the target output is a single landscape PNG suitable for the
thesis / portfolio summary slide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_PIE_HAS = "#e07b6a"      # warm red for the positive class — matches the AC reference
_PIE_NO = "#5e9bd0"       # cool blue for the negative class
_BAR_CAPACITY = "#5e9bd0"
_BAR_SC = "#7da953"       # green for self-consumption
_BAR_EDGE = "#333"


def _bool_series(series: pd.Series) -> pd.Series:
    """Coerce a result column into a boolean Series without treating "False" as truthy."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    return series.astype("string").str.lower().fillna("").isin({"true", "1", "yes", "y"})


def plot_pv_detection_summary(
    results: pd.DataFrame,
    title: str = "PV Detection Results — RE Dataset",
):
    """Build a 3-panel matplotlib Figure summarising the PV portfolio.

    Panels (left to right):
      1. Pie chart of `has_pv` vs `no_pv` over the full population, with
         percentages and total user count.
      2. Aggregated installed capacity (MWp) computed only over
         `has_pv == True` customers, with summed CI bounds as error bars.
         Annotated with the same n_PV count + mean/median per-customer kWp.
      3. Aggregated self-consumption restricted to `has_pv == True`
         customers: capacity-weighted mean of `sc_share` with capacity-
         weighted lower/upper CI bounds (matches the legacy aggregator
         in `_pv_portfolio_v1._capacity_weighted_sc_aggregate`).

    All stat panels share one denominator — the number of PV-positive
    customers — so the counts are directly comparable.

    Args:
        results: Per-customer joined results table. Required columns:
            `has_pv`. Recommended: `pv_capacity_kwp`, `pv_ci_lower`,
            `pv_ci_upper`, `sc_share`. Missing columns produce an
            annotated empty panel rather than raising.
        title: Suptitle for the figure.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt

    if "has_pv" not in results.columns:
        raise ValueError("results must include 'has_pv'")

    pv_mask = _bool_series(results["has_pv"])
    pv_only = results.loc[pv_mask].copy()
    n_pv = int(pv_mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    _draw_pv_ratio_pie(axes[0], results, pv_mask)
    _draw_aggregate_capacity_bar(axes[1], pv_only, n_pv)
    _draw_aggregate_sc_bar(axes[2], pv_only, n_pv)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Panel implementations
# ---------------------------------------------------------------------------


def _draw_pv_ratio_pie(ax, results: pd.DataFrame, pv_mask: Optional[pd.Series] = None) -> None:
    if pv_mask is None:
        pv_mask = _bool_series(results["has_pv"])
    n_total = int(pv_mask.notna().sum())
    n_pv = int(pv_mask.sum())
    n_no = max(0, n_total - n_pv)
    if n_total == 0:
        ax.set_title("Overall PV ratio\n(no customers)")
        ax.axis("off")
        return

    sizes = [n_pv, n_no]
    labels = ["has_pv", "no_pv"]
    colors = [_PIE_HAS, _PIE_NO]
    # Tiny explode on the positive slice so it reads cleanly even when small.
    explode = (0.04, 0.0)

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        startangle=90,
        autopct="%1.1f%%",
        textprops={"fontsize": 13, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title(f"Overall PV ratio\n(n={n_total:,} users)", fontsize=12)


def _draw_aggregate_capacity_bar(ax, pv_only: pd.DataFrame, n_pv: int) -> None:
    """Aggregate capacity over the PV-positive subset (n_pv customers).

    pv_only is already pre-filtered to has_pv == True. Capacity values that
    are NaN or non-positive within that subset are excluded from the sum
    (and counted separately so the annotation is honest).
    """
    if "pv_capacity_kwp" not in pv_only.columns:
        ax.text(0.5, 0.5, "pv_capacity_kwp not in results",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated PV capacity")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    cap = pd.to_numeric(pv_only["pv_capacity_kwp"], errors="coerce")
    valid = cap.notna() & (cap > 0)
    cap_valid = cap.loc[valid]
    if cap_valid.empty:
        ax.text(0.5, 0.5, "No positive capacity estimates among PV customers",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated PV capacity")
        return

    total_kwp = float(cap_valid.sum())
    lower_kwp = (
        float(pd.to_numeric(pv_only.loc[valid, "pv_ci_lower"], errors="coerce").fillna(cap_valid).sum())
        if "pv_ci_lower" in pv_only.columns else total_kwp
    )
    upper_kwp = (
        float(pd.to_numeric(pv_only.loc[valid, "pv_ci_upper"], errors="coerce").fillna(cap_valid).sum())
        if "pv_ci_upper" in pv_only.columns else total_kwp
    )
    err_lo = max(0.0, total_kwp - lower_kwp) / 1000.0
    err_hi = max(0.0, upper_kwp - total_kwp) / 1000.0

    n_estimates = int(valid.sum())
    mean_kwp = float(cap_valid.mean())
    median_kwp = float(cap_valid.median())
    missing = n_pv - n_estimates

    ax.bar(
        ["PV portfolio"],
        [total_kwp / 1000.0],
        yerr=[[err_lo], [err_hi]],
        color=_BAR_CAPACITY,
        edgecolor=_BAR_EDGE,
        linewidth=1.0,
        width=0.5,
        capsize=10,
        ecolor=_BAR_EDGE,
        error_kw={"linewidth": 1.5},
    )
    ax.set_ylabel("Installed capacity (MWp)", fontsize=11)
    ax.set_title(
        f"Aggregated PV capacity\n"
        f"{total_kwp / 1000.0:.1f} MWp  (CI {lower_kwp / 1000.0:.1f}–{upper_kwp / 1000.0:.1f} MWp)",
        fontsize=12,
    )
    extra = f"  ({missing:,} missing capacity)" if missing > 0 else ""
    ax.text(
        0.5, 0.96,
        f"PV customers: {n_pv:,}{extra}\n"
        f"mean / median: {mean_kwp:.1f} / {median_kwp:.1f} kWp",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="#444",
    )
    # Headroom above the bar so the annotation doesn't overlap.
    top = (total_kwp + max(err_hi * 1000, total_kwp * 0.1)) / 1000.0
    ax.set_ylim(0, top * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_aggregate_sc_bar(ax, pv_only: pd.DataFrame, n_pv: int) -> None:
    """Aggregate self-consumption over the PV-positive subset (n_pv customers).

    Uses the legacy `_capacity_weighted_sc_aggregate` formulation:

        agg_sc       = sum(kWp_i * sc_share_i) / sum(kWp_i)
        agg_sc_lower = sum(kWp_i * sc_ci_lower_i) / sum(kWp_i)
        agg_sc_upper = sum(kWp_i * sc_ci_upper_i) / sum(kWp_i)

    Capacity weighting is the right physical aggregation: a 50 kWp system
    contributes 5x more to "what fraction of kWh produced is consumed locally"
    than a 10 kWp system. NaN bounds fall back to the point estimate so the
    error bar collapses for customers without a CI rather than dragging the
    aggregate to extremes.
    """
    needed = ("sc_share", "pv_capacity_kwp")
    missing_cols = [c for c in needed if c not in pv_only.columns]
    if missing_cols:
        ax.text(0.5, 0.5, f"missing columns: {missing_cols}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated self-consumption")
        return

    sc = pd.to_numeric(pv_only["sc_share"], errors="coerce")
    kwp = pd.to_numeric(pv_only["pv_capacity_kwp"], errors="coerce")
    valid = sc.notna() & (sc >= 0) & (sc <= 1) & kwp.notna() & (kwp > 0)
    sc = sc.loc[valid]
    kwp = kwp.loc[valid]
    if sc.empty:
        ax.text(0.5, 0.5, "No usable sc_share + capacity rows",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated self-consumption")
        return

    w_sum = float(kwp.sum())
    agg_sc = float((kwp * sc).sum() / w_sum)

    sc_lo = pd.to_numeric(pv_only.loc[valid, "sc_ci_lower"], errors="coerce").fillna(sc) if "sc_ci_lower" in pv_only.columns else sc
    sc_hi = pd.to_numeric(pv_only.loc[valid, "sc_ci_upper"], errors="coerce").fillna(sc) if "sc_ci_upper" in pv_only.columns else sc
    agg_lo = float(np.clip((kwp * sc_lo).sum() / w_sum, 0.0, 1.0))
    agg_hi = float(np.clip((kwp * sc_hi).sum() / w_sum, 0.0, 1.0))
    err_lo = max(0.0, agg_sc - agg_lo)
    err_hi = max(0.0, agg_hi - agg_sc)

    missing = n_pv - int(valid.sum())

    ax.bar(
        ["Self-consumption"],
        [agg_sc * 100],
        yerr=[[err_lo * 100], [err_hi * 100]],
        color=_BAR_SC,
        edgecolor=_BAR_EDGE,
        linewidth=1.0,
        width=0.5,
        capsize=10,
        ecolor=_BAR_EDGE,
        error_kw={"linewidth": 1.5},
    )
    ax.set_ylabel("Self-consumption share (%)", fontsize=11)
    # Headroom above the error bar so the annotation does not overlap.
    top_pct = max(40.0, (agg_sc + err_hi) * 100 * 1.4)
    ax.set_ylim(0, min(100.0, top_pct))
    ax.set_title(
        f"Aggregated self-consumption\n"
        f"{agg_sc * 100:.1f}%  (CI {agg_lo * 100:.1f}–{agg_hi * 100:.1f}%)",
        fontsize=12,
    )
    extra = f"  ({missing:,} missing sc_share)" if missing > 0 else ""
    ax.text(
        0.5, 0.96,
        f"PV customers: {n_pv:,}{extra}\n"
        f"capacity-weighted across {int(valid.sum()):,} customers",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="#444",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_pv_detection_summary(
    results: pd.DataFrame,
    output_path: Path,
    title: str = "PV Detection Results — RE Dataset",
    dpi: int = 150,
) -> Path:
    """Render the 3-panel summary and write it to ``output_path`` as PNG."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_pv_detection_summary(results, title=title)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pv_capacity_vs_sc_share(
    results: pd.DataFrame,
    title: str = "System Size vs. Self-Consumption Share",
    sc_floor: float = 1e-6,
):
    """Scatter of estimated PV capacity vs self-consumption share.

    Restricted to ``has_pv == True`` customers (consistent with the 3-panel
    summary's stat panels). Self-consumption is plotted on a log scale;
    values below ``sc_floor`` are clamped to that floor so they appear at
    the bottom edge instead of silently disappearing on the log axis.

    Args:
        results: Per-customer joined results table.
        title: Plot title.
        sc_floor: Lower clip for sc_share so 0 / near-zero values remain
            visible on the log axis. The y-axis label reflects this floor.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt

    if "has_pv" not in results.columns:
        raise ValueError("results must include 'has_pv'")
    if "pv_capacity_kwp" not in results.columns or "sc_share" not in results.columns:
        raise ValueError("results must include 'pv_capacity_kwp' and 'sc_share'")

    pv = results.loc[_bool_series(results["has_pv"])]
    cap = pd.to_numeric(pv["pv_capacity_kwp"], errors="coerce")
    sc = pd.to_numeric(pv["sc_share"], errors="coerce")
    valid = cap.notna() & (cap > 0) & sc.notna() & (sc >= 0)
    cap = cap.loc[valid]
    sc = sc.loc[valid].clip(lower=sc_floor, upper=1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    if cap.empty:
        ax.text(0.5, 0.5, "No PV customers with capacity + sc_share",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        return fig

    ax.scatter(
        cap, sc,
        s=14, alpha=0.45,
        color=_BAR_CAPACITY,
        edgecolors="none",
    )
    ax.set_yscale("log")
    ax.set_ylim(sc_floor, 1.5)
    ax.set_xlim(0, float(cap.max()) * 1.03)
    ax.set_xlabel("Estimated PV capacity (kWp)", fontsize=11)
    ax.set_ylabel(f"Self-consumption share (log scale, ≥{sc_floor:g})", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", axis="both", alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_pts = len(cap)
    n_floored = int((sc <= sc_floor * 1.01).sum())
    extra = f"  ({n_floored:,} clipped to floor)" if n_floored else ""
    ax.text(
        0.99, 0.02,
        f"n PV customers: {n_pts:,}{extra}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color="#444",
    )
    fig.tight_layout()
    return fig


def save_pv_capacity_vs_sc_share(
    results: pd.DataFrame,
    output_path: Path,
    title: str = "System Size vs. Self-Consumption Share",
    dpi: int = 150,
) -> Path:
    """Render the capacity-vs-sc scatter and write it to ``output_path`` as PNG."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_pv_capacity_vs_sc_share(results, title=title)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
