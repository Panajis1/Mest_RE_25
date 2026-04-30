"""Matplotlib 3-panel PV detection summary PNG.

Mirrors the AC Detection Results layout: pie chart of has_pv share, aggregated
installed capacity bar with CI, aggregated self-consumption bar with IQR. Built
in matplotlib (not plotly) because the target output is a single landscape PNG
suitable for the thesis / portfolio summary slide.
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
      1. Pie chart of `has_pv` vs `no_pv` with percentages and total user count.
      2. Aggregated installed capacity (MWp) with summed lower/upper CI as
         asymmetric error bars. Annotated with detected/estimate counts and
         mean/median per-customer kWp.
      3. Aggregated self-consumption: median portfolio `sc_share` with IQR as
         the error bar, plus n customers and the mean.

    Args:
        results: Per-customer joined results table. Required columns:
            `has_pv`. Recommended: `pv_capacity_kwp`, `pv_ci_lower`,
            `pv_ci_upper`, `sc_share` (any missing column is gracefully
            replaced by an annotation explaining what's absent).
        title: Suptitle for the figure.

    Returns:
        matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt

    if "has_pv" not in results.columns:
        raise ValueError("results must include 'has_pv'")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # --- Panel 1: PV ratio pie -------------------------------------------------
    _draw_pv_ratio_pie(axes[0], results)

    # --- Panel 2: aggregate capacity ------------------------------------------
    _draw_aggregate_capacity_bar(axes[1], results)

    # --- Panel 3: aggregate self-consumption ----------------------------------
    _draw_aggregate_sc_bar(axes[2], results)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Panel implementations
# ---------------------------------------------------------------------------


def _draw_pv_ratio_pie(ax, results: pd.DataFrame) -> None:
    has = _bool_series(results["has_pv"])
    n_total = int(has.notna().sum())
    n_pv = int(has.sum())
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


def _draw_aggregate_capacity_bar(ax, results: pd.DataFrame) -> None:
    if "pv_capacity_kwp" not in results.columns:
        ax.text(0.5, 0.5, "pv_capacity_kwp not in results",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated PV capacity")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    cap = pd.to_numeric(results["pv_capacity_kwp"], errors="coerce")
    valid = cap.notna() & (cap > 0)
    cap_valid = cap.loc[valid]
    if cap_valid.empty:
        ax.text(0.5, 0.5, "No positive capacity estimates",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated PV capacity")
        return

    total_kwp = float(cap_valid.sum())
    lower_kwp = (
        float(pd.to_numeric(results.loc[valid, "pv_ci_lower"], errors="coerce").fillna(cap_valid).sum())
        if "pv_ci_lower" in results.columns else total_kwp
    )
    upper_kwp = (
        float(pd.to_numeric(results.loc[valid, "pv_ci_upper"], errors="coerce").fillna(cap_valid).sum())
        if "pv_ci_upper" in results.columns else total_kwp
    )
    err_lo = max(0.0, total_kwp - lower_kwp) / 1000.0
    err_hi = max(0.0, upper_kwp - total_kwp) / 1000.0

    n_estimates = int(valid.sum())
    n_detected = (
        int(_bool_series(results["has_pv"]).sum())
        if "has_pv" in results.columns else n_estimates
    )
    mean_kwp = float(cap_valid.mean())
    median_kwp = float(cap_valid.median())

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
    ax.text(
        0.5, 0.96,
        f"detected: {n_detected:,}   estimates: {n_estimates:,}\n"
        f"mean / median: {mean_kwp:.1f} / {median_kwp:.1f} kWp",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="#444",
    )
    # Headroom above the bar so the annotation doesn't overlap.
    top = (total_kwp + max(err_hi * 1000, total_kwp * 0.1)) / 1000.0
    ax.set_ylim(0, top * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_aggregate_sc_bar(ax, results: pd.DataFrame) -> None:
    if "sc_share" not in results.columns:
        ax.text(0.5, 0.5, "sc_share not in results",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated self-consumption")
        return

    sc = pd.to_numeric(results["sc_share"], errors="coerce").dropna()
    sc = sc[(sc >= 0) & (sc <= 1)]
    if sc.empty:
        ax.text(0.5, 0.5, "No self-consumption values in [0, 1]",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Aggregated self-consumption")
        return

    median_sc = float(sc.median())
    p25 = float(sc.quantile(0.25))
    p75 = float(sc.quantile(0.75))
    mean_sc = float(sc.mean())

    ax.bar(
        ["Self-consumption"],
        [median_sc * 100],
        yerr=[[(median_sc - p25) * 100], [(p75 - median_sc) * 100]],
        color=_BAR_SC,
        edgecolor=_BAR_EDGE,
        linewidth=1.0,
        width=0.5,
        capsize=10,
        ecolor=_BAR_EDGE,
        error_kw={"linewidth": 1.5},
    )
    ax.set_ylabel("Self-consumption share (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Aggregated self-consumption\n"
        f"median {median_sc * 100:.1f}%  (IQR {p25 * 100:.1f}–{p75 * 100:.1f}%)",
        fontsize=12,
    )
    ax.text(
        0.5, 0.96,
        f"n customers: {len(sc):,}\nmean: {mean_sc * 100:.1f}%",
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
