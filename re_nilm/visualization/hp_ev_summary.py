"""Matplotlib HP/EV portfolio plots saved as PNG figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RED = "#c9252b"
_BLACK = "#111111"
_GRAY = "#888888"


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    return series.astype("string").str.lower().fillna("").isin({"true", "1", "yes", "y"})


def plot_hp_customer_mix_pie(
    results: pd.DataFrame,
    title: str = "Heat pump customer mix",
):
    import matplotlib.pyplot as plt

    n_total = int(len(results))
    if n_total == 0:
        raise ValueError("results is empty")

    hp_type = results["hp_type"].astype("string").str.lower().fillna("no_hp") if "hp_type" in results.columns else None
    has_hp = _bool_series(results["has_hp"]) if "has_hp" in results.columns else pd.Series(False, index=results.index)

    if hp_type is None:
        n_winter = int(has_hp.sum())
        n_summer = 0
    else:
        n_winter = int(hp_type.eq("winter_hp").sum())
        n_summer = int(hp_type.eq("summer_hp").sum())
    n_no = max(0, n_total - n_winter - n_summer)

    sizes = [n_winter, n_no, n_summer]
    labels = ["Customers with Winter HPs", "Customers without HPs", "Customers with Summer HPs"]
    colors = [_RED, _BLACK, _GRAY]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        explode=(0.03, 0.0, 0.02),
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
    )
    for i, t in enumerate(autotexts):
        if sizes[i] > 0:
            pct = sizes[i] / n_total * 100.0
            t.set_text(f"{sizes[i]}\n{pct:.0f}%")
            t.set_color("white")
            t.set_fontsize(11)
    ax.set_title(title, fontsize=13)
    ax.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def plot_hp_annual_consumption_pdf(
    hp_annual_kwh: pd.Series,
    title: str = "PDF of annual heat pump consumption",
):
    import matplotlib.pyplot as plt

    values = pd.to_numeric(hp_annual_kwh, errors="coerce")
    values = values[values > 0].dropna()
    if values.empty:
        raise ValueError("No positive annual HP consumption values available")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    values.plot.kde(ax=ax, color=_RED, linewidth=1.4)
    x = ax.lines[-1].get_xdata()
    y = ax.lines[-1].get_ydata()
    ax.fill_between(x, y, color=_RED, alpha=0.25)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Annual HP consumption (kWh)")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_ev_probability_histogram(
    results: pd.DataFrame,
    title: str = "EV probability distribution (all customers)",
):
    import matplotlib.pyplot as plt

    if "prob_ev" not in results.columns:
        raise ValueError("results must include 'prob_ev'")

    probs = pd.to_numeric(results["prob_ev"], errors="coerce").clip(0, 1).dropna()
    if probs.empty:
        raise ValueError("No EV probability values available")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.hist(probs, bins=np.linspace(0, 1, 31), color=_RED, edgecolor="white", linewidth=0.6)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("EV probability")
    ax.set_ylabel("Number of customers")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _save(fig, output_path: Path, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_hp_customer_mix_pie(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_hp_customer_mix_pie(results), output_path=output_path, dpi=dpi)


def save_hp_annual_consumption_pdf(hp_annual_kwh: pd.Series, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_hp_annual_consumption_pdf(hp_annual_kwh), output_path=output_path, dpi=dpi)


def save_ev_probability_histogram(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_ev_probability_histogram(results), output_path=output_path, dpi=dpi)
