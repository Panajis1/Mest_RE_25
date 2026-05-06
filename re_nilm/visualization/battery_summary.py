"""Matplotlib battery portfolio plots saved as PNG figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RED = "#c9252b"
_BLACK = "#111111"
_GREY = "#888888"
_GRID = "#bfbfbf"

_PROB_CANDIDATES = ("prob_battery", "battery_prob", "battery_probability")
_STATUS_CANDIDATES = ("battery_status",)
_CAPACITY_CANDIDATES = ("battery_capacity_kwh", "estimated_battery_capacity_kwh", "capacity_kwh")
_POWER_CANDIDATES = ("battery_power_kw", "estimated_battery_power_kw", "power_kw")
_PV_FLAG_CANDIDATES = ("has_pv",)
_PV_PROB_CANDIDATES = ("prob_pv", "has_pv_prob")
_PV_CAP_CANDIDATES = ("pv_capacity_kwp", "pv_capacity_ci_upper", "pv_capacity_kwp_floor")


def _pick_column(columns, candidates):
    by_lower = {str(col).lower(): col for col in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        match = by_lower.get(cand.lower())
        if match is not None:
            return match
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    return series.astype("string").str.lower().fillna("").isin({"true", "1", "yes", "y"})


def _prob_pct(results: pd.DataFrame) -> pd.Series:
    """Return battery probability as 0–100 regardless of input scale."""
    col = _pick_column(results.columns, _PROB_CANDIDATES)
    if col is None:
        raise ValueError(
            "results must include one of " + ", ".join(_PROB_CANDIDATES)
        )
    raw = pd.to_numeric(results[col], errors="coerce").fillna(0.0)
    return (raw * 100.0 if raw.max() <= 1.0 else raw).clip(0.0, 100.0)


def _detected_mask(results: pd.DataFrame) -> pd.Series:
    if "has_battery" in results.columns:
        return _bool_series(results["has_battery"])
    # Fall back to threshold on probability.
    return _prob_pct(results) >= 50.0


def _is_pv_customer(results: pd.DataFrame) -> pd.Series:
    flag = _pick_column(results.columns, _PV_FLAG_CANDIDATES)
    if flag is not None:
        return _bool_series(results[flag])
    prob = _pick_column(results.columns, _PV_PROB_CANDIDATES)
    if prob is not None:
        return pd.to_numeric(results[prob], errors="coerce").fillna(0.0) >= 0.5
    cap = _pick_column(results.columns, _PV_CAP_CANDIDATES)
    if cap is not None:
        return pd.to_numeric(results[cap], errors="coerce").fillna(0.0) > 0.0
    return pd.Series(False, index=results.index)


def _battery_reliable_mask(
    results: pd.DataFrame,
    threshold: float = 0.5,
    reliability_margin: float = 0.15,
) -> pd.Series:
    probs = _prob_pct(results) / 100.0
    status_col = _pick_column(results.columns, _STATUS_CANDIDATES)
    if status_col is None:
        status_ok = pd.Series(True, index=results.index)
    else:
        text = results[status_col].astype("string").str.lower().fillna("")
        status_ok = text.eq("success") | text.eq("ok") | text.str.startswith("success")
    return status_ok & (probs.sub(float(threshold)).abs() >= float(reliability_margin))


def plot_battery_capacity_histogram(
    results: pd.DataFrame,
    title: str = "Estimated battery capacity among detected battery customers",
):
    import matplotlib.pyplot as plt

    cap_col = _pick_column(results.columns, _CAPACITY_CANDIDATES)
    if cap_col is None:
        raise ValueError("results must include a battery capacity column")

    detected = _detected_mask(results)
    cap = pd.to_numeric(results.loc[detected, cap_col], errors="coerce")
    cap = cap[cap > 0].dropna()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    if cap.empty:
        ax.text(
            0.5, 0.5,
            "No positive battery capacity estimates\namong detected customers",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="#444444",
        )
    else:
        bins = max(12, min(30, int(np.sqrt(len(cap)) * 2)))
        ax.hist(cap, bins=bins, color=_RED, edgecolor=_BLACK, linewidth=1.0)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Estimated battery capacity (kWh)")
    ax.set_ylabel("Customers")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, color=_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_battery_reliability_summary(
    results: pd.DataFrame,
    threshold: float = 0.5,
    reliability_margin: float = 0.15,
):
    import matplotlib.pyplot as plt

    reliable = _battery_reliable_mask(results, threshold=threshold, reliability_margin=reliability_margin)
    detected = _detected_mask(results)

    n_total = int(len(results))
    n_reliable = int(reliable.sum())
    n_not_reliable = max(0, n_total - n_reliable)
    n_detected_reliable = int((detected & reliable).sum())
    n_detected_review = int((detected & ~reliable).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    axes[0].pie(
        [n_reliable, n_not_reliable] if n_total else [1, 0],
        colors=[_RED, _BLACK],
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%" if n_total else None,
        textprops={"color": "white", "fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.1},
    )
    axes[0].set_title(
        "Overall reliability assessment\n"
        f"(status success + |prob - {threshold * 100:.0f}%| >= {reliability_margin * 100:.0f}%)",
        fontsize=11,
    )

    axes[1].bar(
        ["Reliable detected", "Detected needs review"],
        [n_detected_reliable, n_detected_review],
        color=[_RED, _BLACK],
        edgecolor=_BLACK,
        linewidth=0.9,
    )
    axes[1].set_title("Reliability among detected battery customers", fontsize=11)
    axes[1].set_ylabel("Customers")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color=_GRID)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_battery_probability_distribution(
    results: pd.DataFrame,
    threshold: float = 0.5,
    title: str = "Battery probability distribution",
):
    import matplotlib.pyplot as plt

    probs = _prob_pct(results)
    detected = _detected_mask(results)
    n = len(probs)
    if n == 0:
        raise ValueError("results is empty")

    bins = np.linspace(0, 100, 51)
    w_no = np.full((~detected).sum(), 1.0 / n)
    w_yes = np.full(detected.sum(), 1.0 / n)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.hist(
        [probs.loc[~detected], probs.loc[detected]],
        bins=bins,
        weights=[w_no, w_yes],
        color=[_BLACK, _RED],
        label=["No battery detected", "Battery detected"],
        stacked=True,
        edgecolor=_BLACK,
        linewidth=0.3,
    )
    ax.axvline(float(threshold) * 100.0, color="#555555", linestyle="--", linewidth=1.0, label="Threshold")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Battery probability (%)")
    ax.set_ylabel("Share of customers")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, color=_GRID)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_battery_detected_share_pie(
    results: pd.DataFrame,
    threshold: float = 0.5,
):
    """Pie of PV customers split into detected-battery vs not."""
    import matplotlib.pyplot as plt

    pv_mask = _is_pv_customer(results)
    pv_df = results.loc[pv_mask]
    n_pv = int(len(pv_df))
    if n_pv == 0:
        raise ValueError("No PV customers in results")
    n_det = int(_detected_mask(pv_df).sum())
    n_not = n_pv - n_det

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(
        [n_not, n_det],
        labels=["No battery", "Battery detected"],
        colors=[_BLACK, _RED],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": "white"},
    )
    ax.set_title(
        f"Detected battery share among PV customers\n"
        f"(threshold = {threshold * 100:.0f}%, n_pv = {n_pv})"
    )
    fig.tight_layout()
    return fig


def plot_battery_status_breakdown(
    results: pd.DataFrame,
    top_n: int = 12,
):
    import matplotlib.pyplot as plt

    status_col = _pick_column(results.columns, _STATUS_CANDIDATES)
    if status_col is None:
        raise ValueError("results must include 'battery_status'")
    counts = results[status_col].fillna("Unknown").astype(str).value_counts().head(top_n)
    if counts.empty:
        raise ValueError("No battery_status values to plot")

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [_BLACK if i % 2 == 0 else _RED for i in range(len(counts))]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Detection status breakdown")
    ax.set_xlabel("Status")
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_battery_pv_split_pie(results: pd.DataFrame):
    """Among detected-battery customers, share with vs without PV flag."""
    import matplotlib.pyplot as plt

    detected = results.loc[_detected_mask(results)]
    if detected.empty:
        raise ValueError("No detected battery customers")
    pv = _is_pv_customer(detected)
    n_pv = int(pv.sum())
    n_no = int(len(detected) - n_pv)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(
        [n_no, n_pv],
        labels=["No PV flag", "PV flag"],
        colors=[_BLACK, _RED],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": "white"},
    )
    ax.set_title(f"PV split among detected battery customers\n(n = {len(detected)})")
    fig.tight_layout()
    return fig


def _grouped_boxplot(values_by_label, title, ylabel):
    import matplotlib.pyplot as plt

    groups, labels, colors = [], [], []
    for label, values, color in values_by_label:
        if len(values):
            groups.append(values)
            labels.append(f"{label}\n(n={len(values)})")
            colors.append(color)
    if not groups:
        raise ValueError("No data to plot")

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        groups, tick_labels=labels, patch_artist=True, notch=False,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_battery_capacity_boxplot(results: pd.DataFrame):
    cap_col = _pick_column(results.columns, _CAPACITY_CANDIDATES)
    if cap_col is None:
        raise ValueError("results must include a battery capacity column")
    detected = results.loc[_detected_mask(results)].copy()
    detected["_cap"] = pd.to_numeric(detected[cap_col], errors="coerce")
    detected = detected.dropna(subset=["_cap"])
    if detected.empty:
        raise ValueError("No detected customers with capacity")
    pv = _is_pv_customer(detected)
    return _grouped_boxplot(
        [
            ("PV customers", detected.loc[pv, "_cap"].values, _RED),
            ("Non-PV customers", detected.loc[~pv, "_cap"].values, _GREY),
            ("All detected", detected["_cap"].values, _BLACK),
        ],
        title="Battery capacity distribution — detected customers",
        ylabel="Estimated battery capacity (kWh)",
    )


def plot_battery_power_distribution(results: pd.DataFrame):
    import matplotlib.pyplot as plt

    pwr_col = _pick_column(results.columns, _POWER_CANDIDATES)
    if pwr_col is None:
        raise ValueError("results must include a battery power column")
    detected = pd.to_numeric(
        results.loc[_detected_mask(results), pwr_col], errors="coerce"
    ).dropna()
    if detected.empty:
        raise ValueError("No detected customers with power")

    bins = max(10, min(40, int(np.sqrt(len(detected)) * 3)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(detected, bins=bins, color=_RED, edgecolor="white", alpha=0.85)
    ax.axvline(detected.median(), color="white", linestyle="--", linewidth=1.5,
               label=f"Median: {detected.median():.2f} kW")
    ax.axvline(detected.mean(), color=_GREY, linestyle=":", linewidth=1.5,
               label=f"Mean: {detected.mean():.2f} kW")
    ax.set_title("Estimated battery power distribution — detected customers")
    ax.set_xlabel("Estimated battery power (kW)")
    ax.set_ylabel("Customers")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_battery_power_boxplot(results: pd.DataFrame):
    pwr_col = _pick_column(results.columns, _POWER_CANDIDATES)
    if pwr_col is None:
        raise ValueError("results must include a battery power column")
    detected = results.loc[_detected_mask(results)].copy()
    detected["_pwr"] = pd.to_numeric(detected[pwr_col], errors="coerce")
    detected = detected.dropna(subset=["_pwr"])
    if detected.empty:
        raise ValueError("No detected customers with power")
    pv = _is_pv_customer(detected)
    return _grouped_boxplot(
        [
            ("PV customers", detected.loc[pv, "_pwr"].values, _RED),
            ("Non-PV customers", detected.loc[~pv, "_pwr"].values, _GREY),
            ("All detected", detected["_pwr"].values, _BLACK),
        ],
        title="Battery power distribution — detected customers",
        ylabel="Estimated battery power (kW)",
    )


def plot_battery_capacity_vs_power_scatter(results: pd.DataFrame):
    import matplotlib.pyplot as plt

    cap_col = _pick_column(results.columns, _CAPACITY_CANDIDATES)
    pwr_col = _pick_column(results.columns, _POWER_CANDIDATES)
    if cap_col is None or pwr_col is None:
        raise ValueError("results must include capacity and power columns")
    detected = results.loc[_detected_mask(results)].copy()
    detected["_cap"] = pd.to_numeric(detected[cap_col], errors="coerce")
    detected["_pwr"] = pd.to_numeric(detected[pwr_col], errors="coerce")
    detected = detected.dropna(subset=["_cap", "_pwr"])
    if detected.empty:
        raise ValueError("No detected customers with capacity and power")
    pv = _is_pv_customer(detected)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(detected.loc[pv, "_cap"], detected.loc[pv, "_pwr"],
               color=_RED, alpha=0.65, s=40, label=f"PV customers (n={pv.sum()})")
    ax.scatter(detected.loc[~pv, "_cap"], detected.loc[~pv, "_pwr"],
               color=_GREY, alpha=0.65, s=40, label=f"Non-PV customers (n={(~pv).sum()})")
    ax.set_title("Battery capacity vs. estimated power — detected customers")
    ax.set_xlabel("Estimated battery capacity (kWh)")
    ax.set_ylabel("Estimated battery power (kW)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pv_vs_battery_capacity_scatter(results: pd.DataFrame):
    import matplotlib.pyplot as plt

    cap_col = _pick_column(results.columns, _CAPACITY_CANDIDATES)
    pv_cap_col = _pick_column(results.columns, _PV_CAP_CANDIDATES)
    if cap_col is None or pv_cap_col is None:
        raise ValueError("results must include battery capacity and PV capacity columns")
    pv_bat = results.loc[_detected_mask(results) & _is_pv_customer(results)].copy()
    pv_bat["_bat"] = pd.to_numeric(pv_bat[cap_col], errors="coerce")
    pv_bat["_pv"] = pd.to_numeric(pv_bat[pv_cap_col], errors="coerce")
    pv_bat = pv_bat.dropna(subset=["_bat", "_pv"])
    if len(pv_bat) < 3:
        raise ValueError("Need ≥3 PV+battery customers for scatter")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pv_bat["_pv"], pv_bat["_bat"], color=_RED, alpha=0.65, s=40)
    ax.set_title(f"PV capacity vs. battery capacity\n(PV + battery customers, n={len(pv_bat)})")
    ax.set_xlabel("PV system capacity (kWp)")
    ax.set_ylabel("Estimated battery capacity (kWh)")
    ax.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def battery_portfolio_summary(
    results: pd.DataFrame,
    threshold: float = 0.5,
    reliability_margin: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, segment_overview_df) of portfolio-level battery metrics."""
    pv = _is_pv_customer(results)
    detected = _detected_mask(results)
    reliable = _battery_reliable_mask(results, threshold=threshold, reliability_margin=reliability_margin)
    probs_pct = _prob_pct(results)

    cap_col = _pick_column(results.columns, _CAPACITY_CANDIDATES)
    pwr_col = _pick_column(results.columns, _POWER_CANDIDATES)
    cap_series = pd.to_numeric(results.loc[detected, cap_col], errors="coerce").dropna() if cap_col else pd.Series(dtype=float)
    pwr_series = pd.to_numeric(results.loc[detected, pwr_col], errors="coerce").dropna() if pwr_col else pd.Series(dtype=float)

    n_total = int(len(results))
    n_pv = int(pv.sum())
    n_non_pv = n_total - n_pv
    n_det = int(detected.sum())
    n_det_pv = int((detected & pv).sum())
    n_det_non_pv = n_det - n_det_pv

    def _q(s, q):
        return round(float(s.quantile(q)), 3) if not s.empty else np.nan

    summary = pd.DataFrame(
        [
            ("threshold_pct", round(threshold * 100.0, 2)),
            ("reliability_margin_pct", round(reliability_margin * 100.0, 2)),
            ("total_customers", n_total),
            ("total_pv_customers", n_pv),
            ("total_non_pv_customers", n_non_pv),
            ("detected_battery_total", n_det),
            ("detected_battery_pv", n_det_pv),
            ("detected_battery_non_pv", n_det_non_pv),
            ("detected_share_of_all_customers_pct", round(100.0 * n_det / max(n_total, 1), 2)),
            ("detected_share_of_pv_customers_pct", round(100.0 * n_det_pv / max(n_pv, 1), 2)),
            ("detected_non_pv_share_of_detected_pct", round(100.0 * n_det_non_pv / max(n_det, 1), 2)),
            ("reliable_total", int(reliable.sum())),
            ("reliable_detected", int((detected & reliable).sum())),
            ("reliable_detected_share_pct", round(100.0 * int((detected & reliable).sum()) / max(n_det, 1), 2)),
            ("avg_detected_battery_probability_pct",
             round(float(probs_pct.loc[detected].mean()), 2) if n_det else np.nan),
            ("median_detected_battery_probability_pct",
             round(float(probs_pct.loc[detected].median()), 2) if n_det else np.nan),
            ("avg_detected_capacity_kwh", _q(cap_series, 0.5) if cap_series.empty else round(float(cap_series.mean()), 3)),
            ("median_detected_capacity_kwh", _q(cap_series, 0.5)),
            ("p25_detected_capacity_kwh", _q(cap_series, 0.25)),
            ("p75_detected_capacity_kwh", _q(cap_series, 0.75)),
            ("avg_detected_power_kw", _q(pwr_series, 0.5) if pwr_series.empty else round(float(pwr_series.mean()), 3)),
            ("median_detected_power_kw", _q(pwr_series, 0.5)),
            ("p25_detected_power_kw", _q(pwr_series, 0.25)),
            ("p75_detected_power_kw", _q(pwr_series, 0.75)),
        ],
        columns=["metric", "value"],
    )

    overview = pd.DataFrame(
        [
            {
                "segment": "PV customers",
                "customers": n_pv,
                "detected_battery": n_det_pv,
                "not_detected": n_pv - n_det_pv,
                "detected_share_pct": round(100.0 * n_det_pv / max(n_pv, 1), 2),
            },
            {
                "segment": "Non-PV customers",
                "customers": n_non_pv,
                "detected_battery": n_det_non_pv,
                "not_detected": n_non_pv - n_det_non_pv,
                "detected_share_pct": round(100.0 * n_det_non_pv / max(n_non_pv, 1), 2),
            },
            {
                "segment": "Total portfolio",
                "customers": n_total,
                "detected_battery": n_det,
                "not_detected": n_total - n_det,
                "detected_share_pct": round(100.0 * n_det / max(n_total, 1), 2),
            },
        ]
    )
    return summary, overview


def _save(fig, output_path: Path, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_battery_capacity_histogram(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_capacity_histogram(results), output_path=output_path, dpi=dpi)


def save_battery_reliability_summary(
    results: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.5,
    reliability_margin: float = 0.15,
    dpi: int = 150,
) -> Path:
    fig = plot_battery_reliability_summary(
        results, threshold=threshold, reliability_margin=reliability_margin,
    )
    return _save(fig, output_path=output_path, dpi=dpi)


def save_battery_probability_distribution(
    results: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.5,
    dpi: int = 150,
) -> Path:
    return _save(plot_battery_probability_distribution(results, threshold=threshold), output_path=output_path, dpi=dpi)


def save_battery_detected_share_pie(
    results: pd.DataFrame, output_path: Path, threshold: float = 0.5, dpi: int = 150,
) -> Path:
    return _save(plot_battery_detected_share_pie(results, threshold=threshold), output_path=output_path, dpi=dpi)


def save_battery_status_breakdown(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_status_breakdown(results), output_path=output_path, dpi=dpi)


def save_battery_pv_split_pie(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_pv_split_pie(results), output_path=output_path, dpi=dpi)


def save_battery_capacity_boxplot(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_capacity_boxplot(results), output_path=output_path, dpi=dpi)


def save_battery_power_distribution(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_power_distribution(results), output_path=output_path, dpi=dpi)


def save_battery_power_boxplot(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_power_boxplot(results), output_path=output_path, dpi=dpi)


def save_battery_capacity_vs_power_scatter(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_battery_capacity_vs_power_scatter(results), output_path=output_path, dpi=dpi)


def save_pv_vs_battery_capacity_scatter(results: pd.DataFrame, output_path: Path, dpi: int = 150) -> Path:
    return _save(plot_pv_vs_battery_capacity_scatter(results), output_path=output_path, dpi=dpi)


def save_battery_portfolio_summary_tables(
    results: pd.DataFrame,
    output_dir: Path,
    threshold: float = 0.5,
    reliability_margin: float = 0.15,
) -> dict[str, Path]:
    """Write portfolio summary CSVs. Returns {name: path}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df, overview_df = battery_portfolio_summary(
        results, threshold=threshold, reliability_margin=reliability_margin,
    )
    summary_path = output_dir / "battery_portfolio_summary_table.csv"
    overview_path = output_dir / "battery_portfolio_overview_table.csv"
    summary_df.to_csv(summary_path, index=False)
    overview_df.to_csv(overview_path, index=False)
    return {"summary": summary_path, "overview": overview_path}
