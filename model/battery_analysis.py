from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_CANDIDATES = ("ID", "customer_id", "id_customer", "id")
PROB_CANDIDATES = ("battery_prob", "battery_probability")
STATUS_CANDIDATES = ("status",)
CAPACITY_CANDIDATES = ("estimated_battery_capacity_kwh", "capacity_kwh")
PV_FLAG_CANDIDATES = ("is_pv_customer",)
PV_PROB_CANDIDATES = ("has_pv_prob",)
PV_CAP_CANDIDATES = ("pv_capacity_kwp", "pv_capacity_ci_upper", "pv_capacity_kwp_floor")
COLOR_BLACK = "#111111"
COLOR_RED = "#C62828"


def _pick_column(columns, candidates):
    by_lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def _normalize_probability_to_pct(df: pd.DataFrame, prob_col: str) -> pd.Series:
    prob_raw = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
    if prob_raw.max() <= 1.0:
        return (prob_raw * 100.0).clip(0.0, 100.0)
    return prob_raw.clip(0.0, 100.0)


def _infer_is_pv_customer(df: pd.DataFrame) -> pd.Series:
    pv_flag_col = _pick_column(df.columns, PV_FLAG_CANDIDATES)
    if pv_flag_col is not None:
        raw = df[pv_flag_col].astype(str).str.strip().str.lower()
        return raw.isin(("yes", "true", "1"))

    pv_prob_col = _pick_column(df.columns, PV_PROB_CANDIDATES)
    if pv_prob_col is not None:
        pv_prob = pd.to_numeric(df[pv_prob_col], errors="coerce")
        return pv_prob.fillna(0.0) >= 0.5

    pv_cap_col = _pick_column(df.columns, PV_CAP_CANDIDATES)
    if pv_cap_col is not None:
        pv_cap = pd.to_numeric(df[pv_cap_col], errors="coerce")
        return pv_cap.fillna(0.0) > 0.0

    return pd.Series(False, index=df.index)


def _compute_reliability_flag(
    df: pd.DataFrame,
    status_col: str | None,
    threshold_pct: float,
    reliability_margin_pct: float,
) -> pd.Series:
    prob_margin = (df["battery_prob_pct"] - threshold_pct).abs()
    if status_col is None:
        status_ok = pd.Series(True, index=df.index)
    else:
        status_text = df[status_col].fillna("unknown").astype(str).str.strip().str.lower()
        status_ok = status_text.str.startswith("success") | status_text.eq("ok")

    # A result is considered reliable when status is successful and the score
    # is sufficiently far from the decision threshold.
    reliable = status_ok & (prob_margin >= reliability_margin_pct)
    return reliable.astype(bool)


def _load_and_prepare(
    input_csv: Path,
    threshold_pct: float,
    reliability_margin_pct: float,
) -> tuple[pd.DataFrame, str | None]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input file has no rows: {input_csv}")

    id_col = _pick_column(df.columns, ID_CANDIDATES)
    if id_col is not None:
        df[id_col] = df[id_col].astype(str)

    prob_col = _pick_column(df.columns, PROB_CANDIDATES)
    if prob_col is None:
        raise ValueError(
            "No battery probability column found. Expected one of: "
            f"{', '.join(PROB_CANDIDATES)}"
        )

    df["battery_prob_pct"] = _normalize_probability_to_pct(df, prob_col)
    df["detected_battery"] = df["battery_prob_pct"] >= threshold_pct
    df["is_pv_customer_bool"] = _infer_is_pv_customer(df)

    if id_col is not None:
        df = (
            df.sort_values("battery_prob_pct", ascending=False)
            .drop_duplicates(subset=id_col, keep="first")
            .reset_index(drop=True)
        )

    status_col = _pick_column(df.columns, STATUS_CANDIDATES)
    df["is_reliable_detection"] = _compute_reliability_flag(
        df=df,
        status_col=status_col,
        threshold_pct=threshold_pct,
        reliability_margin_pct=reliability_margin_pct,
    )
    return df, status_col


def _plot_pie_detected_share(df: pd.DataFrame, output_dir: Path, threshold_pct: float):
    pv_df = df[df["is_pv_customer_bool"]]
    total_pv_count = int(len(pv_df))
    detected_count = int(pv_df["detected_battery"].sum())
    non_detected_count = total_pv_count - detected_count

    if total_pv_count == 0:
        return

    plt.figure(figsize=(6.5, 6.5))
    plt.pie(
        [non_detected_count, detected_count],
        labels=["PV customers without detected battery", "PV customers with detected battery"],
        colors=[COLOR_BLACK, COLOR_RED],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": "white"},
    )
    plt.title(
        f"Detected battery share among PV customers\n"
        f"(threshold = {threshold_pct:.1f}%, n_pv = {total_pv_count})"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "battery_detected_share_pie.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_probability_distribution(df: pd.DataFrame, output_dir: Path, threshold_pct: float):
    detected = df.loc[df["detected_battery"], "battery_prob_pct"]
    not_detected = df.loc[~df["detected_battery"], "battery_prob_pct"]
    bins = np.arange(0, 101, 2)
    total_count = max(len(df), 1)

    count_no, edges = np.histogram(not_detected, bins=bins)
    count_yes, _ = np.histogram(detected, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    plt.figure(figsize=(11, 5))
    plt.bar(
        centers,
        count_no / total_count,
        width=widths,
        color=COLOR_BLACK,
        label="No battery detected",
    )
    plt.bar(
        centers,
        count_yes / total_count,
        width=widths,
        bottom=count_no / total_count,
        color=COLOR_RED,
        label="Battery detected",
    )
    plt.axvline(threshold_pct, color=COLOR_BLACK, linestyle="--", linewidth=1.2, label="Threshold")
    plt.title("Battery probability distribution")
    plt.xlabel("Battery probability (%)")
    plt.ylabel("Share of customers")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "battery_probability_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_status_breakdown(df: pd.DataFrame, output_dir: Path, status_col: str | None):
    if status_col is None:
        return

    status_counts = df[status_col].fillna("Unknown").astype(str).value_counts().head(12)
    if status_counts.empty:
        return

    plt.figure(figsize=(11, 5))
    colors = [COLOR_BLACK if i % 2 == 0 else COLOR_RED for i in range(len(status_counts))]
    plt.bar(status_counts.index, status_counts.values, color=colors)
    plt.title("Detection status breakdown")
    plt.xlabel("Status")
    plt.ylabel("Customers")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "battery_detection_status_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_capacity_distribution(df: pd.DataFrame, output_dir: Path):
    capacity_col = _pick_column(df.columns, CAPACITY_CANDIDATES)
    if capacity_col is None:
        return

    detected_capacity = pd.to_numeric(
        df.loc[df["detected_battery"], capacity_col],
        errors="coerce",
    ).dropna()
    if detected_capacity.empty:
        return

    plt.figure(figsize=(10, 5))
    bins = max(10, min(40, int(np.sqrt(len(detected_capacity)) * 3)))
    plt.hist(detected_capacity, bins=bins, color=COLOR_RED, edgecolor=COLOR_BLACK)
    plt.title("Estimated battery capacity among detected battery customers")
    plt.xlabel("Estimated battery capacity (kWh)")
    plt.ylabel("Customers")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "battery_capacity_detected_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_pv_split_among_detected(df: pd.DataFrame, output_dir: Path):
    detected = df[df["detected_battery"]]
    if detected.empty:
        return

    pv_detected = int(detected["is_pv_customer_bool"].sum())
    non_pv_detected = int(len(detected) - pv_detected)

    plt.figure(figsize=(6.5, 6.5))
    plt.pie(
        [non_pv_detected, pv_detected],
        labels=["Detected battery, no PV flag", "Detected battery + PV flag"],
        colors=[COLOR_BLACK, COLOR_RED],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": "white"},
    )
    plt.title(f"PV split among detected battery customers\n(n = {len(detected)})")
    plt.tight_layout()
    plt.savefig(output_dir / "battery_detected_pv_split_pie.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_reliability_assessment(
    df: pd.DataFrame,
    output_dir: Path,
    threshold_pct: float,
    reliability_margin_pct: float,
):
    reliable_count = int(df["is_reliable_detection"].sum())
    needs_review_count = int(len(df) - reliable_count)
    detected = df[df["detected_battery"]]
    reliable_detected = int(detected["is_reliable_detection"].sum())
    review_detected = int(len(detected) - reliable_detected)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(
        [reliable_count, needs_review_count],
        labels=["Reliable", "Needs review"],
        colors=[COLOR_BLACK, COLOR_RED],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"color": "white"},
    )
    axes[0].set_title(
        "Overall reliability assessment\n"
        f"(status success + |prob - {threshold_pct:.0f}%| >= {reliability_margin_pct:.0f}%)"
    )

    axes[1].bar(
        ["Reliable detected", "Detected needs review"],
        [reliable_detected, review_detected],
        color=[COLOR_BLACK, COLOR_RED],
    )
    axes[1].set_title("Reliability among detected battery customers")
    axes[1].set_ylabel("Customers")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "battery_detection_reliability_assessment.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_portfolio_summary_tables(
    df: pd.DataFrame,
    output_dir: Path,
    threshold_pct: float,
    reliability_margin_pct: float,
):
    total_customers = int(len(df))
    total_pv_customers = int(df["is_pv_customer_bool"].sum())
    total_non_pv_customers = total_customers - total_pv_customers

    detected_total = int(df["detected_battery"].sum())
    detected_pv = int((df["detected_battery"] & df["is_pv_customer_bool"]).sum())
    detected_non_pv = int((df["detected_battery"] & ~df["is_pv_customer_bool"]).sum())

    reliable_total = int(df["is_reliable_detection"].sum())
    reliable_detected = int((df["is_reliable_detection"] & df["detected_battery"]).sum())

    capacity_series = pd.to_numeric(
        df.loc[df["detected_battery"], _pick_column(df.columns, CAPACITY_CANDIDATES)] if _pick_column(df.columns, CAPACITY_CANDIDATES) else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()

    summary_rows = [
        ("threshold_pct", float(threshold_pct)),
        ("reliability_margin_pct", float(reliability_margin_pct)),
        ("total_customers", total_customers),
        ("total_pv_customers", total_pv_customers),
        ("total_non_pv_customers", total_non_pv_customers),
        ("detected_battery_total", detected_total),
        ("detected_battery_pv", detected_pv),
        ("detected_battery_non_pv", detected_non_pv),
        (
            "detected_share_of_all_customers_pct",
            round(100.0 * detected_total / max(total_customers, 1), 2),
        ),
        (
            "detected_share_of_pv_customers_pct",
            round(100.0 * detected_pv / max(total_pv_customers, 1), 2),
        ),
        (
            "detected_non_pv_share_of_detected_pct",
            round(100.0 * detected_non_pv / max(detected_total, 1), 2),
        ),
        ("reliable_total", reliable_total),
        ("reliable_detected", reliable_detected),
        (
            "reliable_detected_share_pct",
            round(100.0 * reliable_detected / max(detected_total, 1), 2),
        ),
        (
            "avg_detected_battery_probability_pct",
            round(float(df.loc[df["detected_battery"], "battery_prob_pct"].mean()), 2)
            if detected_total > 0
            else np.nan,
        ),
        (
            "median_detected_battery_probability_pct",
            round(float(df.loc[df["detected_battery"], "battery_prob_pct"].median()), 2)
            if detected_total > 0
            else np.nan,
        ),
        (
            "avg_detected_capacity_kwh",
            round(float(capacity_series.mean()), 3) if not capacity_series.empty else np.nan,
        ),
        (
            "median_detected_capacity_kwh",
            round(float(capacity_series.median()), 3) if not capacity_series.empty else np.nan,
        ),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_df.to_csv(output_dir / "battery_portfolio_summary_table.csv", index=False)

    overview_df = pd.DataFrame(
        [
            {
                "segment": "PV customers",
                "customers": int(total_pv_customers),
                "detected_battery": int(detected_pv),
                "not_detected": int(total_pv_customers - detected_pv),
                "detected_share_pct": round(100.0 * detected_pv / max(total_pv_customers, 1), 2),
            },
            {
                "segment": "Non-PV customers",
                "customers": int(total_non_pv_customers),
                "detected_battery": int(detected_non_pv),
                "not_detected": int(total_non_pv_customers - detected_non_pv),
                "detected_share_pct": round(100.0 * detected_non_pv / max(total_non_pv_customers, 1), 2),
            },
            {
                "segment": "Total portfolio",
                "customers": int(total_customers),
                "detected_battery": int(detected_total),
                "not_detected": int(total_customers - detected_total),
                "detected_share_pct": round(100.0 * detected_total / max(total_customers, 1), 2),
            },
        ]
    )
    overview_df.to_csv(output_dir / "battery_portfolio_overview_table.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate key customer battery-analysis plots from a battery results CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("battery_residential_results_v7.csv"),
        help="Path to customer-level battery result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/battery_detection_plots"),
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=50.0,
        help="Battery detection threshold in percent (default: 50).",
    )
    parser.add_argument(
        "--reliability-margin-pct",
        type=float,
        default=15.0,
        help=(
            "Minimum absolute distance from threshold (in percentage points) "
            "to consider a classification reliable (default: 15)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df, status_col = _load_and_prepare(
        args.input_csv,
        args.threshold_pct,
        args.reliability_margin_pct,
    )

    _plot_pie_detected_share(results_df, args.output_dir, args.threshold_pct)
    _plot_probability_distribution(results_df, args.output_dir, args.threshold_pct)
    _plot_status_breakdown(results_df, args.output_dir, status_col)
    _plot_capacity_distribution(results_df, args.output_dir)
    _plot_pv_split_among_detected(results_df, args.output_dir)
    _plot_reliability_assessment(
        results_df,
        args.output_dir,
        args.threshold_pct,
        args.reliability_margin_pct,
    )
    _save_portfolio_summary_tables(
        results_df,
        args.output_dir,
        args.threshold_pct,
        args.reliability_margin_pct,
    )

    detected_count = int(results_df["detected_battery"].sum())
    total_count = int(len(results_df))
    total_pv = int(results_df["is_pv_customer_bool"].sum())
    detected_pv = int((results_df["detected_battery"] & results_df["is_pv_customer_bool"]).sum())
    detected_non_pv = int((results_df["detected_battery"] & ~results_df["is_pv_customer_bool"]).sum())
    print(f"Wrote plots to: {args.output_dir}")
    print(
        f"Detected battery customers: {detected_count}/{total_count} "
        f"({(100.0 * detected_count / max(total_count, 1)):.2f}%)"
    )
    print(
        f"Detected among PV customers: {detected_pv}/{total_pv} "
        f"({(100.0 * detected_pv / max(total_pv, 1)):.2f}%)"
    )
    print(
        f"Detected without PV flag: {detected_non_pv} "
        f"({(100.0 * detected_non_pv / max(detected_count, 1)):.2f}% of detected)"
    )
    print("Wrote summary tables:")
    print(f"- {args.output_dir / 'battery_portfolio_summary_table.csv'}")
    print(f"- {args.output_dir / 'battery_portfolio_overview_table.csv'}")


if __name__ == "__main__":
    main()