"""Legacy-compatible portfolio aggregation/evaluation helpers from pv_detection.py."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _capacity_weighted_sc_aggregate(sc_df: pd.DataFrame) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    Capacity-weighted mean of ``sc_share_mean`` and conservative / independence CIs.

    Expects columns: ``pv_capacity_kwp``, ``sc_share_mean``,
    ``sc_share_ci_lower``, ``sc_share_ci_upper``.
    """
    if sc_df.empty:
        return float(np.nan), (float(np.nan), float(np.nan)), (float(np.nan), float(np.nan))
    weights = sc_df["pv_capacity_kwp"]
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return float(np.nan), (float(np.nan), float(np.nan)), (float(np.nan), float(np.nan))

    agg_sc = float((weights * sc_df["sc_share_mean"]).sum() / w_sum)
    sc_lo = sc_df["sc_share_ci_lower"].fillna(sc_df["sc_share_mean"])
    sc_hi = sc_df["sc_share_ci_upper"].fillna(sc_df["sc_share_mean"])
    agg_sc_ci_conservative = (
        float(np.clip((weights * sc_lo).sum() / w_sum, 0, 1)),
        float(np.clip((weights * sc_hi).sum() / w_sum, 0, 1)),
    )
    sc_sigma_i = (sc_hi - sc_lo) / (2.0 * 1.96)
    sc_sigma_agg = float(np.sqrt(((weights ** 2) * (sc_sigma_i ** 2)).sum()) / w_sum)
    agg_sc_ci_independence = (
        float(np.clip(agg_sc - 1.96 * sc_sigma_agg, 0, 1)),
        float(np.clip(agg_sc + 1.96 * sc_sigma_agg, 0, 1)),
    )
    return agg_sc, agg_sc_ci_conservative, agg_sc_ci_independence


def aggregate_portfolio_estimates(
    prob_summary: pd.DataFrame,
    pv_indicators: Optional[pd.DataFrame] = None,
) -> dict:
    """Aggregate per-customer PV capacity and self-consumption to portfolio level."""
    if prob_summary.empty:
        return {}

    df = prob_summary.copy()

    valid = df["pv_capacity_kwp"].notna() & (df["pv_capacity_kwp"] > 0)
    df_valid = df[valid]
    n = len(df_valid)
    if n == 0:
        return {}

    total_hybrid = float(df_valid["pv_capacity_kwp"].sum())
    total_regression = float(df_valid["pv_capacity_kwp_regression_only"].fillna(0).sum())
    total_floor = float(df_valid["pv_capacity_kwp_floor"].fillna(0).sum())

    ci_lo = df_valid["pv_capacity_kwp_ci_lower"].fillna(df_valid["pv_capacity_kwp"])
    ci_hi = df_valid["pv_capacity_kwp_ci_upper"].fillna(df_valid["pv_capacity_kwp"])
    sigma_i = (ci_hi - ci_lo) / (2.0 * 1.96)

    cap_ci_conservative = (float(ci_lo.sum()), float(ci_hi.sum()))
    sigma_agg = float(np.sqrt((sigma_i ** 2).sum()))
    cap_ci_independence = (
        max(0.0, total_hybrid - 1.96 * sigma_agg),
        total_hybrid + 1.96 * sigma_agg,
    )

    sc_valid = df_valid.dropna(subset=["sc_share_mean"])
    agg_sc, agg_sc_ci_conservative, agg_sc_ci_independence = _capacity_weighted_sc_aggregate(
        sc_valid
    )

    sc_nonzero = sc_valid[sc_valid["sc_share_mean"] > 0]
    n_nonzero_sc = int(len(sc_nonzero))
    agg_sc_nz, agg_sc_ci_con_nz, agg_sc_ci_ind_nz = _capacity_weighted_sc_aggregate(sc_nonzero)

    portfolio_f2r = total_floor / total_regression if total_regression > 0 else np.nan

    caps = df_valid["pv_capacity_kwp"]
    capacity_stats = {
        "mean": float(caps.mean()),
        "median": float(caps.median()),
        "std": float(caps.std()),
        "min": float(caps.min()),
        "max": float(caps.max()),
    }

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
        "aggregate_sc_share_nonzero_sc": agg_sc_nz,
        "aggregate_sc_ci_conservative_nonzero_sc": agg_sc_ci_con_nz,
        "aggregate_sc_ci_independence_nonzero_sc": agg_sc_ci_ind_nz,
        "n_customers_nonzero_sc": n_nonzero_sc,
        "portfolio_floor_to_reg_ratio": portfolio_f2r,
        "capacity_stats": capacity_stats,
        **yearly_totals,
    }
    return result


def validate_yield_plausibility(
    prob_summary: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    yield_low: float = 800.0,
    yield_high: float = 1200.0,
) -> pd.DataFrame:
    """Per-customer specific-yield plausibility check."""
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


def compute_segment_stats(
    prob_summary: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    min_segment_size: int = 30,
) -> pd.DataFrame:
    """Per-segment capacity distribution stats for outlier detection."""
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
    """Flag customers with implausible capacity estimates using segment fences."""
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


def evaluate_portfolio(
    prob_summary: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    yield_bounds: tuple = (800.0, 1200.0),
) -> dict:
    """Comprehensive portfolio evaluation."""
    result: dict = {}

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

    if seg_stats.shape[0] > 1:
        seg_summary = seg_stats[["segment", "n_customers", "cap_median", "cap_mean"]].copy()
        result["cross_segment"] = seg_summary
    else:
        result["cross_segment"] = seg_stats

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
    print("\n--- Yield Plausibility ---")
    print(f"  Specific yield: median = {ys.get('median', float('nan')):.0f}, "
          f"mean = {ys.get('mean', float('nan')):.0f}, "
          f"std = {ys.get('std', float('nan')):.0f} kWh/kWp")
    for flag, count in ys.get("flag_counts", {}).items():
        print(f"  {flag}: {count} customers")

    ea = evaluation.get("estimator_agreement", {})
    if ea:
        print("\n--- Estimator Agreement (floor vs regression) ---")
        print(f"  Floor/Reg ratio: median = {ea.get('floor_to_reg_ratio_median', float('nan')):.2f}, "
              f"mean = {ea.get('floor_to_reg_ratio_mean', float('nan')):.2f}")
        print(f"  Floor > Regression: {ea.get('pct_floor_gt_regression', float('nan')):.1f}%")
        print(f"  Within ±30%: {ea.get('pct_within_30pct', float('nan')):.1f}%")

    print("\n--- Capacity Flags ---")
    for flag, count in evaluation.get("capacity_flag_counts", {}).items():
        print(f"  {flag}: {count}")

    cs = evaluation.get("cross_segment")
    if cs is not None and not cs.empty:
        print("\n--- Cross-Segment Consistency ---")
        print(cs.to_string(index=False))

    pc = evaluation.get("portfolio_check", {})
    if pc:
        print("\n--- Portfolio-Level Check ---")
        print(f"  Total capacity: {pc.get('total_capacity_kwp', float('nan')):.0f} kWp")
        print(f"  Total yearly export: {pc.get('total_yearly_export_kwh', float('nan')):.0f} kWh")
        print(f"  Implied yield: {pc.get('implied_yield_kwh_kwp', float('nan')):.0f} kWh/kWp")
    print("=" * 70)
