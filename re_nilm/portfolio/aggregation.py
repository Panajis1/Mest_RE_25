"""Portfolio-level aggregation of per-customer detection and capacity estimates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Delegate to pv_detection.py for the heavy aggregation logic
_PV_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_PV_DIR) not in sys.path:
    sys.path.insert(0, str(_PV_DIR))

try:
    from pv_detection import (
        aggregate_portfolio_estimates as _aggregate_portfolio_estimates,
        compute_segment_stats as _compute_segment_stats,
    )
    _PV_AVAILABLE = True
except ImportError:
    _PV_AVAILABLE = False


def aggregate_portfolio_estimates(
    results: pd.DataFrame,
    pv_indicators: Optional[pd.DataFrame] = None,
) -> dict:
    """Aggregate per-customer PV capacity and self-consumption to portfolio level.

    Computes two uncertainty bounds:
    - Conservative (perfect correlation): sum of individual CI endpoints.
    - Independence (zero correlation): propagate via sqrt-sum-of-variances.

    Args:
        results: Per-customer results DataFrame with columns [customer_id,
            pv_capacity_kwp, pv_capacity_kwp_ci_lower, pv_capacity_kwp_ci_upper,
            sc_share_mean, ...].
        pv_indicators: Optional DataFrame with yearly production/consumption totals.

    Returns:
        Dict with portfolio-level capacity, self-consumption, and uncertainty stats.
    """
    if results.empty:
        return {}
    if not _PV_AVAILABLE:
        # Minimal fallback aggregation when pv_detection is not importable
        valid = results["pv_capacity_kwp"].notna() & (results["pv_capacity_kwp"] > 0)
        df = results[valid]
        return {
            "n_customers": len(df),
            "total_pv_capacity_kwp": float(df["pv_capacity_kwp"].sum()),
            "mean_pv_capacity_kwp": float(df["pv_capacity_kwp"].mean()),
        }
    return _aggregate_portfolio_estimates(results, pv_indicators)


def compute_segment_stats(
    results: pd.DataFrame,
    segment_col: str,
    value_col: str = "pv_capacity_kwp",
) -> pd.DataFrame:
    """Compute summary statistics broken down by a segment column.

    Args:
        results: Per-customer results DataFrame.
        segment_col: Column to segment by (e.g. 'has_battery', 'hp_type').
        value_col: Column to aggregate (default: pv_capacity_kwp).

    Returns:
        DataFrame with one row per segment and statistics: n, mean, median, std, sum.
    """
    if _PV_AVAILABLE:
        return _compute_segment_stats(results, segment_col, value_col)

    # Fallback
    return (
        results.groupby(segment_col)[value_col]
        .agg(n="count", mean="mean", median="median", std="std", total="sum")
        .reset_index()
    )


def capacity_weighted_sc_aggregate(
    results: pd.DataFrame,
    capacity_col: str = "pv_capacity_kwp",
    sc_col: str = "sc_share",
) -> float:
    """Compute capacity-weighted average self-consumption share across the portfolio.

    Args:
        results: Per-customer results with capacity and SC columns.
        capacity_col: Column with installed capacity values.
        sc_col: Column with per-customer SC share [0, 1].

    Returns:
        Weighted SC share (float), or NaN if inputs are empty.
    """
    valid = results[[capacity_col, sc_col]].dropna()
    valid = valid[(valid[capacity_col] > 0) & (valid[sc_col] >= 0)]
    if valid.empty:
        return np.nan
    weights = valid[capacity_col]
    return float((valid[sc_col] * weights).sum() / weights.sum())
