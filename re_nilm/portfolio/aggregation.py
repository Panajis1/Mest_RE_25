"""Portfolio-level aggregation of per-customer detection and capacity estimates."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from re_nilm.portfolio._pv_portfolio_v1 import (
    aggregate_portfolio_estimates as _aggregate_portfolio_estimates,
    compute_segment_stats as _compute_segment_stats,
)


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
    return _aggregate_portfolio_estimates(results, pv_indicators)


def compute_segment_stats(
    results: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    min_segment_size: int = 30,
) -> pd.DataFrame:
    """Compute per-segment capacity fence statistics (legacy-compatible).

    Args:
        results: Per-customer results DataFrame.
        metadata: Optional customer metadata for segmentation.
        segment_col: Metadata column used for segment assignment.
        min_segment_size: Segments smaller than this are merged into 'other'.

    Returns:
        DataFrame of segment-level capacity statistics and IQR fences.
    """
    return _compute_segment_stats(results, metadata, segment_col, min_segment_size)


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
