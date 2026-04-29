"""Portfolio evaluation: plausibility checks and yield validation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from re_nilm.portfolio._pv_portfolio_v1 import (
    evaluate_portfolio as _evaluate_portfolio,
    flag_implausible_estimates as _flag_implausible_estimates,
)


def evaluate_portfolio(
    results: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    yield_bounds: tuple = (800.0, 1200.0),
) -> dict:
    """Run the full portfolio evaluation suite.

    Checks:
    - Capacity plausibility via segment IQR fences.
    - Specific yield within expected bounds.
    - Self-consumption share validity.

    Args:
        results: Per-customer scalar results (from results_all_customers.parquet).
        pv_indicators: Per-customer annual production/consumption totals.
        metadata: Optional customer metadata (for segmentation).
        segment_col: Metadata column to segment by.
        yield_bounds: (min, max) kWh/kWp/year for plausible yield.

    Returns:
        Dict with evaluation summary statistics and per-customer flags.
    """
    return _evaluate_portfolio(results, pv_indicators, metadata, segment_col, yield_bounds)


def flag_implausible_estimates(
    results: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    segment_col: str = "customer_type",
    yield_bounds: tuple = (800.0, 1200.0),
) -> pd.DataFrame:
    """Flag per-customer capacity estimates that fall outside plausible bounds.

    Returns results with added columns:
    - segment: customer segment
    - capacity_flag: 'ok', 'too_low', 'too_high', or 'no_data'

    Args:
        results: Per-customer results with pv_capacity_kwp column.
        metadata: Optional customer metadata for segmentation.
        segment_col: Metadata column to segment by.
        yield_bounds: (min, max) kWh/kWp/year for plausible specific yield.
    """
    # flag_implausible_estimates requires segment_stats; derive them here
    from re_nilm.portfolio.aggregation import compute_segment_stats

    seg_stats = compute_segment_stats(results, metadata=metadata, segment_col=segment_col)
    return _flag_implausible_estimates(results, seg_stats, metadata, segment_col, yield_bounds)


def validate_yield_plausibility(
    results: pd.DataFrame,
    pv_indicators: pd.DataFrame,
    min_kwh_per_kwp: float = 600.0,
    max_kwh_per_kwp: float = 1400.0,
) -> pd.DataFrame:
    """Check that specific yield (kWh/kWp/year) is within a plausible range.

    Args:
        results: Must have [customer_id, pv_capacity_kwp].
        pv_indicators: Must have [customer_id, yearly_prod_kwh].
        min_kwh_per_kwp, max_kwh_per_kwp: Plausible yield bounds for Switzerland.

    Returns:
        Merged DataFrame with a 'yield_flag' column ('ok', 'low', 'high', 'missing').
    """
    merged = results.merge(
        pv_indicators[["customer_id", "yearly_prod_kwh"]],
        on="customer_id",
        how="left",
    )
    cap = merged["pv_capacity_kwp"]
    prod = merged["yearly_prod_kwh"]
    specific_yield = prod / cap.replace(0, np.nan)

    def _flag(sy):
        if pd.isna(sy):
            return "missing"
        if sy < min_kwh_per_kwp:
            return "low"
        if sy > max_kwh_per_kwp:
            return "high"
        return "ok"

    merged["specific_yield_kwh_kwp"] = specific_yield
    merged["yield_flag"] = specific_yield.apply(_flag)
    return merged
