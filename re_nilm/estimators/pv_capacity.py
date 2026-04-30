"""PV capacity and self-consumption estimator.

Uses migrated legacy-compatible helpers inside re_nilm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from re_nilm.estimators.base import AbstractEstimator
from re_nilm.estimators._pv_capacity_v1 import (
    _bootstrap_capacity_and_sc,
    _capacity_and_sc_from_data,
)
from re_nilm.features.pv_daily import build_customer_daily_features, compute_daily_weather


_STC_FACTOR = 4000.0  # (kWh/15min)/(W/m²) → kWp


def _compute_daily_weather_cached(weather: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Compute the daily weather summary, caching the result on the supplied dict.

    Cache is keyed by id(weather); the caller owns the dict's lifetime, so the
    cache is bounded by the estimator instance (not module-global).
    """
    key = id(weather)
    if key not in cache:
        w_indexed = weather.set_index("dt_utc")
        w_indexed.index.name = "timestamp"
        cache[key] = compute_daily_weather(w_indexed)
    return cache[key]


def _make_cust_days(customer_ts: pd.DataFrame, weather: pd.DataFrame, daily_weather_cache: dict):
    """Build (merged_15min, daily) DataFrames that match what pv_detection expects.

    Uses compute_daily_weather + build_customer_daily_features from pv_detection.py
    so that rad_bucket categorisation (monthly percentile-based) and midday aggregation
    (sum of 10-16h) exactly replicate the reference pipeline.
    """
    df = customer_ts.copy()
    df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")

    # Normalize to datetime64[us] — pandas 2.x merge_asof requires identical units
    df["DT_UTC"] = df["DT_UTC"].astype("datetime64[us]")
    wdf = weather.rename(columns={"dt_utc": "DT_UTC"})[["DT_UTC", "t_2m_C", "global_rad_W"]].copy()
    wdf["DT_UTC"] = wdf["DT_UTC"].astype("datetime64[us]")
    merged = pd.merge_asof(
        df.sort_values("DT_UTC"),
        wdf.sort_values("DT_UTC"),
        on="DT_UTC",
        direction="nearest",
        tolerance=pd.Timedelta("16min"),
    )
    merged["date"] = merged["DT_UTC"].dt.date

    daily_weather = _compute_daily_weather_cached(weather, daily_weather_cache)
    daily = build_customer_daily_features(merged, daily_weather)
    return merged, daily


class PVCapacityEstimator(AbstractEstimator):
    """Bootstrap PV capacity and self-consumption share estimator.

    Args:
        bootstrap_n: Number of bootstrap iterations.
        capacity_min_kwp: Floor for reported capacity (estimates below this → NaN).
        confidence_level: Bootstrap CI level (default 0.9 → [5th, 95th] percentiles).
    """

    def __init__(
        self,
        bootstrap_n: int = 200,
        capacity_min_kwp: float = 0.1,
        confidence_level: float = 0.9,
    ):
        self.bootstrap_n = bootstrap_n
        self.capacity_min_kwp = capacity_min_kwp
        self.confidence_level = confidence_level
        self._daily_weather_cache: dict = {}

    def estimate(
        self,
        customer_ts: pd.DataFrame,
        detection_result: dict,
        weather: pd.DataFrame,
    ) -> dict | None:
        customer_id = detection_result.get("customer_id", "")

        if not detection_result.get("has_pv", False):
            return None

        try:
            merged, daily = _make_cust_days(customer_ts, weather, self._daily_weather_cache)
            point = _capacity_and_sc_from_data(merged, daily)
            cap_samples, sc_samples = _bootstrap_capacity_and_sc(
                merged,
                daily,
                n_bootstrap=self.bootstrap_n,
                rng=np.random.default_rng(),
                stratify_by_month=True,
                block_size=1,
            )
        except Exception as exc:
            return {"customer_id": customer_id, "pv_capacity_kwp": np.nan, "error": str(exc)}

        hybrid_kwp = point.get("pv_capacity_hybrid_kwp", np.nan)

        ci_lo_pct = (1.0 - self.confidence_level) / 2.0 * 100
        ci_hi_pct = 100.0 - ci_lo_pct

        if cap_samples is not None and len(cap_samples) > 0:
            ci_lower_kwp = float(np.nanpercentile(cap_samples, ci_lo_pct)) * _STC_FACTOR
            ci_upper_kwp = float(np.nanpercentile(cap_samples, ci_hi_pct)) * _STC_FACTOR
        else:
            ci_lower_kwp = ci_upper_kwp = np.nan

        if sc_samples is not None and len(sc_samples) > 0:
            sc_med = float(np.nanmedian(sc_samples))
            sc_lo = float(np.nanpercentile(sc_samples, ci_lo_pct))
            sc_hi = float(np.nanpercentile(sc_samples, ci_hi_pct))
        else:
            sc_med = point.get("sc_share", np.nan)
            sc_lo = sc_hi = np.nan

        kwp = hybrid_kwp if not np.isnan(hybrid_kwp) else np.nan
        if not np.isnan(kwp) and kwp < self.capacity_min_kwp:
            kwp = np.nan

        return {
            "customer_id": customer_id,
            "pv_capacity_kwp": kwp,
            "pv_ci_lower": ci_lower_kwp,
            "pv_ci_upper": ci_upper_kwp,
            "sc_share": sc_med,
            "sc_ci_lower": sc_lo,
            "sc_ci_upper": sc_hi,
        }
