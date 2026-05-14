"""Battery detector — heuristic sigmoid scorer from internal battery v7 logic."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from re_nilm.detectors.base import AbstractDetector
from re_nilm.detectors._battery_v7 import analyze_battery_residential_v7 as _analyze_battery


class BatteryDetector(AbstractDetector):
    """Heuristic battery detector based on paired dark/sunny day load-shift analysis.

    Requires PV detection result when enforce_pv_required=True (the default),
    as batteries paired with PV show a distinctive evening load shift on sunny days.

    The detector is passed the pv_result dict in context['pv_result'].

    Scoring uses a logistic sigmoid over five signals: peak_shift, gap_ratio,
    injection_bonus, consistency, and shift_ratio. The raw probability is then
    multiplied by two physical guardrail scalers:
      - security_scaler (0.4): applied when avg_gap_kwh is outside [2, 30] kWh.
      - phys_scaler (0.7): applied when avg_gap_kwh exceeds 1.5× the mean
        full-day PV potential of matched sunny days (charges bounded by daily generation).

    Args:
        classification_threshold: Minimum prob_battery to call has_battery=True.
            Default 0.5.
        enforce_pv_required: If True, skip customers with has_pv=False. Default True.
        dark_day_rad_max_w: Max peak W/m² for a day to count as 'dark'. Default 100.
        sunny_day_rad_min_w: Min peak W/m² for a day to count as 'sunny'. Default 100.
        temp_buffer_c: Temperature matching tolerance when pairing dark/sunny days.
            Default 2.0 °C.
        sigmoid_intercept: Sigmoid bias term (more negative = stricter). Default -2.5.
        strict_min_matched_sunny_days: If fewer matched sunny days are found, apply a
            z-score penalty of `low_matched_days_z_penalty`. Default 7.
        low_matched_days_z_penalty: Penalty subtracted from z when matched days are
            below `strict_min_matched_sunny_days`. Default 0.5.
        nominal_capacity_discharge_fraction: Assumed depth of discharge per evening
            event, used to scale shift energy to capacity estimate. Default 0.40,
            reflecting typical daily cycling of residential batteries.
        pv_anchor_kwh_per_kwp: kWh capacity per kWp PV for the capacity anchor.
            Default 1.0, corresponding to the Swiss 1:1 rule of thumb.
        pv_anchor_blend_weight: Blend weight of the anchor in the final capacity
            estimate (0 = shift-only, 1 = anchor-only). Default 0.55.
        consumption_anchor_kwh_per_annual_kwh: Scales annual consumption to a capacity
            estimate (default 0.001 = 1 kWh per 1 000 kWh annual usage). The effective
            anchor is max(pv_anchor, consumption_anchor), so PV size is the floor and
            consumption raises it for larger households.
        min_dark_reference_days: Minimum number of dark reference days required to run
            analysis. Customers with fewer dark days are rejected as "No Baseline".
            Default 2.
        min_sunny_match_days: Minimum number of sunny days required (both before and
            after temperature-bin matching). Customers with fewer are rejected.
            Default 2.
    """

    def __init__(
        self,
        classification_threshold: float = 0.5,
        enforce_pv_required: bool = True,
        dark_day_rad_max_w: float = 100.0,
        sunny_day_rad_min_w: float = 100.0,
        temp_buffer_c: float = 2.0,
        sigmoid_intercept: float = -2.5,
        strict_min_matched_sunny_days: int = 7,
        low_matched_days_z_penalty: float = 0.5,
        nominal_capacity_discharge_fraction: float = 0.40,
        pv_anchor_kwh_per_kwp: float = 1.0,
        pv_anchor_blend_weight: float = 0.55,
        consumption_anchor_kwh_per_annual_kwh: float = 0.001,
        min_dark_reference_days: int = 2,
        min_sunny_match_days: int = 2,
    ):
        self.classification_threshold = classification_threshold
        self.enforce_pv_required = enforce_pv_required
        self.dark_day_rad_max_w = dark_day_rad_max_w
        self.sunny_day_rad_min_w = sunny_day_rad_min_w
        self.temp_buffer_c = temp_buffer_c
        self.sigmoid_intercept = sigmoid_intercept
        self.strict_min_matched_sunny_days = strict_min_matched_sunny_days
        self.low_matched_days_z_penalty = low_matched_days_z_penalty
        self.nominal_capacity_discharge_fraction = nominal_capacity_discharge_fraction
        self.pv_anchor_kwh_per_kwp = pv_anchor_kwh_per_kwp
        self.pv_anchor_blend_weight = pv_anchor_blend_weight
        self.consumption_anchor_kwh_per_annual_kwh = consumption_anchor_kwh_per_annual_kwh
        self.min_dark_reference_days = min_dark_reference_days
        self.min_sunny_match_days = min_sunny_match_days
        # Cache the indexed weather frame so it's not rebuilt per customer.
        # Keyed by id(weather_df); cache lifetime matches this detector instance.
        self._weather_indexed_cache: dict = {}

    def _get_indexed_weather(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        key = id(weather_df)
        cached = self._weather_indexed_cache.get(key)
        if cached is not None:
            return cached
        w = weather_df.copy()
        w["dt_utc"] = pd.to_datetime(w["dt_utc"], errors="coerce")
        w = w.dropna(subset=["dt_utc"]).set_index("dt_utc").sort_index()
        self._weather_indexed_cache[key] = w
        return w

    def predict_customer(
        self,
        customer_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        **context,
    ) -> dict | None:
        if customer_df.empty:
            return None

        customer_id = str(customer_df["ID"].iloc[0])
        pv_result = context.get("pv_result") or {}

        # PV guard
        if self.enforce_pv_required and not pv_result.get("has_pv", False):
            return {
                "customer_id": customer_id,
                "has_battery": False,
                "prob_battery": 0.0,
                "battery_status": "skipped_no_pv",
            }

        df = customer_df.copy()
        _dt = pd.to_datetime(df["DT_UTC"], errors="coerce")
        if _dt.dt.tz is not None:
            _dt = _dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df["DT_UTC"] = _dt
        df = df.dropna(subset=["DT_UTC"]).set_index("DT_UTC").sort_index()

        # Build a pv_row dict compatible with _battery_v7's expected keys.
        # prob_pv from PVDetector is forwarded as has_pv_prob for result reporting.
        pv_row = {
            "pv_capacity_kwp": pv_result.get("pv_capacity_kwp", 0.0),
            "has_pv": pv_result.get("has_pv", False),
            "has_pv_prob": pv_result.get("prob_pv", np.nan),
        }

        weather = self._get_indexed_weather(weather_df)

        try:
            result = _analyze_battery(
                df,
                weather,
                pv_row,
                temp_buffer=self.temp_buffer_c,
                sigmoid_intercept=self.sigmoid_intercept,
                strict_min_matched_sunny_days=self.strict_min_matched_sunny_days,
                low_matched_days_z_penalty=self.low_matched_days_z_penalty,
                nominal_capacity_discharge_fraction=self.nominal_capacity_discharge_fraction,
                pv_anchor_kwh_per_kwp=self.pv_anchor_kwh_per_kwp,
                pv_anchor_blend_weight=self.pv_anchor_blend_weight,
                consumption_anchor_kwh_per_annual_kwh=self.consumption_anchor_kwh_per_annual_kwh,
                min_dark_reference_days=self.min_dark_reference_days,
                min_sunny_match_days=self.min_sunny_match_days,
            )
        except Exception as exc:
            logger.warning("battery detection failed for customer %s: %s", customer_id, exc)
            return {
                "customer_id": customer_id,
                "has_battery": False,
                "prob_battery": np.nan,
                "battery_status": f"error: {exc}",
            }

        # battery_prob from analyze_battery_residential_v7 is always in percent (0–100).
        prob = float(result.get("battery_prob", 0.0)) / 100.0
        has_battery = prob >= self.classification_threshold

        return {
            "customer_id": customer_id,
            "has_battery": has_battery,
            "prob_battery": prob,
            "battery_status": result.get("status", "ok"),
            "n_matched_sunny_days": result.get("n_matched_sunny_days", 0),
            "n_dark_days": result.get("n_dark_days", 0),
        }
