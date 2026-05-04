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
    injection_bonus, consistency, and shift_ratio. Parameters reflect migrated
    battery v7 defaults.

    Args:
        classification_threshold: Minimum battery_prob to call has_battery=True.
        enforce_pv_required: If True, skip customers with has_pv=False.
        dark_day_rad_max_w: Max peak W/m² for a day to count as 'dark'.
        sunny_day_rad_min_w: Min peak W/m² for a day to count as 'sunny'.
        temp_buffer_c: Temperature matching tolerance when pairing dark/sunny days.
        sigmoid_intercept: Sigmoid bias term (more negative = stricter). Tuned to -2.5.
        strict_min_matched_sunny_days: If fewer matched sunny days are found, apply a
            z-score penalty of `low_matched_days_z_penalty`. Tuned to 5.
        low_matched_days_z_penalty: Penalty subtracted from z when matched days are
            below `strict_min_matched_sunny_days`. Tuned to 0.5.
        nominal_capacity_discharge_fraction: Assumed fraction of battery discharged per
            evening event, used to scale shift energy → capacity estimate. Tuned to 0.35.
        pv_anchor_kwh_per_kwp: kWh/kWp used for the PV-size capacity anchor (1:1 ratio). Tuned to 1.0.
        pv_anchor_blend_weight: Blend weight for the PV anchor in capacity estimation.
            Tuned to 0.45.
    """

    def __init__(
        self,
        classification_threshold: float = 0.5,
        enforce_pv_required: bool = True,
        dark_day_rad_max_w: float = 100.0,
        sunny_day_rad_min_w: float = 100.0,
        temp_buffer_c: float = 2.0,
        sigmoid_intercept: float = -2.5,
        strict_min_matched_sunny_days: int = 5,
        low_matched_days_z_penalty: float = 0.5,
        nominal_capacity_discharge_fraction: float = 0.35,
        pv_anchor_kwh_per_kwp: float = 1.0,
        pv_anchor_blend_weight: float = 0.45,
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

        # Build a pv_row dict compatible with the legacy function signature.
        # has_pv_prob is the key _battery_v7 uses for the inner PV gate (requires >= 0.5).
        # PVDetector returns this value under the key "prob_pv".
        pv_row = {
            "pv_capacity_kwp": pv_result.get("pv_capacity_kwp", 0.0),
            "has_pv": pv_result.get("has_pv", False),
            "has_pv_prob": pv_result.get("prob_pv", np.nan),
        }

        weather = weather_df.copy()
        weather["dt_utc"] = pd.to_datetime(weather["dt_utc"], errors="coerce")
        weather = weather.dropna(subset=["dt_utc"]).set_index("dt_utc").sort_index()

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
            )
        except Exception as exc:
            logger.warning("battery detection failed for customer %s: %s", customer_id, exc)
            return {
                "customer_id": customer_id,
                "has_battery": False,
                "prob_battery": np.nan,
                "battery_status": f"error: {exc}",
            }

        # battery_prob from analyze_battery_residential_v7 is in percent (0–100); normalise to 0–1
        prob_raw = float(result.get("battery_prob", 0.0))
        prob = prob_raw / 100.0 if prob_raw > 1.0 else prob_raw
        has_battery = prob >= self.classification_threshold

        return {
            "customer_id": customer_id,
            "has_battery": has_battery,
            "prob_battery": prob,
            "battery_status": result.get("status", "ok"),
        }
