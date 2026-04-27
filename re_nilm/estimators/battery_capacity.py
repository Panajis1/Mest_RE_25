"""Battery capacity estimator.

Delegates to analyze_battery_residential_v7 in model/battery_detection.py which
returns the detection probability and shift-based capacity estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from re_nilm.estimators.base import AbstractEstimator

_BATTERY_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_BATTERY_DIR) not in sys.path:
    sys.path.insert(0, str(_BATTERY_DIR))

try:
    from battery_detection import analyze_battery_residential_v7 as _analyze_battery
    _BATTERY_AVAILABLE = True
except ImportError:
    _BATTERY_AVAILABLE = False


class BatteryCapacityEstimator(AbstractEstimator):
    """Shift-based battery capacity estimator using sigmoid-scored paired days.

    Args:
        capacity_min_kwh: Floor below which capacity is reported as NaN.
        capacity_max_kwh: Ceiling above which capacity is capped.
        temp_buffer_c: Temperature matching tolerance for pairing days.
    """

    def __init__(
        self,
        capacity_min_kwh: float = 5.0,
        capacity_max_kwh: float = 30.0,
        temp_buffer_c: float = 2.0,
    ):
        self.capacity_min_kwh = capacity_min_kwh
        self.capacity_max_kwh = capacity_max_kwh
        self.temp_buffer_c = temp_buffer_c

    def estimate(
        self,
        customer_ts: pd.DataFrame,
        detection_result: dict,
        weather: pd.DataFrame,
    ) -> dict | None:
        customer_id = detection_result.get("customer_id", "")

        if not detection_result.get("has_battery", False):
            return None

        if not _BATTERY_AVAILABLE:
            return {"customer_id": customer_id, "battery_capacity_kwh": np.nan, "error": "import_error"}

        pv_result = detection_result.get("pv_result", {})
        pv_row = {
            "pv_capacity_kwp": pv_result.get("pv_capacity_kwp", 0.0),
            "has_pv": pv_result.get("has_pv", False),
        }

        df = customer_ts.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
        df = df.dropna(subset=["DT_UTC"]).set_index("DT_UTC").sort_index()

        wdf = weather.copy()
        wdf["dt_utc"] = pd.to_datetime(wdf["dt_utc"], errors="coerce")
        wdf = wdf.dropna(subset=["dt_utc"]).set_index("dt_utc").sort_index()

        try:
            result = _analyze_battery(df, wdf, pv_row, temp_buffer=self.temp_buffer_c)
        except Exception as exc:
            return {"customer_id": customer_id, "battery_capacity_kwh": np.nan, "error": str(exc)}

        # Key names from analyze_battery_residential_v7
        cap = float(result.get("estimated_battery_capacity_kwh", np.nan))
        cap_lo = float(result.get("capacity_ci_lower_kwh", np.nan))
        cap_hi = float(result.get("capacity_ci_upper_kwh", np.nan))
        power_kw = float(result.get("estimated_battery_power_kw", np.nan))

        if not np.isnan(cap):
            if cap < self.capacity_min_kwh or cap > self.capacity_max_kwh:
                cap = np.nan

        return {
            "customer_id": customer_id,
            "battery_capacity_kwh": cap,
            "batt_ci_lower": cap_lo,
            "batt_ci_upper": cap_hi,
            "battery_power_kw": power_kw,
        }
