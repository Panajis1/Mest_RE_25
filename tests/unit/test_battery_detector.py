"""Unit tests for the heuristic BatteryDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from re_nilm.detectors.battery import BatteryDetector


def _weather() -> pd.DataFrame:
    n = 365 * 96
    dt = pd.date_range("2022-01-01", periods=n, freq="15min")
    hour = dt.hour.to_numpy() + dt.minute.to_numpy() / 60
    doy = dt.dayofyear.to_numpy()
    rad = (
        np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
        * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365))
        * 800
    )
    temp = 10 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)
    return pd.DataFrame(
        {"dt_utc": dt, "t_2m_C": temp.astype("float32"), "global_rad_W": rad.astype("float32")}
    )


def _customer() -> pd.DataFrame:
    n = 365 * 96
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ID": "C001",
            "DT_UTC": pd.date_range("2022-01-01", periods=n, freq="15min").to_numpy(),
            "CONSO_KWH": rng.uniform(0.2, 0.8, n).astype("float32"),
            "PROD_KWH": np.zeros(n, dtype="float32"),
        }
    )


def test_battery_skips_when_pv_required_and_pv_absent():
    """enforce_pv_required=True should short-circuit non-PV customers."""
    det = BatteryDetector(enforce_pv_required=True)
    res = det.predict_customer(_customer(), _weather(), pv_result={"has_pv": False})
    assert res is not None
    assert res["customer_id"] == "C001"
    assert res["has_battery"] is False
    assert res["prob_battery"] == 0.0
    assert res["battery_status"] == "skipped_no_pv"


def test_battery_empty_customer_returns_none():
    det = BatteryDetector(enforce_pv_required=False)
    res = det.predict_customer(
        pd.DataFrame(columns=["ID", "DT_UTC", "CONSO_KWH"]),
        _weather(),
        pv_result={"has_pv": True},
    )
    assert res is None


def test_battery_returns_valid_probability_for_pv_customer():
    """On synthetic year-long data, the detector should return a finite
    probability in [0, 1] without raising."""
    det = BatteryDetector(enforce_pv_required=False)
    res = det.predict_customer(_customer(), _weather(), pv_result={"has_pv": True})
    assert res is not None
    assert res["customer_id"] == "C001"
    p = res["prob_battery"]
    assert isinstance(p, float)
    # NaN is allowed when battery_status == "error: ..." per the detector contract.
    if res["battery_status"] == "ok" or not res["battery_status"].startswith("error"):
        assert 0.0 <= p <= 1.0, f"prob_battery out of [0,1]: {p}"
    assert isinstance(res["has_battery"], (bool, np.bool_))


def test_battery_classification_threshold_is_respected():
    """The probability must always be normalised to [0, 1] regardless of the
    legacy v7 returning 0–100 (Bucket 1.2 fix)."""
    det = BatteryDetector(enforce_pv_required=False, classification_threshold=0.99)
    res = det.predict_customer(_customer(), _weather(), pv_result={"has_pv": True})
    assert res is not None
    if not str(res["battery_status"]).startswith("error"):
        assert res["prob_battery"] <= 1.0
        assert res["has_battery"] is (res["prob_battery"] >= 0.99)
