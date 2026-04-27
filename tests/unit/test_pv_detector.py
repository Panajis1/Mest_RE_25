"""Unit tests for PVDetector and PVCapacityEstimator using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from re_nilm.detectors.pv import PVDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weather(n: int = 365 * 96) -> pd.DataFrame:
    """Synthetic one-year weather at 15-min resolution."""
    dt = pd.date_range("2022-01-01", periods=n, freq="15min")
    hour = dt.hour.to_numpy() + dt.minute.to_numpy() / 60
    day_of_year = dt.dayofyear.to_numpy()
    solar_angle = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    rad = np.clip(solar_angle * seasonal * 800, 0, None)
    temp = 10 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.default_rng(0).normal(0, 2, n)
    return pd.DataFrame({"dt_utc": dt, "t_2m_C": temp, "global_rad_W": rad})


def _make_customer_with_pv(customer_id: str, weather: pd.DataFrame, pv_fraction: float = 0.4) -> pd.DataFrame:
    """Customer whose PROD_KWH correlates with radiation (has PV)."""
    n = len(weather)
    rng = np.random.default_rng(42)
    conso = rng.uniform(0.2, 0.8, n)
    # Production proportional to radiation (scaled to reasonable kWh)
    prod = (weather["global_rad_W"].values / 1000.0 * pv_fraction + rng.normal(0, 0.02, n)).clip(0)
    return pd.DataFrame({
        "ID": customer_id,
        "DT_UTC": weather["dt_utc"].values,
        "CONSO_KWH": conso,
        "PROD_KWH": prod,
    })


def _make_customer_no_pv(customer_id: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Customer with zero production (no PV)."""
    n = len(weather)
    rng = np.random.default_rng(7)
    conso = rng.uniform(0.2, 0.8, n)
    return pd.DataFrame({
        "ID": customer_id,
        "DT_UTC": weather["dt_utc"].values,
        "CONSO_KWH": conso,
        "PROD_KWH": np.zeros(n),
    })


# ---------------------------------------------------------------------------
# PVDetector tests
# ---------------------------------------------------------------------------

def test_pv_detector_detects_pv_customer():
    weather = _make_weather()
    customer = _make_customer_with_pv("C001", weather, pv_fraction=0.5)
    detector = PVDetector(corr_threshold=0.3, min_yearly_prod_kwh=1.0)
    result = detector.predict_customer(customer, weather)
    assert result is not None
    assert result["customer_id"] == "C001"
    assert result["has_pv"] is True
    assert result["yearly_prod_kwh"] > 0


def test_pv_detector_no_pv_customer():
    weather = _make_weather()
    customer = _make_customer_no_pv("C002", weather)
    detector = PVDetector(min_yearly_prod_kwh=1.0)
    result = detector.predict_customer(customer, weather)
    assert result is not None
    assert result["has_pv"] is False
    assert result["yearly_prod_kwh"] == pytest.approx(0.0, abs=1e-6)


def test_pv_detector_empty_df_returns_none():
    weather = _make_weather(96)
    detector = PVDetector()
    result = detector.predict_customer(pd.DataFrame(), weather)
    assert result is None


def test_pv_detector_output_schema():
    weather = _make_weather()
    customer = _make_customer_with_pv("C003", weather)
    detector = PVDetector()
    result = detector.predict_customer(customer, weather)
    assert result is not None
    for key in ("customer_id", "has_pv", "prob_pv", "yearly_prod_kwh", "corr_prod_rad",
                "DeltaProd", "DeltaNet", "beta_regression"):
        assert key in result, f"Missing key: {key}"


def test_pv_detector_filter_detected():
    results = [
        {"customer_id": "A", "has_pv": True, "prob_pv": 0.8},
        {"customer_id": "B", "has_pv": False, "prob_pv": 0.1},
        {"customer_id": "C", "has_pv": True, "prob_pv": 0.6},
    ]
    detector = PVDetector()
    detected = detector.filter_detected(results)
    assert len(detected) == 2
    assert all(r["has_pv"] for r in detected)


# ---------------------------------------------------------------------------
# PVCapacityEstimator — smoke test (no legacy import needed)
# ---------------------------------------------------------------------------

def test_pv_capacity_skips_non_pv_customers():
    from re_nilm.estimators.pv_capacity import PVCapacityEstimator
    estimator = PVCapacityEstimator()
    result = estimator.estimate(
        pd.DataFrame(),
        {"customer_id": "X", "has_pv": False},
        pd.DataFrame(),
    )
    assert result is None
