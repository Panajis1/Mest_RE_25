"""Unit tests for EVDetector and EVSessionEstimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from re_nilm.detectors.ev import EVDetector
from re_nilm.estimators.ev_sessions import EVSessionEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_customer(n_days: int = 90, cid: str = "C001") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (customer_df, weather_df) with flat 0.2 kW consumption."""
    dt = pd.date_range("2024-01-01", periods=n_days * 96, freq="15min")
    df = pd.DataFrame(
        {
            "ID": cid,
            "DT_UTC": dt,
            "CONSO_KWH": 0.2 / 4,  # 0.2 kW × 15min = 0.05 kWh
            "PROD_KWH": 0.0,
        }
    )
    weather = pd.DataFrame(
        {
            "dt_utc": dt,
            "t_2m_C": 10.0,
            "global_rad_W": 0.0,
        }
    )
    return df, weather


def _add_ev_sessions(df: pd.DataFrame, power_kw: float = 3.5, hour_start: int = 21,
                     n_hours: float = 2.0, every_n_days: int = 2) -> pd.DataFrame:
    """Add regular overnight EV charging blocks to customer_df."""
    df = df.copy()
    dt = pd.to_datetime(df["DT_UTC"])
    steps = int(n_hours * 4)  # 15-min steps
    for i, ts in enumerate(dt):
        if ts.hour == hour_start and ts.minute == 0:
            day_idx = (ts - dt.iloc[0]).days
            if day_idx % every_n_days == 0:
                mask = (dt >= ts) & (dt < ts + pd.Timedelta(hours=n_hours))
                df.loc[mask, "CONSO_KWH"] = power_kw * 0.25  # kWh per 15min
    return df


# ---------------------------------------------------------------------------
# EVDetector tests
# ---------------------------------------------------------------------------

class TestEVDetector:
    def test_ev_detected_on_charging_profile(self):
        """Customer with regular 3.5 kW nightly charging → has_ev=True."""
        df, weather = _make_customer(n_days=90)
        df = _add_ev_sessions(df, power_kw=3.5, hour_start=21, n_hours=2.0, every_n_days=2)
        det = EVDetector(prob_threshold=0.3)
        result = det.predict_customer(df, weather)
        assert result is not None
        assert result["has_ev"] is True
        assert result["prob_ev"] > 0.3
        assert result["ev_grid_sessions"] > 0

    def test_ev_not_detected_flat_profile(self):
        """Flat 0.2 kW consumption → has_ev=False."""
        df, weather = _make_customer(n_days=90)
        det = EVDetector(prob_threshold=0.3)
        result = det.predict_customer(df, weather)
        assert result is not None
        assert result["has_ev"] is False
        assert result["prob_ev"] == 0.0
        assert result["ev_total_sessions"] == 0

    def test_ev_filter_high_consumption(self):
        """Commercial-scale customer (>100 MWh/year) → filtered out."""
        df, weather = _make_customer(n_days=365)
        # 30 kW constant = 30 × 8760 = 262 MWh/year >> 100 MWh
        df["CONSO_KWH"] = 30.0 * 0.25
        det = EVDetector(max_yearly_mwh=100.0)
        result = det.predict_customer(df, weather)
        assert result is not None
        assert result["has_ev"] is False
        assert result["ev_filtered_out"] is True

    def test_ev_output_schema(self):
        """Result dict must contain all required keys."""
        df, weather = _make_customer(n_days=60)
        df = _add_ev_sessions(df, power_kw=3.5)
        det = EVDetector()
        result = det.predict_customer(df, weather)
        assert result is not None
        required = {
            "customer_id", "has_ev", "prob_ev",
            "ev_grid_sessions", "ev_pv_sessions", "ev_total_sessions",
            "ev_yearly_mwh", "ev_filtered_out",
        }
        assert required.issubset(result.keys()), f"Missing: {required - result.keys()}"
        assert isinstance(result["has_ev"], bool)
        assert 0.0 <= result["prob_ev"] <= 1.0

    def test_ev_empty_df_returns_none(self):
        det = EVDetector()
        result = det.predict_customer(pd.DataFrame(), pd.DataFrame())
        assert result is None

    def test_ev_no_prod_column(self):
        """Customers without PROD_KWH column should still work."""
        df, weather = _make_customer(n_days=60)
        df = _add_ev_sessions(df, power_kw=3.5)
        df = df.drop(columns=["PROD_KWH"])
        det = EVDetector(prob_threshold=0.3)
        result = det.predict_customer(df, weather)
        assert result is not None
        assert "prob_ev" in result

    def test_ev_prob_increases_with_more_sessions(self):
        """More frequent charging → higher probability."""
        df_sparse, weather = _make_customer(n_days=90)
        df_sparse = _add_ev_sessions(df_sparse, power_kw=3.5, every_n_days=7)

        df_dense, _ = _make_customer(n_days=90)
        df_dense = _add_ev_sessions(df_dense, power_kw=3.5, every_n_days=2)

        det = EVDetector()
        r_sparse = det.predict_customer(df_sparse, weather)
        r_dense = det.predict_customer(df_dense, weather)
        assert r_dense["prob_ev"] >= r_sparse["prob_ev"]


# ---------------------------------------------------------------------------
# EVSessionEstimator tests
# ---------------------------------------------------------------------------

class TestEVSessionEstimator:
    def test_sessions_returned_for_ev_customer(self):
        """EV-positive customer returns non-empty sessions dict."""
        df, weather = _make_customer(n_days=60)
        df = _add_ev_sessions(df, power_kw=3.0, hour_start=20, n_hours=2.5, every_n_days=2)
        det_result = {"customer_id": "C001", "has_ev": True}
        est = EVSessionEstimator(step_threshold_kwh=0.5, min_duration_minutes=30)
        result = est.estimate(df, det_result, weather)
        assert result is not None
        assert "ev_sessions" in result
        sessions = result["ev_sessions"]
        assert isinstance(sessions, pd.DataFrame)
        assert not sessions.empty
        assert "customer_id" in sessions.columns
        assert "duration_h" in sessions.columns
        assert "energy_kWh" in sessions.columns

    def test_sessions_none_for_non_ev_customer(self):
        """Non-EV customer returns None."""
        df, weather = _make_customer()
        det_result = {"customer_id": "C001", "has_ev": False}
        est = EVSessionEstimator()
        result = est.estimate(df, det_result, weather)
        assert result is None

    def test_sessions_none_for_empty_df(self):
        det_result = {"customer_id": "C001", "has_ev": True}
        est = EVSessionEstimator()
        result = est.estimate(pd.DataFrame(), det_result, pd.DataFrame())
        assert result is None
