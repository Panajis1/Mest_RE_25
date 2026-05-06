"""Unit tests for ACDetector and HeatPumpDetector using stub model objects.

Real detector models are RandomForest pipelines pinned to sklearn 1.3.x and
saved as joblib artifacts. To exercise the detector code paths in CI without
shipping a real artifact, these tests inject a tiny stub object that satisfies
the exact API the detector calls (``predict_proba``, ``predict``,
``classes_``). The stub returns deterministic outputs so we can assert the
detector wraps them correctly into the result dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from re_nilm.detectors.ac import ACDetector
from re_nilm.detectors.heat_pump import HeatPumpDetector


# ---------------------------------------------------------------------------
# Synthetic data helpers (one customer, one year of 15-min readings)
# ---------------------------------------------------------------------------

_N = 365 * 96


def _weather() -> pd.DataFrame:
    dt = pd.date_range("2022-01-01", periods=_N, freq="15min")
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
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ID": "C001",
            "DT_UTC": pd.date_range("2022-01-01", periods=_N, freq="15min").to_numpy(),
            "CONSO_KWH": rng.uniform(0.2, 0.8, _N).astype("float32"),
            "PROD_KWH": np.zeros(_N, dtype="float32"),
        }
    )


# ---------------------------------------------------------------------------
# Stub models — match the minimum API the detectors exercise
# ---------------------------------------------------------------------------


class _StubBinaryProba:
    """Mimics sklearn BinaryClassifier.predict_proba() returning a fixed prob."""

    def __init__(self, prob_positive: float):
        self._p = prob_positive

    def predict_proba(self, X):
        return np.array([[1 - self._p, self._p]] * len(X))


class _StubClassifier:
    """The inner classifier step of the HP pipeline — exposes classes_."""

    def __init__(self):
        self.classes_ = np.array([0, 1, 2])


class _StubMulticlass:
    """Mimics the full sklearn Pipeline that HeatPumpDetector unwraps.

    HeatPumpDetector calls ``model.predict``, ``model.predict_proba``, and
    ``model.named_steps['clf'].classes_`` — the last via _classifier_step.
    `_INT_TO_LABEL` maps 0→no_hp, 1→winter_hp, 2→summer_hp.
    """

    def __init__(self, predicted_class: int, probs=(0.1, 0.7, 0.2)):
        self._cls = predicted_class
        self._probs = np.asarray(probs, dtype=float)
        self.named_steps = {"clf": _StubClassifier()}

    def predict(self, X):
        return np.array([self._cls] * len(X))

    def predict_proba(self, X):
        return np.tile(self._probs, (len(X), 1))


# ---------------------------------------------------------------------------
# ACDetector
# ---------------------------------------------------------------------------


def test_ac_detector_returns_positive_when_stub_high_prob():
    det = ACDetector(model=_StubBinaryProba(0.9), prob_threshold=0.55)
    res = det.predict_customer(_customer(), _weather())
    assert res is not None
    assert res["customer_id"] == "C001"
    assert res["has_ac"] is True
    assert res["prob_ac"] >= 0.55


def test_ac_detector_returns_negative_when_stub_low_prob():
    det = ACDetector(model=_StubBinaryProba(0.1), prob_threshold=0.55)
    res = det.predict_customer(_customer(), _weather())
    assert res is not None
    assert res["has_ac"] is False
    assert res["prob_ac"] < 0.55


def test_ac_detector_empty_customer_returns_none():
    det = ACDetector(model=_StubBinaryProba(0.9), prob_threshold=0.55)
    assert det.predict_customer(pd.DataFrame(columns=["ID", "DT_UTC", "CONSO_KWH"]), _weather()) is None


def test_ac_detector_predict_failure_returns_error_row():
    """Per Bucket 3.3 standardisation: predict failures must surface as an
    error row tagged with customer_id, never as silent None."""

    class _Boom:
        def predict_proba(self, X):
            raise RuntimeError("simulated model failure")

    det = ACDetector(model=_Boom(), prob_threshold=0.55)
    res = det.predict_customer(_customer(), _weather())
    assert res is not None
    assert res["customer_id"] == "C001"
    assert res["has_ac"] is False
    assert "error" in res and "ac_predict_failed" in res["error"]


# ---------------------------------------------------------------------------
# HeatPumpDetector
# ---------------------------------------------------------------------------


def test_hp_detector_returns_winter_hp_for_winter_class():
    det = HeatPumpDetector(model=_StubMulticlass(predicted_class=1, probs=(0.1, 0.8, 0.1)))
    res = det.predict_customer(_customer(), _weather())
    assert res is not None
    assert res["customer_id"] == "C001"
    assert res["has_hp"] is True
    assert res["hp_type"] == "winter_hp"


def test_hp_detector_returns_no_hp_for_class_zero():
    det = HeatPumpDetector(model=_StubMulticlass(predicted_class=0, probs=(0.8, 0.1, 0.1)))
    res = det.predict_customer(_customer(), _weather())
    assert res is not None
    assert res["has_hp"] is False
    assert res["hp_type"] == "no_hp"


def test_hp_detector_predict_failure_returns_error_row():
    """Was the regression: heat_pump.py used to silently return None on predict
    failure (Bucket 3.3). Verify it now produces an audit row."""

    class _Boom:
        named_steps = {"clf": _StubClassifier()}

        def predict(self, X):
            raise RuntimeError("simulated model failure")

        def predict_proba(self, X):
            return np.array([[0.5, 0.3, 0.2]] * len(X))

    det = HeatPumpDetector(model=_Boom())
    res = det.predict_customer(_customer(), _weather())
    assert res is not None
    assert res["customer_id"] == "C001"
    assert res["has_hp"] is False
    assert res["hp_type"] == "no_hp"
    assert "error" in res and "hp_predict_failed" in res["error"]
