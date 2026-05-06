"""Heat pump detector — loads pre-trained RF 3-class model, uses night-only features."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from re_nilm.detectors.base import AbstractDetector
from re_nilm.features.load import extract_hp_features

# Feature columns in training order (from hp_detection_functions.py:FEATURE_COLS)
_DEFAULT_FEATURE_COLS: List[str] = [
    "corr_temp_all",
    "corr_temp_cold",
    "corr_temp_hot",
    "season_balance",
    "thermal_balance",
    "coeff_var",
    "acf_1h",
    "acf_24h",
]

# Label names matching HP model training labels (hp_detection_functions.INT_TO_LABEL)
_INT_TO_LABEL = {0: "no_hp", 1: "winter_hp", 2: "summer_hp"}


def _classifier_step(pipeline):
    """Return the RF step — trainer uses 'clf', some legacy scripts use 'rf'."""
    ns = pipeline.named_steps
    if "clf" in ns:
        return ns["clf"]
    if "rf" in ns:
        return ns["rf"]
    raise AttributeError(f"Pipeline has no 'clf' or 'rf' step: {list(ns.keys())}")


def _prob_hp_winter_plus_summer(model, X: np.ndarray) -> float:
    """Mirror the legacy hp_detection_functions HP-class branch."""
    proba = model.predict_proba(X)
    classes = _classifier_step(model).classes_
    hp_probs = {}
    for cls_id, cls_name in _INT_TO_LABEL.items():
        if cls_id in classes:
            idx = int(np.where(classes == cls_id)[0][0])
            hp_probs[cls_name] = float(proba[0, idx])
        else:
            hp_probs[cls_name] = 0.0
    return hp_probs.get("winter_hp", 0.0) + hp_probs.get("summer_hp", 0.0)


class HeatPumpDetector(AbstractDetector):
    """HP presence detector using a pre-trained Random Forest 3-class classifier.

    Decision logic matches legacy ``hp_detection_functions`` / validation script:

    - ``has_hp`` is True iff ``model.predict(X)[0]`` maps to ``winter_hp`` or ``summer_hp``
      (no probability threshold gate).
    - ``prob_hp`` is ``P(winter_hp) + P(summer_hp)`` from ``predict_proba``.

    Outputs one of: no_hp, winter_hp, summer_hp.

    Args:
        model: Loaded sklearn pipeline (SimpleImputer + RandomForestClassifier).
        feature_cols: Feature column names in training order.
        night_rad_threshold: Max W/m² for a timestep to count as 'night'.
        min_night_rows: Minimum night rows required to extract features.
    """

    def __init__(
        self,
        model,
        feature_cols: Optional[List[str]] = None,
        night_rad_threshold: float = 20.0,
        min_night_rows: int = 100,
    ):
        self.model = model
        self.feature_cols = feature_cols or _DEFAULT_FEATURE_COLS
        self.night_rad_threshold = night_rad_threshold
        self.min_night_rows = min_night_rows

    @classmethod
    def load(cls, path: Path, **kwargs) -> "HeatPumpDetector":
        """Load a serialized RF pipeline from path."""
        model = joblib.load(path)
        return cls(model=model, **kwargs)

    def predict_customer(
        self,
        customer_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        **context,
    ) -> dict | None:
        if customer_df.empty:
            return None

        customer_id = str(customer_df["ID"].iloc[0])
        df = customer_df.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
        df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")
        df = df.drop_duplicates(subset=["DT_UTC"], keep="first")

        # Normalize to datetime64[us] — pandas 2.x merge_asof requires identical units
        df["DT_UTC"] = df["DT_UTC"].astype("datetime64[us]")
        weather = weather_df.copy()
        weather["dt_utc"] = weather["dt_utc"].astype("datetime64[us]")
        weather = weather.sort_values("dt_utc")
        merged = pd.merge_asof(
            df.rename(columns={"DT_UTC": "_ts"}),
            weather.rename(columns={"dt_utc": "_ts"}),
            on="_ts",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )

        load = pd.to_numeric(merged["CONSO_KWH"], errors="coerce") * 4.0  # kWh → kW
        temp = pd.to_numeric(merged.get("t_2m_C", pd.Series(dtype=float)), errors="coerce")
        rad = pd.to_numeric(merged.get("global_rad_W", pd.Series(dtype=float)), errors="coerce")
        ts = merged["_ts"]

        feats = extract_hp_features(
            load, temp, rad, ts,
            night_rad_threshold=self.night_rad_threshold,
            min_night_rows=self.min_night_rows,
        )
        if feats is None:
            return None

        X = np.array([[feats.get(c, np.nan) for c in self.feature_cols]])
        try:
            pred = int(self.model.predict(X)[0])
            prob_hp = _prob_hp_winter_plus_summer(self.model, X)
        except Exception as exc:
            return {
                "customer_id": customer_id,
                "has_hp": False,
                "prob_hp": 0.0,
                "hp_type": "no_hp",
                "error": f"hp_predict_failed: {exc}",
            }

        hp_type = _INT_TO_LABEL.get(pred, "no_hp")
        has_hp = hp_type != "no_hp"

        return {
            "customer_id": customer_id,
            "has_hp": has_hp,
            "prob_hp": prob_hp,
            "hp_type": hp_type,
        }
