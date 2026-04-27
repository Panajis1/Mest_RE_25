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

# Label names matching HP model training labels
_INT_TO_LABEL = {0: "no_hp", 1: "winter_hp", 2: "summer_hp"}


class HeatPumpDetector(AbstractDetector):
    """HP presence detector using a pre-trained Random Forest 3-class classifier.

    Outputs one of: no_hp, winter_hp, summer_hp.

    Args:
        model: Loaded sklearn pipeline (SimpleImputer + RandomForestClassifier).
        prob_threshold: Minimum class probability for the HP class to call has_hp=True.
        feature_cols: Feature column names in training order.
        night_rad_threshold: Max W/m² for a timestep to count as 'night'.
        min_night_rows: Minimum night rows required to extract features.
    """

    def __init__(
        self,
        model,
        prob_threshold: float = 0.5,
        feature_cols: Optional[List[str]] = None,
        night_rad_threshold: float = 20.0,
        min_night_rows: int = 100,
    ):
        self.model = model
        self.prob_threshold = prob_threshold
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
            proba = self.model.predict_proba(X)[0]  # shape (3,) for no_hp / winter_hp / summer_hp
            pred_class = int(np.argmax(proba))
        except Exception:
            return None

        hp_type = _INT_TO_LABEL.get(pred_class, "no_hp")
        has_hp = hp_type != "no_hp" and float(proba[pred_class]) >= self.prob_threshold
        prob_hp = float(proba[pred_class]) if has_hp else float(max(proba[1], proba[2]))

        return {
            "customer_id": customer_id,
            "has_hp": has_hp,
            "prob_hp": prob_hp,
            "hp_type": hp_type,
        }
