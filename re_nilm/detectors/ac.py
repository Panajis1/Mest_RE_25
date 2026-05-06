"""AC (air conditioning) detector — loads pre-trained Random Forest, extracts daytime features."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from re_nilm.detectors.base import AbstractDetector
from re_nilm.features.load import extract_ac_features

# Feature columns matching ac_actrainingfunctions.py::FEATURE_COLS (training order)
_DEFAULT_FEATURE_COLS: List[str] = [
    "corr_temp_all",
    "corr_temp_hot",
    "corr_rad_all",
    "season_balance",
    "thermal_balance",
    "summer_share",
    "daytime_share",
    "coeff_var",
    "acf_1h",
    "acf_24h",
    "afternoon_peak_ratio",
    "peak_summer_hour",
    "summer_vs_spring",
    "hot_load_ratio",
]

_AC_ON_LABEL = "has_ac"


class ACDetector(AbstractDetector):
    """AC presence detector using a pre-trained Random Forest classifier.

    The model was trained on Dataport AC sub-meter data by ACDetectorTrainer
    (re_nilm/training/trainers/ac_detector_trainer.py) and serialized with joblib.

    Args:
        model: Loaded sklearn pipeline (SimpleImputer + RandomForestClassifier).
        prob_threshold: Minimum predicted probability to call has_ac=True.
        feature_cols: Feature columns to pass to the model (must match training order).
        day_rad_threshold: W/m² above which a row is considered daytime.
        min_day_rows: Minimum daytime rows required to compute features.
    """

    def __init__(
        self,
        model,
        prob_threshold: float = 0.55,
        feature_cols: Optional[List[str]] = None,
        day_rad_threshold: float = 50.0,
        min_day_rows: int = 100,
    ):
        self.model = model
        self.prob_threshold = prob_threshold
        self.feature_cols = feature_cols or _DEFAULT_FEATURE_COLS
        self.day_rad_threshold = day_rad_threshold
        self.min_day_rows = min_day_rows

    @classmethod
    def load(cls, path: Path, **kwargs) -> "ACDetector":
        """Load a serialized RF pipeline from path."""
        model = joblib.load(path)
        # Disable RF internal parallelism — the pipeline already uses multi-process
        # workers, so per-worker RF parallelism is redundant and triggers a sklearn
        # 1.8.0 warning about joblib.Parallel vs sklearn.utils.parallel.Parallel.
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            model.named_steps["clf"].n_jobs = 1
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

        feats = extract_ac_features(
            load, temp, rad, ts,
            day_rad_threshold=self.day_rad_threshold,
            min_day_rows=self.min_day_rows,
        )
        if feats is None:
            return None

        X = np.array([[feats.get(c, np.nan) for c in self.feature_cols]])
        try:
            prob = float(self.model.predict_proba(X)[0, 1])
        except Exception:
            prob = np.nan

        has_ac = not np.isnan(prob) and prob >= self.prob_threshold

        return {
            "customer_id": customer_id,
            "has_ac": has_ac,
            "prob_ac": prob if not np.isnan(prob) else 0.0,
        }
