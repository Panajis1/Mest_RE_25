"""HP disaggregation estimator — wraps ScientificTwoStageHPModel from hp_model/disaggregation_functions.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import joblib

from re_nilm.estimators.base import AbstractEstimator

_HP_MODEL_DIR = Path(__file__).resolve().parents[2] / "hp_model"
if str(_HP_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_HP_MODEL_DIR))

try:
    from disaggregation_functions import _compute_user_profile, _add_calendar_and_dynamic_features
    _HP_DISAGG_HELPERS_AVAILABLE = True
except ImportError:
    _HP_DISAGG_HELPERS_AVAILABLE = False


class HPDisaggregationEstimator(AbstractEstimator):
    """HP load disaggregation using the pre-trained ScientificTwoStageHPModel.

    Only applied to customers where hp_type='winter_hp'.

    Builds the exact feature set the model expects (per disaggregation_functions.py),
    including scientific_like_score and tot_kw, then calls model.predict().

    Args:
        model: Loaded ScientificTwoStageHPModel instance.
        feature_cols: Feature columns matching training order (from model.feature_cols).
    """

    def __init__(self, model, feature_cols: Optional[List[str]] = None):
        self.model = model
        self.feature_cols = feature_cols

    @classmethod
    def load(cls, path: Path, **kwargs) -> "HPDisaggregationEstimator":
        """Load serialized ScientificTwoStageHPModel."""
        model = joblib.load(path)
        feature_cols = list(model.feature_cols) if hasattr(model, "feature_cols") and model.feature_cols is not None else None
        return cls(model=model, feature_cols=feature_cols, **kwargs)

    def _build_features(self, customer_ts: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """Build the feature DataFrame that ScientificTwoStageHPModel.predict() expects.

        Mirrors build_aligned_dataset / _compute_user_profile /
        _add_calendar_and_dynamic_features from disaggregation_functions.py.
        """
        if not _HP_DISAGG_HELPERS_AVAILABLE:
            return pd.DataFrame()

        df = customer_ts.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
        df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")

        # Normalize to datetime64[us] — pandas 2.x merge_asof requires identical units
        df["DT_UTC"] = df["DT_UTC"].astype("datetime64[us]")
        wdf = weather.copy()
        wdf["dt_utc"] = wdf["dt_utc"].astype("datetime64[us]")
        wdf = wdf.sort_values("dt_utc")
        merged = pd.merge_asof(
            df.rename(columns={"DT_UTC": "dt_utc"}),
            wdf,
            on="dt_utc",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )

        # Map to column names the HP model expects
        merged["tot_kw"] = pd.to_numeric(merged["CONSO_KWH"], errors="coerce") * 4.0
        merged["temp"] = pd.to_numeric(merged.get("t_2m_C", pd.Series(dtype=float)), errors="coerce")
        merged["glob_rad"] = pd.to_numeric(merged.get("global_rad_W", pd.Series(dtype=float)), errors="coerce")
        merged = merged.dropna(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)

        if merged.empty:
            return pd.DataFrame()

        # Compute per-customer profile (includes scientific_like_score)
        profile = _compute_user_profile(merged)
        for k, v in profile.items():
            if k != "profile_class":
                merged[k] = v

        merged = _add_calendar_and_dynamic_features(merged)
        return merged

    def estimate(
        self,
        customer_ts: pd.DataFrame,
        detection_result: dict,
        weather: pd.DataFrame,
    ) -> dict | None:
        customer_id = detection_result.get("customer_id", "")

        if not detection_result.get("has_hp", False) or detection_result.get("hp_type") != "winter_hp":
            return None

        if not _HP_DISAGG_HELPERS_AVAILABLE:
            return {"customer_id": customer_id, "error": "disaggregation_functions helpers not importable"}

        feat_df = self._build_features(customer_ts, weather)
        if feat_df.empty:
            return None

        try:
            pred_df = self.model.predict(feat_df)
        except Exception as exc:
            return {"customer_id": customer_id, "error": str(exc)}

        hp_kw_pred = pred_df["hp_kw_pred"].to_numpy()

        out_df = pd.DataFrame({
            "dt_utc": feat_df["dt_utc"].values,
            "customer_id": customer_id,
            "hp_kw_pred": pred_df["hp_kw_pred"].values.clip(min=0),
            "hp_on_prob": pred_df["hp_on_prob"].values,
        })

        return {
            "customer_id": customer_id,
            "hp_disagg_15min": out_df,
            "hp_mean_kw": float(np.nanmean(hp_kw_pred)),
            "hp_annual_kwh": float(np.nansum(hp_kw_pred) * 0.25),
        }
