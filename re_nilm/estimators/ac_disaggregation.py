"""AC disaggregation estimator — wraps AcTwoStageModel from model/ac_disaggregation.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import joblib

from re_nilm.estimators.base import AbstractEstimator

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

# Import legacy helpers lazily to avoid top-level import errors
try:
    from ac_disaggregation import _compute_user_profile, _add_calendar_and_dynamic_features
    _AC_DISAGG_HELPERS_AVAILABLE = True
except ImportError:
    _AC_DISAGG_HELPERS_AVAILABLE = False


class ACDisaggregationEstimator(AbstractEstimator):
    """AC load disaggregation using the pre-trained AcTwoStageModel.

    Builds the exact feature set the trained model expects (per ac_disaggregation.py),
    then calls AcTwoStageModel.predict() which handles imputation, month-gating,
    and ac_kw prediction internally.

    Args:
        model: Loaded AcTwoStageModel instance.
        feature_cols: Feature column names from training (loaded from JSON sidecar).
    """

    def __init__(self, model, feature_cols: Optional[List[str]] = None):
        self.model = model
        self.feature_cols = feature_cols

    @classmethod
    def load(cls, path: Path, features_path: Optional[Path] = None, **kwargs) -> "ACDisaggregationEstimator":
        """Load serialized AcTwoStageModel and optional feature_cols JSON sidecar."""
        model = joblib.load(path)
        feature_cols = None
        if features_path is not None and Path(features_path).exists():
            with open(features_path) as f:
                feature_cols = json.load(f)
        elif hasattr(model, "feature_cols") and model.feature_cols is not None:
            feature_cols = list(model.feature_cols)
        return cls(model=model, feature_cols=feature_cols, **kwargs)

    def _build_features(self, customer_ts: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """Build the feature DataFrame that AcTwoStageModel.predict() expects.

        Mirrors build_aligned_dataset / _compute_user_profile /
        _add_calendar_and_dynamic_features from ac_disaggregation.py.
        """
        if not _AC_DISAGG_HELPERS_AVAILABLE:
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

        # Map to the column names the model expects
        merged["tot_kw"] = pd.to_numeric(merged["CONSO_KWH"], errors="coerce") * 4.0
        merged["temp"] = pd.to_numeric(merged.get("t_2m_C", pd.Series(dtype=float)), errors="coerce")
        merged["glob_rad"] = pd.to_numeric(merged.get("global_rad_W", pd.Series(dtype=float)), errors="coerce")
        merged = merged.dropna(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)

        if merged.empty:
            return pd.DataFrame()

        # Add per-customer profile statistics (scalar columns, broadcast to all rows)
        profile = _compute_user_profile(merged)
        for k, v in profile.items():
            if k != "profile_class":
                merged[k] = v

        # Add calendar + lag + rolling features
        merged = _add_calendar_and_dynamic_features(merged)
        return merged

    def estimate(
        self,
        customer_ts: pd.DataFrame,
        detection_result: dict,
        weather: pd.DataFrame,
    ) -> dict | None:
        customer_id = detection_result.get("customer_id", "")

        if not detection_result.get("has_ac", False):
            return None

        if not _AC_DISAGG_HELPERS_AVAILABLE:
            return {"customer_id": customer_id, "error": "ac_disaggregation helpers not importable"}

        feat_df = self._build_features(customer_ts, weather)
        if feat_df.empty:
            return None

        try:
            # model.predict() handles imputation, month gating, clipping, and ac_kw output
            pred_df = self.model.predict(feat_df, apply_month_gating=True)
        except Exception as exc:
            return {"customer_id": customer_id, "error": str(exc)}

        ac_kw_pred = pred_df["ac_kw_pred"].to_numpy()

        out_df = pd.DataFrame({
            "dt_utc": feat_df["dt_utc"].values,
            "customer_id": customer_id,
            "ac_kw_pred": pred_df["ac_kw_pred"].values.clip(min=0),
            "ac_on_prob": pred_df["ac_on_prob"].values,
        })

        return {
            "customer_id": customer_id,
            "ac_disagg_15min": out_df,
            "ac_mean_kw": float(np.nanmean(ac_kw_pred)),
            "ac_annual_kwh": float(np.nansum(ac_kw_pred) * 0.25),
        }
