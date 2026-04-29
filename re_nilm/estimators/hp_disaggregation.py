"""HP disaggregation estimator — wraps ScientificTwoStageHPModel from hp_model/disaggregation_functions.py."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import joblib

from re_nilm.estimators.base import AbstractEstimator
from re_nilm.estimators._hp_disagg_v1 import (
    _add_calendar_and_dynamic_features,
    _compute_user_profile,
)


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

    def _build_features(
        self,
        customer_ts: pd.DataFrame,
        weather: pd.DataFrame,
        pv_capacity_kwp: float = 0.0,
    ) -> pd.DataFrame:
        """Build the feature DataFrame that ScientificTwoStageHPModel.predict() expects.

        Mirrors build_aligned_dataset / _compute_user_profile /
        _add_calendar_and_dynamic_features from disaggregation_functions.py.

        For PV customers (pv_capacity_kwp > 0) the load signal is reconstructed
        as total building load rather than raw grid import:

            tot_kw = (CONSO_KWH - PROD_KWH + pv_forecast_kwh) * 4

        pv_forecast_kwh = (global_rad_W / 1000) * pv_capacity_kwp * 0.25

        pv_capacity_kwp was estimated by regressing PROD_KWH against irradiance
        at STC (_STC_FACTOR = 4000), so system efficiency is already encoded in
        the capacity value — no additional efficiency multiplier is applied here.

        For non-PV customers (pv_capacity_kwp == 0) the formula reduces to
        CONSO_KWH * 4, which is identical to the previous behaviour.
        """
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

        # Reconstruct total building load.
        # For PV customers, raw CONSO_KWH understates the load during self-consumption
        # hours because PV generation reduces apparent grid import.  Adding back the
        # PV forecast and subtracting the grid export (PROD_KWH) recovers the true
        # total building demand that the model was trained on (Dataport sub-meters).
        conso = pd.to_numeric(merged["CONSO_KWH"], errors="coerce").fillna(0.0)
        if pv_capacity_kwp > 0.0:
            prod = pd.to_numeric(merged.get("PROD_KWH", pd.Series(0.0, index=merged.index)),
                                 errors="coerce").fillna(0.0)
            rad = pd.to_numeric(merged.get("global_rad_W", pd.Series(0.0, index=merged.index)),
                                errors="coerce").fillna(0.0)
            # pv_forecast_kwh: (W/m² / 1000) × kWp × 0.25 h — capacity already encodes efficiency
            pv_kwh = (rad.clip(lower=0) / 1000.0) * pv_capacity_kwp * 0.25
            merged["tot_kw"] = (conso - prod + pv_kwh).clip(lower=0) * 4.0
        else:
            merged["tot_kw"] = conso * 4.0

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
        pv_capacity_kwp: float = 0.0,
    ) -> dict | None:
        customer_id = detection_result.get("customer_id", "")

        if not detection_result.get("has_hp", False) or detection_result.get("hp_type") != "winter_hp":
            return None

        feat_df = self._build_features(customer_ts, weather, pv_capacity_kwp=pv_capacity_kwp)
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
