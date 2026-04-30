"""AC disaggregation estimator — wraps AcTwoStageModel from model/ac_disaggregation.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import joblib

from re_nilm.estimators.base import AbstractEstimator
from re_nilm.estimators._ac_disagg_v1 import (
    _add_calendar_and_dynamic_features,
    _compute_user_profile,
)


def _joblib_load_compat(path: Path):
    """Load a joblib file, bridging numpy 2.x + sklearn 1.8 → 1.x pickle formats.

    The AC disaggregator was saved in an environment with numpy 2.x and sklearn 1.8.
    Loading it under numpy 1.x / sklearn 1.3.x requires four compatibility shims:

    1. ``__bit_generator_ctor`` receives the *class* (e.g. PCG64) instead of the
       plain string ``'PCG64'`` — numpy 1.x raises ``ValueError``.

    2. ``__generator_ctor`` receives an already-constructed BitGenerator *instance*
       (numpy 2.x passes it directly) instead of a name string.

    3. ``BitGenerator.__setstate__`` may receive state in a format the Cython
       ``PCG64.state`` setter rejects — silently discarded (random state is
       irrelevant for inference).

    4. The pickle references two modules not resolvable in the current environment:
       - ``_loss`` → ``sklearn._loss._loss`` (compiled Cython extension; stored
         without the sklearn prefix in editable sklearn installs)
       - ``ac_disaggregation`` → legacy ``old_files/model/ac_disaggregation.py``
         (must be on sys.path to unpickle AcTwoStageModel)

    All patches are restored unconditionally in a finally block.
    """
    import sys

    try:
        return joblib.load(path)
    except (ValueError, TypeError, ModuleNotFoundError):
        pass

    import numpy.random._pickle as _np_pickle
    import numpy.random._pcg64 as _pcg64_mod
    import numpy.random as _np_random
    from numpy.random.bit_generator import BitGenerator as _BitGenerator

    _orig_bg_ctor = _np_pickle.__bit_generator_ctor
    _orig_gen_ctor = _np_pickle.__generator_ctor
    _OrigPCG64 = _pcg64_mod.PCG64

    # Locate the legacy model directory relative to this file's package root.
    _legacy_model_dir = str(Path(__file__).parents[2] / "old_files" / "model")

    class _CompatPCG64(_OrigPCG64):
        """PCG64 subclass that silently ignores incompatible numpy 2.x state."""
        def __setstate__(self, state):
            try:
                super().__setstate__(state)
            except Exception:
                pass  # leave freshly initialised; random state is inference-irrelevant

    def _bg_ctor_compat(bg="MT19937"):
        if isinstance(bg, type):
            return bg()
        return _orig_bg_ctor(bg)

    def _gen_ctor_compat(bit_generator_name="MT19937", *args, **kwargs):
        if isinstance(bit_generator_name, _BitGenerator):
            return _np_random.Generator(bit_generator_name)
        return _orig_gen_ctor(bit_generator_name, *args, **kwargs)

    _np_pickle.__bit_generator_ctor = _bg_ctor_compat
    _np_pickle.__generator_ctor = _gen_ctor_compat
    _pcg64_mod.PCG64 = _CompatPCG64  # type: ignore[assignment]
    _np_random.PCG64 = _CompatPCG64  # type: ignore[assignment]

    # Shim 4a: map bare '_loss' to the sklearn._loss Cython extension.
    import sklearn._loss._loss as _sk_loss_ext
    _added_loss = "_loss" not in sys.modules
    sys.modules.setdefault("_loss", _sk_loss_ext)

    # Shim 4b: make the legacy ac_disaggregation module importable.
    _added_legacy_path = _legacy_model_dir not in sys.path
    if _added_legacy_path:
        sys.path.insert(0, _legacy_model_dir)

    try:
        return joblib.load(path)
    finally:
        _pcg64_mod.PCG64 = _OrigPCG64  # type: ignore[assignment]
        _np_random.PCG64 = _OrigPCG64  # type: ignore[assignment]
        _np_pickle.__bit_generator_ctor = _orig_bg_ctor
        _np_pickle.__generator_ctor = _orig_gen_ctor
        if _added_loss:
            sys.modules.pop("_loss", None)
        if _added_legacy_path and _legacy_model_dir in sys.path:
            sys.path.remove(_legacy_model_dir)


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
        # Cache the dt-normalised + sorted weather frame so it isn't rebuilt
        # per customer. Keyed by id(weather_df); cache lifetime matches this
        # estimator instance (one cache entry per unique weather DF identity).
        self._weather_sorted_cache: dict = {}

    def _get_sorted_weather(self, weather: pd.DataFrame) -> pd.DataFrame:
        key = id(weather)
        cached = self._weather_sorted_cache.get(key)
        if cached is not None:
            return cached
        wdf = weather.copy()
        wdf["dt_utc"] = wdf["dt_utc"].astype("datetime64[us]")
        wdf = wdf.sort_values("dt_utc")
        self._weather_sorted_cache[key] = wdf
        return wdf

    @classmethod
    def load(cls, path: Path, features_path: Optional[Path] = None, **kwargs) -> "ACDisaggregationEstimator":
        """Load serialized AcTwoStageModel and optional feature_cols JSON sidecar."""
        model = _joblib_load_compat(Path(path))
        feature_cols = None
        if features_path is not None and Path(features_path).exists():
            with open(features_path) as f:
                feature_cols = json.load(f)
        elif hasattr(model, "feature_cols") and model.feature_cols is not None:
            feature_cols = list(model.feature_cols)
        return cls(model=model, feature_cols=feature_cols, **kwargs)

    def _build_features(
        self,
        customer_ts: pd.DataFrame,
        weather: pd.DataFrame,
        pv_capacity_kwp: float = 0.0,
    ) -> pd.DataFrame:
        """Build the feature DataFrame that AcTwoStageModel.predict() expects.

        Mirrors build_aligned_dataset / _compute_user_profile /
        _add_calendar_and_dynamic_features from ac_disaggregation.py.

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
        wdf = self._get_sorted_weather(weather)
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
        pv_capacity_kwp: float = 0.0,
    ) -> dict | None:
        customer_id = detection_result.get("customer_id", "")

        if not detection_result.get("has_ac", False):
            return None

        feat_df = self._build_features(customer_ts, weather, pv_capacity_kwp=pv_capacity_kwp)
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
