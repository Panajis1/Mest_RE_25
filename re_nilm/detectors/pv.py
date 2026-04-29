"""PV detector using internal re_nilm PV daily feature functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from re_nilm.detectors.base import AbstractDetector
from re_nilm.features.pv_daily import (
    build_customer_daily_features,
    classify_pv_customers,
    compute_daily_weather,
    compute_pv_indicators,
)


def _empty_result(customer_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "has_pv": False,
        "prob_pv": 0.0,
        "yearly_prod_kwh": 0.0,
        "corr_prod_rad": np.nan,
        "DeltaProd": np.nan,
        "DeltaNet": np.nan,
        "beta_regression": np.nan,
    }


class PVDetector(AbstractDetector):
    """Unsupervised PV detector based on production-radiation correlation and delta indicators.

    Uses internal re_nilm feature helpers:
    compute_daily_weather → build_customer_daily_features →
    compute_pv_indicators → classify_pv_customers.

    High/low radiation days are classified using monthly 20th/80th percentile quantiles
    (not fixed thresholds), matching the original batch implementation exactly.

    Args:
        corr_threshold: Minimum prod-radiation correlation to flag as PV.
        delta_net_threshold: DeltaNet below this triggers PV flag.
        min_yearly_prod_kwh: Minimum total annual PROD_KWH to be considered PV.
    """

    def __init__(
        self,
        corr_threshold: float = 0.3,
        delta_net_threshold: float = -0.1,
        min_yearly_prod_kwh: float = 1.0,
    ):
        self.corr_threshold = corr_threshold
        self.delta_net_threshold = delta_net_threshold
        self.min_yearly_prod_kwh = min_yearly_prod_kwh

    def predict_customer(
        self,
        customer_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        **context,
    ) -> dict | None:
        """Detect PV for a single customer.

        Args:
            customer_df: 15-min load data [DT_UTC, CONSO_KWH, PROD_KWH, ID].
            weather_df: Weather data [dt_utc, t_2m_C, global_rad_W].

        Returns:
            Dict with keys: customer_id, has_pv, prob_pv, yearly_prod_kwh,
            corr_prod_rad, DeltaProd, DeltaNet, beta_regression.
        """
        if customer_df.empty:
            return None

        customer_id = str(customer_df["ID"].iloc[0])

        # Build weather indexed by timestamp (required by compute_daily_weather)
        w = weather_df.copy()
        w["dt_utc"] = pd.to_datetime(w["dt_utc"]).astype("datetime64[us]")
        weather_idx = (
            w.rename(columns={"dt_utc": "timestamp"})
            .set_index("timestamp")
            .sort_index()
        )

        # Prepare customer data with global_rad_W merged in
        df = customer_df.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce").astype("datetime64[us]")
        df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")
        df["ID"] = customer_id

        weather_reset = (
            weather_idx[["global_rad_W"]]
            .rename_axis("DT_UTC")
            .reset_index()
        )
        weather_reset["DT_UTC"] = weather_reset["DT_UTC"].astype("datetime64[us]")
        df_with_rad = pd.merge_asof(
            df,
            weather_reset,
            on="DT_UTC",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )

        try:
            daily_weather = compute_daily_weather(weather_idx)
            if daily_weather.empty:
                return _empty_result(customer_id)

            daily_features = build_customer_daily_features(df_with_rad, daily_weather)
            if daily_features.empty:
                return _empty_result(customer_id)

            pv_indicators = compute_pv_indicators(daily_features, df_with_rad)
            if pv_indicators.empty:
                return _empty_result(customer_id)

            pv_classified = classify_pv_customers(
                pv_indicators,
                corr_threshold=self.corr_threshold,
                delta_net_threshold=self.delta_net_threshold,
                min_yearly_prod=self.min_yearly_prod_kwh,
            )
        except Exception as exc:
            return {
                **_empty_result(customer_id),
                "has_pv": False,
                "prob_pv": np.nan,
                "status": f"error: {exc}",
            }

        if pv_classified.empty:
            return _empty_result(customer_id)

        row = pv_classified.iloc[0]
        return {
            "customer_id": customer_id,
            "has_pv": bool(row.get("has_pv", False)),
            "prob_pv": float(row.get("has_pv_prob", 0.0)),
            "yearly_prod_kwh": float(row.get("yearly_prod", np.nan)),
            "corr_prod_rad": float(row.get("corr_prod_rad", np.nan)),
            "DeltaProd": float(row.get("DeltaProd", np.nan)),
            "DeltaNet": float(row.get("DeltaNet", np.nan)),
            "beta_regression": float(row.get("beta_regression", np.nan)),
        }
