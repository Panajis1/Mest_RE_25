"""PV detection — wraps compute_pv_indicators and classify_pv_customers from pv_detection.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from re_nilm.detectors.base import AbstractDetector


class PVDetector(AbstractDetector):
    """Unsupervised PV detector based on production-radiation correlation and delta indicators.

    Mirrors the logic of pv_detection.py:compute_pv_indicators() and classify_pv_customers(),
    adapted to the single-customer streaming interface.

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

        yearly_prod = float(merged["PROD_KWH"].sum())

        # Production vs radiation correlation and regression slope
        g = merged.dropna(subset=["global_rad_W"])
        if g["global_rad_W"].var() > 0 and g["PROD_KWH"].var() > 0:
            corr_prod_rad = float(g["PROD_KWH"].corr(g["global_rad_W"]))
            x = g["global_rad_W"].to_numpy()
            y = g["PROD_KWH"].to_numpy()
            x_mean, y_mean = x.mean(), y.mean()
            denom = ((x - x_mean) ** 2).sum()
            beta = float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom > 0 else 0.0
        else:
            corr_prod_rad = np.nan
            beta = 0.0

        # Delta indicators: midday on high-rad vs low-rad days
        merged["_date"] = merged["_ts"].dt.date
        daily = merged.groupby("_date").agg(
            G_midday=("global_rad_W", "max"),
            Prod_midday=("PROD_KWH", lambda x: x.iloc[len(x) // 3: 2 * len(x) // 3].mean()),
            Net_midday=("CONSO_KWH", lambda x: x.iloc[len(x) // 3: 2 * len(x) // 3].mean()),
        )
        high = daily[daily["G_midday"] >= 400]
        low = daily[daily["G_midday"] < 100]

        DeltaProd = (
            float(high["Prod_midday"].mean() - low["Prod_midday"].mean())
            if not high.empty and not low.empty else np.nan
        )
        DeltaNet = (
            float(high["Net_midday"].mean() - low["Net_midday"].mean())
            if not high.empty and not low.empty else np.nan
        )

        # Classification
        has_pv = bool(
            (yearly_prod > self.min_yearly_prod_kwh)
            or (not np.isnan(corr_prod_rad) and corr_prod_rad > self.corr_threshold)
            or (not np.isnan(DeltaProd) and DeltaProd > 0.01)
            or (not np.isnan(DeltaNet) and DeltaNet < self.delta_net_threshold)
        )

        corr_clipped = corr_prod_rad if not np.isnan(corr_prod_rad) else 0.0
        prob_pv = float(np.clip((corr_clipped - 0.1) / 0.4, 0.0, 1.0)) if has_pv else 0.0

        return {
            "customer_id": customer_id,
            "has_pv": has_pv,
            "prob_pv": prob_pv,
            "yearly_prod_kwh": yearly_prod,
            "corr_prod_rad": corr_prod_rad,
            "DeltaProd": DeltaProd,
            "DeltaNet": DeltaNet,
            "beta_regression": beta,
        }
