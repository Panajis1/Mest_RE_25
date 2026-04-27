"""EV (electric vehicle) detector — heuristic, no ML model required.

Ported from ev_scripts/newlogic_ev.ipynb (detect_ev_probability_pv_aware).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from re_nilm.detectors.base import AbstractDetector

_STEP_H = 0.25  # 15-min timestep in hours


def _extract_sessions(
    mask: pd.Series,
    load_col: pd.Series,
    min_len: int,
    max_rel_std: float,
) -> pd.DataFrame:
    """Group consecutive True values in mask into charging sessions."""
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, ok in enumerate(mask):
        if ok:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                groups.append(cur)
            cur = []
    if len(cur) >= min_len:
        groups.append(cur)

    sessions = []
    for g in groups:
        load = load_col.iloc[g]
        mean_p = load.mean()
        if mean_p > 0 and load.std() / mean_p <= max_rel_std:
            duration_h = len(g) * _STEP_H
            sessions.append(
                {
                    "start": load_col.index[g[0]],
                    "duration_h": duration_h,
                    "mean_power_kW": mean_p,
                    "energy_kWh": mean_p * duration_h,
                }
            )
    return pd.DataFrame(sessions)


def _ev_probability(
    df: pd.DataFrame,
    max_yearly_mwh: float,
    grid_threshold_kw: float,
    min_session_steps: int,
    max_rel_std: float,
) -> dict:
    """Core EV scoring logic (Algorithm A from newlogic_ev.ipynb).

    Args:
        df: DT_UTC-indexed DataFrame with columns [CONSO_KWH, PROD_KWH].
        max_yearly_mwh: Yearly consumption filter (commercial buildings).
        grid_threshold_kw: Minimum active load to flag as grid charging.
        min_session_steps: Minimum number of 15-min steps per session.
        max_rel_std: Maximum std/mean ratio for a session to be accepted.

    Returns:
        Dict with EV_probability and session counts.
    """
    yearly_mwh = df["CONSO_KWH"].sum() / 1000.0

    if yearly_mwh >= max_yearly_mwh:
        return {
            "EV_probability": 0.0,
            "filtered_out": True,
            "yearly_consumption_MWh": round(float(yearly_mwh), 3),
            "grid_sessions": 0,
            "pv_sessions": 0,
            "total_sessions": 0,
        }

    df = df.copy()
    df["Consommation"] = df["CONSO_KWH"] / _STEP_H  # kWh → kW
    df["Excedent"] = df["PROD_KWH"] / _STEP_H

    hour = df.index.hour
    night_mask = (hour >= 1) & (hour < 4)
    df["net_load"] = df["Consommation"] - df["Excedent"]
    baseload = float(df.loc[night_mask, "net_load"].median()) if night_mask.any() else 0.0

    # Grid charging sessions: sustained active load above threshold
    df["active_load"] = df["net_load"] - baseload
    grid_mask = df["active_load"] >= grid_threshold_kw
    grid_sess = _extract_sessions(grid_mask, df["active_load"], min_session_steps, max_rel_std)

    # PV-surplus charging sessions: PROD drops + stable net_load
    pv_mask = (
        (df["Excedent"].diff() < -0.5)
        & (df["net_load"].rolling(4).std() < 0.25)
        & (df["net_load"] > 0.8)
    )
    pv_sess = _extract_sessions(pv_mask, df["net_load"], min_session_steps, max_rel_std)

    sessions = pd.concat([grid_sess, pv_sess], ignore_index=True)

    if sessions.empty:
        ev_prob = 0.0
    else:
        def _logistic(x: float, x0: float, k: float) -> float:
            return 1.0 / (1.0 + np.exp(-k * (x - x0)))

        def _clip01(x: float) -> float:
            return max(0.0, min(1.0, x))

        sessions["week"] = (
            sessions["start"].dt.tz_localize(None).dt.to_period("W")
        )
        n_sessions = len(sessions)
        max_weekly = int(sessions.groupby("week").size().max())
        mean_power = float(sessions["mean_power_kW"].mean())
        mean_energy = float(sessions["energy_kWh"].mean())
        flatness = float(sessions["energy_kWh"].std() / (mean_energy + 1e-6))

        ev_prob = _clip01(
            0.30 * _logistic(n_sessions, 8, 0.4)
            + 0.25 * _logistic(max_weekly, 2, 1.2)
            + 0.20 * _clip01((mean_power - 0.8) / (2.5 - 0.8))
            + 0.15 * _clip01((mean_energy - 4.0) / (20.0 - 4.0))
            + 0.10 * _clip01(1.0 - flatness)
        )

    return {
        "EV_probability": round(ev_prob, 3),
        "filtered_out": False,
        "yearly_consumption_MWh": round(float(yearly_mwh), 3),
        "grid_sessions": len(grid_sess),
        "pv_sessions": len(pv_sess),
        "total_sessions": len(sessions),
    }


class EVDetector(AbstractDetector):
    """Heuristic EV detector based on charging session pattern analysis.

    Uses a soft probability score from multiple signals:
    session count, weekly frequency, mean power, mean energy, flatness.
    Detects both grid charging and PV-surplus charging.

    No serialized model file required — fully heuristic.

    Args:
        prob_threshold: Probability cut-off for has_ev classification.
        max_yearly_mwh: Skip customers with yearly consumption above this
            (commercial/industrial buildings).
        grid_threshold_kw: Minimum sustained active load to count as a grid
            charging session.
        min_session_steps: Minimum number of 15-min intervals per session.
        max_rel_std: Maximum coefficient of variation (std/mean) for a session
            to be accepted as flat/stable charging.
    """

    def __init__(
        self,
        prob_threshold: float = 0.3,
        max_yearly_mwh: float = 100.0,
        grid_threshold_kw: float = 2.8,
        min_session_steps: int = 6,
        max_rel_std: float = 0.15,
    ):
        self.prob_threshold = prob_threshold
        self.max_yearly_mwh = max_yearly_mwh
        self.grid_threshold_kw = grid_threshold_kw
        self.min_session_steps = min_session_steps
        self.max_rel_std = max_rel_std

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
        df = df.dropna(subset=["DT_UTC", "CONSO_KWH"]).sort_values("DT_UTC")
        if df.empty:
            return None

        # Algorithm expects a UTC-localized DatetimeIndex
        df = df.set_index("DT_UTC")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Fill missing PROD_KWH with 0 (non-PV customers have no export column)
        if "PROD_KWH" not in df.columns:
            df["PROD_KWH"] = 0.0
        else:
            df["PROD_KWH"] = df["PROD_KWH"].fillna(0.0)

        scores = _ev_probability(
            df,
            max_yearly_mwh=self.max_yearly_mwh,
            grid_threshold_kw=self.grid_threshold_kw,
            min_session_steps=self.min_session_steps,
            max_rel_std=self.max_rel_std,
        )

        ev_prob = scores["EV_probability"]
        return {
            "customer_id": customer_id,
            "has_ev": bool(ev_prob >= self.prob_threshold),
            "prob_ev": ev_prob,
            "ev_grid_sessions": scores["grid_sessions"],
            "ev_pv_sessions": scores["pv_sessions"],
            "ev_total_sessions": scores["total_sessions"],
            "ev_yearly_mwh": scores["yearly_consumption_MWh"],
            "ev_filtered_out": scores["filtered_out"],
        }
