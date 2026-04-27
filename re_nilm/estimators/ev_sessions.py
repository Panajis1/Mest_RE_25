"""EV session estimator — per-event charging detail for EV-positive customers.

Ported from ev_scripts/run_for_all_data.ipynb (detect_ev_charging_fast).
Applied only to customers where EVDetector.has_ev == True.
"""

from __future__ import annotations

import pytz
import numpy as np
import pandas as pd

from re_nilm.estimators.base import AbstractEstimator

_TZ_CH = pytz.timezone("Europe/Zurich")


def _detect_ev_charging_fast(
    df: pd.DataFrame,
    step_threshold_kwh: float = 1.5,
    min_duration_minutes: int = 60,
) -> pd.DataFrame:
    """Detect discrete EV charging events via diff-threshold method (Algorithm B).

    Args:
        df: DT_UTC-indexed DataFrame with column CONSO_KWH (15-min energy).
        step_threshold_kwh: kWh diff that triggers a session start/end.
        min_duration_minutes: Minimum session duration to keep.

    Returns:
        DataFrame of events: [start, end, day, duration_h, energy_kWh,
        consumption_start, consumption_trigger_start,
        consumption_before_end, consumption_trigger_end]
        or empty DataFrame if no events found.
    """
    df_15 = df["CONSO_KWH"].resample("15min").sum().fillna(0.0)
    diff = df_15.diff()

    starts = df_15.index[diff > step_threshold_kwh]
    ends_idx = list(df_15.index[diff < -step_threshold_kwh])

    events = []
    ei = 0
    n_ends = len(ends_idx)
    local_index = df_15.index.tz_convert(_TZ_CH)
    tz_map = dict(zip(df_15.index, local_index))

    for s in starts:
        while ei < n_ends and ends_idx[ei] <= s:
            ei += 1
        if ei >= n_ends:
            break

        end_time = ends_idx[ei]
        duration_min = (end_time - s).total_seconds() / 60.0
        if duration_min < min_duration_minutes:
            continue

        # Check steps during charging; exclude end_time (that's the step-down row).
        pre_end = end_time - pd.Timedelta(minutes=15)
        interval = df_15.loc[s:pre_end] if pre_end >= s else df_15.loc[s:s]
        if not (interval >= step_threshold_kwh).all():
            continue

        s_local = tz_map[s]
        e_local = tz_map[end_time]
        energy = float(interval.sum())

        events.append(
            {
                "start": s_local.strftime("%d.%m.%Y %H:%M"),
                "end": e_local.strftime("%d.%m.%Y %H:%M"),
                "day": s_local.strftime("%Y-%m-%d"),
                "duration_h": duration_min / 60.0,
                "energy_kWh": energy,
                "consumption_start": float(df_15.shift(1).loc[s]),
                "consumption_trigger_start": float(df_15.loc[s]),
                "consumption_before_end": float(
                    df_15.loc[end_time - pd.Timedelta(minutes=15)]
                    if end_time - pd.Timedelta(minutes=15) in df_15.index
                    else np.nan
                ),
                "consumption_trigger_end": float(df_15.loc[end_time]),
            }
        )

    return pd.DataFrame(events)


class EVSessionEstimator(AbstractEstimator):
    """Per-session EV charging detail estimator.

    Applied only to customers where EVDetector.has_ev == True.
    Produces a DataFrame of discrete charging events per customer,
    saved to ev_sessions_15min.parquet by the orchestrator.

    Args:
        step_threshold_kwh: kWh step in 15-min CONSO_KWH to trigger session.
        min_duration_minutes: Minimum session length to keep.
    """

    def __init__(
        self,
        step_threshold_kwh: float = 1.5,
        min_duration_minutes: int = 60,
    ):
        self.step_threshold_kwh = step_threshold_kwh
        self.min_duration_minutes = min_duration_minutes

    def estimate(
        self,
        customer_ts: pd.DataFrame,
        detection_result: dict,
        weather: pd.DataFrame,
    ) -> dict | None:
        if not detection_result.get("has_ev", False):
            return None

        customer_id = detection_result.get("customer_id", "")
        if customer_ts.empty:
            return None

        df = customer_ts.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
        df = df.dropna(subset=["DT_UTC", "CONSO_KWH"]).sort_values("DT_UTC")
        if df.empty:
            return None

        df = df.set_index("DT_UTC")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        sessions = _detect_ev_charging_fast(
            df,
            step_threshold_kwh=self.step_threshold_kwh,
            min_duration_minutes=self.min_duration_minutes,
        )

        if sessions.empty:
            return None

        sessions.insert(0, "customer_id", customer_id)
        return {"customer_id": customer_id, "ev_sessions": sessions}
