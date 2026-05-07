"""Internal battery v7 detection logic migrated from legacy script."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

LOCAL_TIMEZONE = "Europe/Zurich"
WEATHER_MERGE_DIRECTION = "backward"
DARK_DAY_RAD_MAX_W = 100.0
SUNNY_DAY_RAD_MIN_W = 100.0
MIN_DARK_REFERENCE_DAYS = 2
MIN_SUNNY_MATCH_DAYS = 2
TEMP_MATCH_BUFFER_C = 2.0
SUNSET_SCAN_START_HOUR = 14
SUNSET_RAD_THRESHOLD_W = 12
DEFAULT_SUNSET_TIME = time(18, 0)
EVENING_END_TIME = "23:45"
RAD_W_TO_KWH_PER_15MIN = 0.00025
PV_POTENTIAL_EFFICIENCY = 0.85
DEFAULT_PV_CAPACITY_KWP = 5.0
BATTERY_CLASSIFICATION_THRESHOLD = 0.5
ENFORCE_PV_FOR_BATTERY_DETECTION = True
NON_PV_REJECTION_STATUS = "REJECTED: No PV Signal"
SIGMOID_INTERCEPT = -3.0
SIGMOID_PEAK_SHIFT_WEIGHT = 0.04
SIGMOID_GAP_RATIO_WEIGHT = 6.2
INJECTION_RATIO_REFERENCE = 0.4
INJECTION_BONUS_WEIGHT = 2.0
SIGMOID_CONSISTENCY_WEIGHT = 1.1
SIGMOID_SHIFT_RATIO_WEIGHT = 0.7
STRICT_MIN_MATCHED_SUNNY_DAYS = 7
LOW_MATCHED_DAYS_Z_PENALTY = 0.5
CONSISTENCY_GAP_THRESHOLD_KWH = 0.4
INJECTION_RATIO_MIN_POTENTIAL_KWH = 0.1
INJECTION_RATIO_CLIP_MIN = 0.0
INJECTION_RATIO_CLIP_MAX = 2.0
SIGMOID_CLIP_MIN = -20
SIGMOID_CLIP_MAX = 20
SECURITY_MIN_GAP_KWH = 2.0
SECURITY_MAX_GAP_KWH = 30.0
SECURITY_OUTSIDE_SCALER = 0.4
PHYSICAL_GAP_FACTOR = 2.5
PHYSICAL_OUTSIDE_SCALER = 0.7
SHIFT_BLEND_GAP_WEIGHT = 0.60
SHIFT_BLEND_SELFCONS_WEIGHT = 0.40
CAPACITY_ESTIMATE_QUANTILE = 0.90
CAPACITY_LOW_QUANTILE = 0.25
CAPACITY_HIGH_QUANTILE = 0.75
CAPACITY_MIN_REALISTIC_KWH = 5.0
CAPACITY_MAX_REALISTIC_KWH = 30.0
NOMINAL_CAPACITY_DISCHARGE_FRACTION = 0.25
MIN_SHIFT_FOR_NOMINAL_CAPACITY_KWH = 0.0
PV_ANCHOR_KWH_PER_KWP = 1.5
PV_ANCHOR_BLEND_WEIGHT = 0.55
CAPACITY_MAX_PER_KWP = 4.5
SET_CAPACITY_NAN_WHEN_BELOW_REALISTIC_FLOOR = True
PEAK_SHIFT_STEP_MINUTES = 15


def _pick_pv_capacity_kwp(pv_row):
    v = pv_row.get("pv_capacity_kwp", np.nan)
    return float(v) if pd.notna(v) and float(v) > 0 else np.nan


def _extract_pv_probability(pv_row):
    raw_value = pv_row.get("has_pv_prob", np.nan)
    if pd.isna(raw_value):
        return np.nan
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return np.nan


def _detect_added_selfcons_columns(columns):
    selected = []
    for col in columns:
        low = str(col).lower()
        if "self" in low and ("cons" in low or "consumption" in low):
            selected.append(col)
            continue
        if "added" in low and "pv" in low:
            selected.append(col)
    return sorted(set(selected))



def _detect_dynamic_sunset_time(avg_rad_profile):
    if avg_rad_profile.empty:
        return DEFAULT_SUNSET_TIME
    return next(
        (
            t
            for t in avg_rad_profile.index
            if t.hour >= SUNSET_SCAN_START_HOUR and avg_rad_profile[t] < SUNSET_RAD_THRESHOLD_W
        ),
        DEFAULT_SUNSET_TIME,
    )


def analyze_battery_residential_v7(
    df_customer,
    weather_df,
    pv_row,
    temp_buffer=TEMP_MATCH_BUFFER_C,
    sigmoid_intercept=None,
    strict_min_matched_sunny_days=None,
    low_matched_days_z_penalty=None,
    nominal_capacity_discharge_fraction=None,
    pv_anchor_kwh_per_kwp=None,
    pv_anchor_blend_weight=None,
):
    _sigmoid_intercept = SIGMOID_INTERCEPT if sigmoid_intercept is None else sigmoid_intercept
    _strict_min_days = STRICT_MIN_MATCHED_SUNNY_DAYS if strict_min_matched_sunny_days is None else strict_min_matched_sunny_days
    _low_days_penalty = LOW_MATCHED_DAYS_Z_PENALTY if low_matched_days_z_penalty is None else low_matched_days_z_penalty
    _discharge_fraction = NOMINAL_CAPACITY_DISCHARGE_FRACTION if nominal_capacity_discharge_fraction is None else nominal_capacity_discharge_fraction
    _pv_anchor_kwh_per_kwp = PV_ANCHOR_KWH_PER_KWP if pv_anchor_kwh_per_kwp is None else pv_anchor_kwh_per_kwp
    _pv_anchor_blend = PV_ANCHOR_BLEND_WEIGHT if pv_anchor_blend_weight is None else pv_anchor_blend_weight

    res = {
        "battery_prob": 0,
        "has_battery": "No",
        "status": "Success",
        "observed_gap_kwh": 0,
        "profiles": None,
        "estimated_battery_capacity_kwh": np.nan,
        "capacity_ci_lower_kwh": np.nan,
        "capacity_ci_upper_kwh": np.nan,
        "capacity_estimation_method": "nominal_from_shift_plus_pv_anchor",
        "capacity_capped_by_pv_scaling": False,
        "max_physical_from_pv_kwh": np.nan,
        "is_pv_customer": False,
        "has_pv_prob": np.nan,
        "pv_capacity_kwp": np.nan,
    }
    try:
        # Library convention: storage and merging are tz-naive UTC with matching
        # datetime64[us] unit for merge_asof. Defensive normalisation for inputs
        # that arrive tz-aware.
        idx = df_customer.index
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df_customer.index = idx.astype("datetime64[us]")

        widx = pd.to_datetime(weather_df.index, errors="coerce")
        if widx.tz is not None:
            widx = widx.tz_convert("UTC").tz_localize(None)
        weather_df = weather_df.copy()
        weather_df.index = widx.astype("datetime64[us]")
        weather_df = weather_df[~weather_df.index.isna()].sort_index()
        merged = pd.merge_asof(
            df_customer.sort_index(),
            weather_df,
            left_index=True,
            right_index=True,
            direction=WEATHER_MERGE_DIRECTION,
        )

        # Clock-time semantics ("evening", sunset, daily totals) are Swiss-local.
        # Switch merged to a tz-naive Europe/Zurich index so between_time(),
        # groupby(index.time), index.normalize() and resample("D") all use local
        # civil time. Keeping the dtype tz-naive preserves downstream behaviour.
        merged.index = (
            merged.index.tz_localize("UTC")
            .tz_convert(LOCAL_TIMEZONE)
            .tz_localize(None)
            .astype("datetime64[us]")
        )

        merged["rad_kwh_per_m2"] = merged["global_rad_W"] * RAD_W_TO_KWH_PER_15MIN
        daily = merged.resample("D").agg(
            {
                "t_2m_C": "mean",
                "global_rad_W": "max",
                "rad_kwh_per_m2": "sum",
                "PROD_KWH": "sum",
            }
        ).dropna()

        dark_days = daily[daily["global_rad_W"] < DARK_DAY_RAD_MAX_W].index
        sunny_days = daily[daily["global_rad_W"] >= SUNNY_DAY_RAD_MIN_W].index
        if len(dark_days) < MIN_DARK_REFERENCE_DAYS:
            res["status"] = "REJECTED: No Baseline"
            return res
        if len(sunny_days) < MIN_SUNNY_MATCH_DAYS:
            res["status"] = "REJECTED: No Sunny Match"
            return res

        merged_dates = merged.index.normalize()
        dark_mask_for_sunset = np.isin(merged_dates, pd.DatetimeIndex(dark_days).normalize())
        sunny_mask_for_sunset = np.isin(merged_dates, pd.DatetimeIndex(sunny_days).normalize())
        dark_slice = merged.loc[dark_mask_for_sunset]
        sunny_slice = merged.loc[sunny_mask_for_sunset]
        dark_rad_prof = dark_slice.groupby(dark_slice.index.time)["global_rad_W"].mean()
        sunny_rad_prof = sunny_slice.groupby(sunny_slice.index.time)["global_rad_W"].mean()
        dark_sunset_time = _detect_dynamic_sunset_time(dark_rad_prof)
        sunny_sunset_time = _detect_dynamic_sunset_time(sunny_rad_prof)
        analysis_start_time = max(dark_sunset_time, sunny_sunset_time)

        added_selfcons_cols = _detect_added_selfcons_columns(merged.columns)
        if added_selfcons_cols:
            for col in added_selfcons_cols:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
            merged["added_selfcons_kwh_est"] = merged[added_selfcons_cols].sum(axis=1)
        else:
            merged["added_selfcons_kwh_est"] = 0.0

        evening_daily = (
            merged.between_time(analysis_start_time, EVENING_END_TIME)
            .resample("D")
            .agg(
                {
                    "CONSO_KWH": "sum",
                    "PROD_KWH": "sum",
                    "rad_kwh_per_m2": "sum",
                    "added_selfcons_kwh_est": "sum",
                    "t_2m_C": "mean",
                }
            )
            .dropna()
            .rename(
                columns={
                    "CONSO_KWH": "conso_evening_kwh",
                    "PROD_KWH": "prod_evening_kwh",
                    "rad_kwh_per_m2": "rad_evening_kwh_per_m2",
                    "t_2m_C": "temp_mean_c",
                }
            )
        )
        if evening_daily.empty:
            res["status"] = "REJECTED: No Evening Features"
            return res

        temp_band = temp_buffer if temp_buffer and temp_buffer > 0 else TEMP_MATCH_BUFFER_C
        evening_daily["temp_bin"] = (
            np.round(evening_daily["temp_mean_c"] / temp_band) * temp_band
        ).astype(float)

        dark_daily = evening_daily.loc[
            evening_daily.index.intersection(pd.DatetimeIndex(dark_days))
        ].copy()
        sunny_daily = evening_daily.loc[
            evening_daily.index.intersection(pd.DatetimeIndex(sunny_days))
        ].copy()
        if len(dark_daily) < MIN_DARK_REFERENCE_DAYS:
            res["status"] = "REJECTED: No Baseline"
            return res
        if len(sunny_daily) < MIN_SUNNY_MATCH_DAYS:
            res["status"] = "REJECTED: No Sunny Match"
            return res

        dark_by_bin = dark_daily.groupby("temp_bin").agg(
            dark_evening_baseline_kwh=("conso_evening_kwh", "mean"),
            dark_days_in_bin=("conso_evening_kwh", "count"),
        )
        sunny_matched = sunny_daily.merge(
            dark_by_bin, left_on="temp_bin", right_index=True, how="inner"
        )
        if len(sunny_matched) < MIN_SUNNY_MATCH_DAYS:
            res["status"] = "REJECTED: No Temp-Matched Sunny"
            return res

        matched_dark_mask = np.isin(
            merged_dates, pd.DatetimeIndex(dark_daily.index).normalize()
        )
        matched_sunny_mask = np.isin(
            merged_dates, pd.DatetimeIndex(sunny_matched.index).normalize()
        )
        dark_evening = merged.loc[matched_dark_mask].between_time(
            analysis_start_time, EVENING_END_TIME
        )
        sunny_evening_profile = merged.loc[matched_sunny_mask].between_time(
            analysis_start_time, EVENING_END_TIME
        )
        prof_dark = dark_evening.groupby(dark_evening.index.time)["CONSO_KWH"].mean()
        prof_sun = sunny_evening_profile.groupby(
            sunny_evening_profile.index.time
        )["CONSO_KWH"].mean()
        prof_pair = pd.concat(
            [prof_dark.rename("dark_kwh"), prof_sun.rename("sun_kwh")],
            axis=1,
        ).dropna()
        if prof_pair.empty:
            res["status"] = "REJECTED: No Matched Profiles"
            return res

        sunny_matched["evening_gap_kwh"] = (
            sunny_matched["dark_evening_baseline_kwh"] - sunny_matched["conso_evening_kwh"]
        ).clip(lower=0.0)
        sunny_matched["gap_ratio"] = np.where(
            sunny_matched["dark_evening_baseline_kwh"] > 0.0,
            sunny_matched["evening_gap_kwh"] / sunny_matched["dark_evening_baseline_kwh"],
            0.0,
        )

        pv_capacity_kwp = _pick_pv_capacity_kwp(pv_row)
        pv_prob = _extract_pv_probability(pv_row)
        is_pv_customer = bool(pv_row.get("has_pv", False))
        pv_cap_for_potential = (
            pv_capacity_kwp if pd.notna(pv_capacity_kwp) else DEFAULT_PV_CAPACITY_KWP
        )
        sunny_matched["pv_potential_kwh"] = (
            pv_cap_for_potential
            * sunny_matched["rad_evening_kwh_per_m2"]
            * PV_POTENTIAL_EFFICIENCY
        ).clip(lower=0.0)
        sunny_matched["available_surplus_kwh"] = (
            sunny_matched["pv_potential_kwh"] - sunny_matched["prod_evening_kwh"]
        ).clip(lower=0.0)
        sunny_matched["added_selfcons_kwh_est"] = sunny_matched[
            "added_selfcons_kwh_est"
        ].clip(lower=0.0)

        has_added_signal = float(sunny_matched["added_selfcons_kwh_est"].sum()) > 0.0
        if has_added_signal:
            sunny_matched["inferred_shift_kwh"] = (
                SHIFT_BLEND_GAP_WEIGHT * sunny_matched["evening_gap_kwh"]
                + SHIFT_BLEND_SELFCONS_WEIGHT * sunny_matched["added_selfcons_kwh_est"]
            )
        else:
            sunny_matched["inferred_shift_kwh"] = np.minimum(
                sunny_matched["evening_gap_kwh"],
                sunny_matched["available_surplus_kwh"].fillna(np.inf),
            )
        sunny_matched["inferred_shift_kwh"] = sunny_matched["inferred_shift_kwh"].clip(
            lower=0.0
        )
        valid_shift = sunny_matched["inferred_shift_kwh"].dropna()

        # NumPy added trapezoid in newer versions; keep backward compatibility.
        area_fn = getattr(np, "trapezoid", np.trapz)
        dark_energy = area_fn(prof_pair["dark_kwh"].values)
        sunny_energy = area_fn(prof_pair["sun_kwh"].values)
        avg_gap_kwh = dark_energy - sunny_energy
        peak_shift = max(0, (
            np.argmax(prof_pair["sun_kwh"].values)
            - np.argmax(prof_pair["dark_kwh"].values)
        )) * PEAK_SHIFT_STEP_MINUTES
        gap_ratio = float(np.nanmedian(sunny_matched["gap_ratio"])) if len(sunny_matched) else 0.0
        consistency = (
            float(np.mean(sunny_matched["evening_gap_kwh"] > CONSISTENCY_GAP_THRESHOLD_KWH))
            if len(sunny_matched)
            else 0.0
        )
        valid_potential = sunny_matched["pv_potential_kwh"].replace(0, np.nan)
        shift_ratio = (
            float(np.nanmedian(sunny_matched["inferred_shift_kwh"] / valid_potential))
            if len(sunny_matched)
            else 0.0
        )

        pot_high = (
            pv_cap_for_potential
            * daily.loc[sunny_days, "rad_kwh_per_m2"].mean()
            * PV_POTENTIAL_EFFICIENCY
        )
        avg_inj = daily.loc[sunny_days, "PROD_KWH"].mean()
        inj_ratio = np.clip(
            avg_inj / pot_high if pot_high > INJECTION_RATIO_MIN_POTENTIAL_KWH else 1.0,
            INJECTION_RATIO_CLIP_MIN,
            INJECTION_RATIO_CLIP_MAX,
        )

        inj_bonus = INJECTION_BONUS_WEIGHT * (INJECTION_RATIO_REFERENCE - inj_ratio)
        z = (
            _sigmoid_intercept
            + (SIGMOID_PEAK_SHIFT_WEIGHT * peak_shift)
            + (SIGMOID_GAP_RATIO_WEIGHT * gap_ratio)
            + inj_bonus
            + SIGMOID_CONSISTENCY_WEIGHT * float(np.clip(consistency, 0.0, 1.0))
            + SIGMOID_SHIFT_RATIO_WEIGHT
            * float(np.clip(np.nan_to_num(shift_ratio, nan=0.0), 0.0, 1.0))
        )
        if len(sunny_matched) < _strict_min_days:
            z -= _low_days_penalty

        prob = 1 / (1 + np.exp(-np.clip(z, SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX)))

        security_scaler = (
            1.0
            if SECURITY_MIN_GAP_KWH <= avg_gap_kwh <= SECURITY_MAX_GAP_KWH
            else SECURITY_OUTSIDE_SCALER
        )
        phys_scaler = (
            1.0
            if avg_gap_kwh <= (pot_high * PHYSICAL_GAP_FACTOR)
            else PHYSICAL_OUTSIDE_SCALER
        )

        final_prob = prob * security_scaler * phys_scaler

        max_physical_from_pv_kwh = (
            float(max(CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_PER_KWP * pv_capacity_kwp))
            if pd.notna(pv_capacity_kwp)
            else np.nan
        )
        capacity_capped_by_pv_scaling = False

        cap_method = "nominal_from_shift_plus_pv_anchor"
        capacity_shift = valid_shift[valid_shift >= MIN_SHIFT_FOR_NOMINAL_CAPACITY_KWH]
        if capacity_shift.empty:
            # No observable shift: fall back to PV-anchor estimate if PV size is known.
            if pd.notna(pv_capacity_kwp):
                pv_anchor = pv_capacity_kwp * _pv_anchor_kwh_per_kwp
                cap_est = float(np.clip(pv_anchor, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH))
                cap_low = cap_est
                cap_high = cap_est
                cap_method = "pv_anchor_only"
            else:
                cap_est = np.nan
                cap_low = np.nan
                cap_high = np.nan
                cap_method = "no_estimate"
        else:
            nominal_capacity_series = capacity_shift / max(_discharge_fraction, 0.05)

            if pd.notna(pv_capacity_kwp):
                pv_anchor = pv_capacity_kwp * _pv_anchor_kwh_per_kwp
                nominal_capacity_series = (
                    (1.0 - _pv_anchor_blend) * nominal_capacity_series
                    + _pv_anchor_blend * pv_anchor
                )
                cap_method = "nominal_from_shift_plus_pv_anchor"
            else:
                cap_method = "nominal_from_shift_only"

            raw_cap_est = float(
                nominal_capacity_series.quantile(CAPACITY_ESTIMATE_QUANTILE)
            )
            cap_low = float(nominal_capacity_series.quantile(CAPACITY_LOW_QUANTILE))
            cap_high = float(nominal_capacity_series.quantile(CAPACITY_HIGH_QUANTILE))
            cap_est = raw_cap_est

            if pd.notna(max_physical_from_pv_kwh):
                capacity_capped_by_pv_scaling = raw_cap_est > max_physical_from_pv_kwh

            if SET_CAPACITY_NAN_WHEN_BELOW_REALISTIC_FLOOR and raw_cap_est < CAPACITY_MIN_REALISTIC_KWH:
                cap_est = np.nan
                cap_low = np.nan
                cap_high = np.nan
                cap_method = "no_estimate"
            else:
                cap_est = float(
                    np.clip(cap_est, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH)
                )
                cap_low = float(
                    np.clip(cap_low, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH)
                )
                cap_high = float(
                    np.clip(cap_high, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH)
                )

        battery_detected_by_score = final_prob >= BATTERY_CLASSIFICATION_THRESHOLD
        battery_detected = (
            battery_detected_by_score and is_pv_customer
            if ENFORCE_PV_FOR_BATTERY_DETECTION
            else battery_detected_by_score
        )
        if (
            ENFORCE_PV_FOR_BATTERY_DETECTION
            and battery_detected_by_score
            and not is_pv_customer
        ):
            res["status"] = NON_PV_REJECTION_STATUS

        res.update(
            {
                "battery_prob": round(final_prob * 100, 2),
                "has_battery": "Yes" if battery_detected else "No",
                "observed_gap_kwh": round(avg_gap_kwh, 3),
                "profiles": (prof_dark, prof_sun),
                "estimated_battery_capacity_kwh": round(cap_est, 3)
                if pd.notna(cap_est)
                else np.nan,
                "capacity_ci_lower_kwh": round(cap_low, 3)
                if pd.notna(cap_low)
                else np.nan,
                "capacity_ci_upper_kwh": round(cap_high, 3)
                if pd.notna(cap_high)
                else np.nan,
                "capacity_estimation_method": cap_method,
                "capacity_capped_by_pv_scaling": bool(capacity_capped_by_pv_scaling),
                "max_physical_from_pv_kwh": round(float(max_physical_from_pv_kwh), 3)
                if pd.notna(max_physical_from_pv_kwh)
                else np.nan,
                "is_pv_customer": bool(is_pv_customer),
                "has_pv_prob": round(float(pv_prob), 4) if pd.notna(pv_prob) else np.nan,
                "pv_capacity_kwp": round(float(pv_capacity_kwp), 3)
                if pd.notna(pv_capacity_kwp)
                else np.nan,
            }
        )
        return res

    except Exception as e:
        res["status"] = f"ERROR: {e}"
        return res


def get_dynamic_evening_window(daily_data, sunset_threshold=20.0):
    """Find first timestamp below threshold after daily radiation peak."""
    idx_max_rad = daily_data["global_rad_W"].idxmax()
    afternoon_data = daily_data.loc[idx_max_rad:]
    sunset_indices = afternoon_data[
        afternoon_data["global_rad_W"] < sunset_threshold
    ].index
    if not sunset_indices.empty:
        return sunset_indices[0]
    return None
