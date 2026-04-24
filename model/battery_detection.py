import pandas as pd
import numpy as np
import os
import gc
import sys
from datetime import time
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


# --- 1. Path & Setup ---
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import data.envdata as meteo
import data.fast_load_smart_meter as re_data

# --- 2. File Paths (edit first) ---
# Data source root containing parquet files and metadata.
DATA_DIR = "C:\\Users\\Aline\\Documents\\Studium\\Case Study\\processed_data\\ETHZ_ALL"
# Output directory for profile plots of detected battery customers.
PLOT_DIR = "residential_profiles_v7"
# Additional output folder under results/ for customer profile plots.
RESULTS_PLOT_DIR = "results/battery_detection_customer_plots"
# Input side files.
PV_SUMMARY_CSV = "scripts/prob_summary.csv"
WEATHER_CSV = "weather_data.csv"
# Final customer-level results output.
RESULTS_CSV = "battery_residential_results_v7.csv"

# --- 3. Tunable Parameters (edit here) ---
# Partner segment to process from metadata.
PARTNER_TYPE = "Particuliers"
# Batch size used by the parallel parquet loader.
BATCH_SIZE = 2
# Timezone handling for meter timestamps (input assumed UTC, analysis in local time).
INPUT_TIMEZONE = "UTC"
LOCAL_TIMEZONE = "Europe/Zurich"
# Merge policy when attaching weather observations to meter points.
WEATHER_MERGE_DIRECTION = "backward"
# Weather/solar filters used to build dark vs sunny day sets.
DARK_DAY_RAD_MAX_W = 100.0
SUNNY_DAY_RAD_MIN_W = 100.0
MIN_DARK_REFERENCE_DAYS = 2
MIN_SUNNY_MATCH_DAYS = 2
TEMP_MATCH_BUFFER_C = 2.0
# Evening profile extraction window.
SUNSET_SCAN_START_HOUR = 14
SUNSET_RAD_THRESHOLD_W = 12
DEFAULT_SUNSET_TIME = time(18, 0)
EVENING_END_TIME = "23:45"
# Unit conversion and PV potential assumptions.
RAD_W_TO_KWH_PER_15MIN = 0.00025
PV_POTENTIAL_EFFICIENCY = 0.85
DEFAULT_PV_CAPACITY_KWP = 5.0
# PV tagging/classification thresholds.
PV_PROBABILITY_THRESHOLD = 0.5
BATTERY_CLASSIFICATION_THRESHOLD = 0.5
# Option 1 accuracy guard: only classify battery if PV exists.
ENFORCE_PV_FOR_BATTERY_DETECTION = True
NON_PV_REJECTION_STATUS = "REJECTED: No PV Signal"
# Sigmoid scoring weights for battery probability.
SIGMOID_INTERCEPT = -1.8
SIGMOID_PEAK_SHIFT_WEIGHT = 0.05
SIGMOID_GAP_RATIO_WEIGHT = 7.0
INJECTION_RATIO_REFERENCE = 0.4
INJECTION_BONUS_WEIGHT = 2.5
INJECTION_RATIO_MIN_POTENTIAL_KWH = 0.1
INJECTION_RATIO_CLIP_MIN = 0.0
INJECTION_RATIO_CLIP_MAX = 2.0
SIGMOID_CLIP_MIN = -20
SIGMOID_CLIP_MAX = 20
# Physical/robustness scalers applied after sigmoid.
SECURITY_MIN_GAP_KWH = 2.0
SECURITY_MAX_GAP_KWH = 30.0
SECURITY_OUTSIDE_SCALER = 0.4
PHYSICAL_GAP_FACTOR = 2.5
PHYSICAL_OUTSIDE_SCALER = 0.6
# Capacity estimation quantiles and realistic output band.
SHIFT_BLEND_GAP_WEIGHT = 0.60
SHIFT_BLEND_SELFCONS_WEIGHT = 0.40
CAPACITY_ESTIMATE_QUANTILE = 0.90
CAPACITY_LOW_QUANTILE = 0.25
CAPACITY_HIGH_QUANTILE = 0.75
CAPACITY_MIN_REALISTIC_KWH = 5.0
CAPACITY_MAX_REALISTIC_KWH = 30.0
# Alternative nominal-capacity approach:
# We estimate nominal capacity from shifted energy assuming only a fraction
# of the battery is typically discharged in the evening.
NOMINAL_CAPACITY_DISCHARGE_FRACTION = 0.45
MIN_SHIFT_FOR_NOMINAL_CAPACITY_KWH = 0.5
PV_ANCHOR_KWH_PER_KWP = 2.0
PV_ANCHOR_BLEND_WEIGHT = 0.40
# Informative PV-based ceiling (reported only, not used as hard cap).
CAPACITY_MAX_PER_KWP = 3.5
# Use NaN instead of forcing 2 kWh when all inferred shifts are below the realistic floor.
SET_CAPACITY_NAN_WHEN_BELOW_REALISTIC_FLOOR = True
# Power estimation assumptions and quantiles.
DISCHARGE_INTERVAL_HOURS = 0.25
POWER_ESTIMATE_QUANTILE = 0.90
POWER_LOW_QUANTILE = 0.25
POWER_HIGH_QUANTILE = 0.75
# Derived feature scaling/plot styling constants.
PEAK_SHIFT_STEP_MINUTES = 15
PLOT_WIDTH = 8
PLOT_HEIGHT = 4
PLOT_DARK_ALPHA = 0.5
PLOT_FILL_ALPHA = 0.2
PLOT_BACKGROUND_COLOR = "black"
PLOT_TEXT_COLOR = "white"
PLOT_DARK_DAY_COLOR = "white"
PLOT_SUNNY_DAY_COLOR = "red"
# Runtime/reporting controls.
PROGRESS_PRINT_EVERY = 25
GC_EVERY_N_BATCHES = 10


def _pick_pv_capacity_kwp(pv_row):
    candidates = [
        pv_row.get('pv_capacity_kwp', np.nan),
        pv_row.get('pv_capacity_kwp_floor', np.nan),
        pv_row.get('pv_capacity_kwp_ci_upper', np.nan),
        pv_row.get('pv_capacity_ci_upper', np.nan),
    ]
    for value in candidates:
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return np.nan


def _extract_pv_probability(pv_row):
    raw_value = pv_row.get('has_pv_prob', np.nan)
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


def _prepare_weather_for_merge(weather_df):
    prepared = weather_df.copy()
    prepared.index = pd.to_datetime(prepared.index, errors='coerce')
    prepared = prepared[~prepared.index.isna()].sort_index()
    prepared.index = prepared.index.astype('datetime64[ns]')
    return prepared


def _detect_dynamic_sunset_time(avg_rad_profile):
    if avg_rad_profile.empty:
        return DEFAULT_SUNSET_TIME
    return next(
        (
            t for t in avg_rad_profile.index
            if t.hour >= SUNSET_SCAN_START_HOUR and avg_rad_profile[t] < SUNSET_RAD_THRESHOLD_W
        ),
        DEFAULT_SUNSET_TIME,
    )


def _estimate_partner_customer_count(data_dir, partner_type=PARTNER_TYPE):
    metadata_path = Path(data_dir) / "metadata"
    try:
        meta_df = pd.read_parquet(metadata_path, columns=["ID", "TYPE_PARTENAIRE_LIBELLE"])
        partner_ids = meta_df.loc[
            meta_df["TYPE_PARTENAIRE_LIBELLE"] == partner_type,
            "ID",
        ]
        return int(partner_ids.astype(str).nunique())
    except Exception:
        return None


def analyze_battery_residential_v7(df_customer, weather_df, pv_row, temp_buffer=TEMP_MATCH_BUFFER_C):
    res = {"battery_prob": 0, "has_battery": "No", "status": "Success", 
           "observed_gap_kwh": 0, "profiles": None,
           "estimated_battery_capacity_kwh": np.nan,
           "capacity_ci_lower_kwh": np.nan,
           "capacity_ci_upper_kwh": np.nan,
           "capacity_estimation_method": "nominal_from_shift_plus_pv_anchor",
           "capacity_capped_by_pv_scaling": False,
           "max_physical_from_pv_kwh": np.nan,
           "estimated_battery_power_kw": np.nan,
           "power_ci_lower_kw": np.nan,
           "power_ci_upper_kw": np.nan,
           "is_pv_customer": False,
           "has_pv_prob": np.nan,
           "pv_capacity_kwp": np.nan}  
    try:
        # --- Speed Op 1: Fast Alignment ---
        # Convert to local time and force nanosecond precision in one go
        if df_customer.index.tz is None:
            df_customer.index = df_customer.index.tz_localize(INPUT_TIMEZONE)
        
        # Localize to Zurich and strip for naive matching with prepared weather index.
        df_customer.index = df_customer.index.tz_convert(LOCAL_TIMEZONE).tz_localize(None).astype('datetime64[ns]')
        merged = pd.merge_asof(df_customer.sort_index(), weather_df, 
                              left_index=True, right_index=True, direction=WEATHER_MERGE_DIRECTION)

        # --- Speed Op 2: Pre-filter Solar Potential ---
        # Convert 15-min radiation from W/m2 to kWh/m2.
        merged['rad_kwh_per_m2'] = merged['global_rad_W'] * RAD_W_TO_KWH_PER_15MIN
        daily = merged.resample('D').agg({
            't_2m_C': 'mean', 
            'global_rad_W': 'max', 
            'rad_kwh_per_m2': 'sum',
            'PROD_KWH': 'sum'
        }).dropna()

        dark_days = daily[daily['global_rad_W'] < DARK_DAY_RAD_MAX_W].index
        sunny_days = daily[daily['global_rad_W'] >= SUNNY_DAY_RAD_MIN_W].index
        if len(dark_days) < MIN_DARK_REFERENCE_DAYS:
            res["status"] = "REJECTED: No Baseline"
            return res
        if len(sunny_days) < MIN_SUNNY_MATCH_DAYS:
            res["status"] = "REJECTED: No Sunny Match"
            return res

        # Dynamic sunset from both dark and sunny profiles.
        merged_dates = merged.index.normalize()
        dark_mask_for_sunset = np.isin(merged_dates, pd.DatetimeIndex(dark_days).normalize())
        sunny_mask_for_sunset = np.isin(merged_dates, pd.DatetimeIndex(sunny_days).normalize())
        dark_slice = merged.loc[dark_mask_for_sunset]
        sunny_slice = merged.loc[sunny_mask_for_sunset]
        dark_rad_prof = dark_slice.groupby(dark_slice.index.time)['global_rad_W'].mean()
        sunny_rad_prof = sunny_slice.groupby(sunny_slice.index.time)['global_rad_W'].mean()
        dark_sunset_time = _detect_dynamic_sunset_time(dark_rad_prof)
        sunny_sunset_time = _detect_dynamic_sunset_time(sunny_rad_prof)
        # Use the later sunset so dark and sunny windows are directly comparable.
        analysis_start_time = max(dark_sunset_time, sunny_sunset_time)

        # Build added self-consumption signal at interval level.
        added_selfcons_cols = _detect_added_selfcons_columns(merged.columns)
        if added_selfcons_cols:
            for col in added_selfcons_cols:
                merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)
            merged['added_selfcons_kwh_est'] = merged[added_selfcons_cols].sum(axis=1)
        else:
            merged['added_selfcons_kwh_est'] = 0.0

        # Daily evening features for robust dark-vs-sunny comparison by temperature bin.
        evening_daily = (
            merged.between_time(analysis_start_time, EVENING_END_TIME)
            .resample('D')
            .agg({
                'CONSO_KWH': 'sum',
                'PROD_KWH': 'sum',
                'rad_kwh_per_m2': 'sum',
                'added_selfcons_kwh_est': 'sum',
                't_2m_C': 'mean',
            })
            .dropna()
            .rename(columns={
                'CONSO_KWH': 'conso_evening_kwh',
                'PROD_KWH': 'prod_evening_kwh',
                'rad_kwh_per_m2': 'rad_evening_kwh_per_m2',
                't_2m_C': 'temp_mean_c',
            })
        )
        if evening_daily.empty:
            res["status"] = "REJECTED: No Evening Features"
            return res

        temp_band = temp_buffer if temp_buffer and temp_buffer > 0 else TEMP_MATCH_BUFFER_C
        evening_daily['temp_bin'] = (
            np.round(evening_daily['temp_mean_c'] / temp_band) * temp_band
        ).astype(float)

        dark_daily = evening_daily.loc[evening_daily.index.intersection(pd.DatetimeIndex(dark_days))].copy()
        sunny_daily = evening_daily.loc[evening_daily.index.intersection(pd.DatetimeIndex(sunny_days))].copy()
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
        sunny_matched = sunny_daily.merge(dark_by_bin, left_on="temp_bin", right_index=True, how="inner")
        if len(sunny_matched) < MIN_SUNNY_MATCH_DAYS:
            res["status"] = "REJECTED: No Temp-Matched Sunny"
            return res

        # Profile view for plotting and peak-shift signal on matched days.
        matched_dark_mask = np.isin(merged_dates, pd.DatetimeIndex(dark_daily.index).normalize())
        matched_sunny_mask = np.isin(merged_dates, pd.DatetimeIndex(sunny_matched.index).normalize())
        dark_evening = merged.loc[matched_dark_mask].between_time(analysis_start_time, EVENING_END_TIME)
        sunny_evening_profile = merged.loc[matched_sunny_mask].between_time(analysis_start_time, EVENING_END_TIME)
        prof_dark = dark_evening.groupby(dark_evening.index.time)['CONSO_KWH'].mean()
        prof_sun = sunny_evening_profile.groupby(sunny_evening_profile.index.time)['CONSO_KWH'].mean()
        prof_pair = pd.concat(
            [prof_dark.rename("dark_kwh"), prof_sun.rename("sun_kwh")],
            axis=1
        ).dropna()
        if prof_pair.empty:
            res["status"] = "REJECTED: No Matched Profiles"
            return res

        # Robust anomaly-style features (temperature-bin matched).
        sunny_matched['evening_gap_kwh'] = (
            sunny_matched['dark_evening_baseline_kwh'] - sunny_matched['conso_evening_kwh']
        ).clip(lower=0.0)
        sunny_matched['gap_ratio'] = np.where(
            sunny_matched['dark_evening_baseline_kwh'] > 0.0,
            sunny_matched['evening_gap_kwh'] / sunny_matched['dark_evening_baseline_kwh'],
            0.0,
        )

        pv_capacity_kwp = _pick_pv_capacity_kwp(pv_row)
        pv_prob = _extract_pv_probability(pv_row)
        is_pv_customer = (
            pv_prob >= PV_PROBABILITY_THRESHOLD
        ) if pd.notna(pv_prob) else (pd.notna(pv_capacity_kwp) and pv_capacity_kwp > 0)
        pv_cap_for_potential = pv_capacity_kwp if pd.notna(pv_capacity_kwp) else DEFAULT_PV_CAPACITY_KWP
        sunny_matched['pv_potential_kwh'] = (
            pv_cap_for_potential * sunny_matched['rad_evening_kwh_per_m2'] * PV_POTENTIAL_EFFICIENCY
        ).clip(lower=0.0)
        sunny_matched['available_surplus_kwh'] = (
            sunny_matched['pv_potential_kwh'] - sunny_matched['prod_evening_kwh']
        ).clip(lower=0.0)
        sunny_matched['added_selfcons_kwh_est'] = sunny_matched['added_selfcons_kwh_est'].clip(lower=0.0)

        has_added_signal = float(sunny_matched['added_selfcons_kwh_est'].sum()) > 0.0
        if has_added_signal:
            sunny_matched['inferred_shift_kwh'] = (
                SHIFT_BLEND_GAP_WEIGHT * sunny_matched['evening_gap_kwh'] +
                SHIFT_BLEND_SELFCONS_WEIGHT * sunny_matched['added_selfcons_kwh_est']
            )
        else:
            sunny_matched['inferred_shift_kwh'] = np.minimum(
                sunny_matched['evening_gap_kwh'],
                sunny_matched['available_surplus_kwh'].fillna(np.inf)
            )
        sunny_matched['inferred_shift_kwh'] = sunny_matched['inferred_shift_kwh'].clip(lower=0.0)
        valid_shift = sunny_matched['inferred_shift_kwh'].dropna()
        valid_power_kw = (sunny_matched['inferred_shift_kwh'] / DISCHARGE_INTERVAL_HOURS).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        dark_energy = np.trapezoid(prof_pair["dark_kwh"].values)
        sunny_energy = np.trapezoid(prof_pair["sun_kwh"].values)
        avg_gap_kwh = dark_energy - sunny_energy
        peak_shift = (np.argmax(prof_pair["sun_kwh"].values) - np.argmax(prof_pair["dark_kwh"].values)) * PEAK_SHIFT_STEP_MINUTES
        gap_ratio = float(np.nanmedian(sunny_matched['gap_ratio'])) if len(sunny_matched) else 0.0
        consistency = float(np.mean(sunny_matched['evening_gap_kwh'] > 0.5)) if len(sunny_matched) else 0.0
        valid_potential = sunny_matched['pv_potential_kwh'].replace(0, np.nan)
        shift_ratio = float(np.nanmedian(sunny_matched['inferred_shift_kwh'] / valid_potential)) if len(sunny_matched) else 0.0

        # Injection Logic: Is the solar surplus "missing" during the day?
        pot_high = pv_cap_for_potential * daily.loc[sunny_days, 'rad_kwh_per_m2'].mean() * PV_POTENTIAL_EFFICIENCY
        avg_inj = daily.loc[sunny_days, 'PROD_KWH'].mean()
        inj_ratio = np.clip(
            avg_inj / pot_high if pot_high > INJECTION_RATIO_MIN_POTENTIAL_KWH else 1.0,
            INJECTION_RATIO_CLIP_MIN,
            INJECTION_RATIO_CLIP_MAX,
        )

        # --- THE SIGMOID (Optimized for High Recall, now with robust matched-day terms) ---
        inj_bonus = INJECTION_BONUS_WEIGHT * (INJECTION_RATIO_REFERENCE - inj_ratio)
        z = (
            SIGMOID_INTERCEPT
            + (SIGMOID_PEAK_SHIFT_WEIGHT * peak_shift)
            + (SIGMOID_GAP_RATIO_WEIGHT * gap_ratio)
            + inj_bonus
            + 1.5 * float(np.clip(consistency, 0.0, 1.0))
            + 1.0 * float(np.clip(np.nan_to_num(shift_ratio, nan=0.0), 0.0, 1.0))
        )

        # Robust Sigmoid to prevent Overflow
        prob = 1 / (1 + np.exp(-np.clip(z, SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX)))

        # Broad residential guardrails (2kWh to 30kWh)
        security_scaler = 1.0 if SECURITY_MIN_GAP_KWH <= avg_gap_kwh <= SECURITY_MAX_GAP_KWH else SECURITY_OUTSIDE_SCALER
        phys_scaler = 1.0 if avg_gap_kwh <= (pot_high * PHYSICAL_GAP_FACTOR) else PHYSICAL_OUTSIDE_SCALER

        final_prob = prob * security_scaler * phys_scaler

        max_physical_from_pv_kwh = (
            float(max(CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_PER_KWP * pv_capacity_kwp))
            if pd.notna(pv_capacity_kwp)
            else np.nan
        )
        capacity_capped_by_pv_scaling = False

        # Alternative capacity approach:
        # 1) Use inferred shifted energy on matched sunny days
        # 2) Convert to nominal capacity with discharge-fraction scaling
        # 3) Blend with PV-size anchor when available
        capacity_shift = valid_shift[valid_shift >= MIN_SHIFT_FOR_NOMINAL_CAPACITY_KWH]
        if capacity_shift.empty:
            cap_est = np.nan
            cap_low = np.nan
            cap_high = np.nan
        else:
            nominal_capacity_series = capacity_shift / max(NOMINAL_CAPACITY_DISCHARGE_FRACTION, 0.05)

            if pd.notna(pv_capacity_kwp):
                pv_anchor = pv_capacity_kwp * PV_ANCHOR_KWH_PER_KWP
                nominal_capacity_series = (
                    (1.0 - PV_ANCHOR_BLEND_WEIGHT) * nominal_capacity_series
                    + PV_ANCHOR_BLEND_WEIGHT * pv_anchor
                )

            raw_cap_est = float(nominal_capacity_series.quantile(CAPACITY_ESTIMATE_QUANTILE))
            cap_low = float(nominal_capacity_series.quantile(CAPACITY_LOW_QUANTILE))
            cap_high = float(nominal_capacity_series.quantile(CAPACITY_HIGH_QUANTILE))
            cap_est = raw_cap_est

            if pd.notna(max_physical_from_pv_kwh):
                capacity_capped_by_pv_scaling = raw_cap_est > max_physical_from_pv_kwh

            # Force realistic residential capacity range [5, 30] kWh.
            cap_est = float(np.clip(cap_est, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH))
            cap_low = float(np.clip(cap_low, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH))
            cap_high = float(np.clip(cap_high, CAPACITY_MIN_REALISTIC_KWH, CAPACITY_MAX_REALISTIC_KWH))

        if valid_power_kw.empty:
            power_est = np.nan
            power_low = np.nan
            power_high = np.nan
        else:
            power_est = float(valid_power_kw.quantile(POWER_ESTIMATE_QUANTILE))
            power_low = float(valid_power_kw.quantile(POWER_LOW_QUANTILE))
            power_high = float(valid_power_kw.quantile(POWER_HIGH_QUANTILE))
        battery_detected_by_score = final_prob >= BATTERY_CLASSIFICATION_THRESHOLD
        battery_detected = (
            battery_detected_by_score and is_pv_customer
            if ENFORCE_PV_FOR_BATTERY_DETECTION
            else battery_detected_by_score
        )
        if ENFORCE_PV_FOR_BATTERY_DETECTION and battery_detected_by_score and not is_pv_customer:
            res["status"] = NON_PV_REJECTION_STATUS
        
        res.update({
            "battery_prob": round(final_prob * 100, 2),
            "has_battery": "Yes" if battery_detected else "No",
            "observed_gap_kwh": round(avg_gap_kwh, 3),
            "profiles": (prof_dark, prof_sun),
            "estimated_battery_capacity_kwh": round(cap_est, 3) if pd.notna(cap_est) else np.nan,
            "capacity_ci_lower_kwh": round(cap_low, 3) if pd.notna(cap_low) else np.nan,
            "capacity_ci_upper_kwh": round(cap_high, 3) if pd.notna(cap_high) else np.nan,
            "capacity_capped_by_pv_scaling": bool(capacity_capped_by_pv_scaling),
            "max_physical_from_pv_kwh": round(float(max_physical_from_pv_kwh), 3) if pd.notna(max_physical_from_pv_kwh) else np.nan,
            "estimated_battery_power_kw": round(power_est, 3) if pd.notna(power_est) else np.nan,
            "power_ci_lower_kw": round(power_low, 3) if pd.notna(power_low) else np.nan,
            "power_ci_upper_kw": round(power_high, 3) if pd.notna(power_high) else np.nan,
            "is_pv_customer": bool(is_pv_customer),
            "has_pv_prob": round(float(pv_prob), 4) if pd.notna(pv_prob) else np.nan,
            "pv_capacity_kwp": round(float(pv_capacity_kwp), 3) if pd.notna(pv_capacity_kwp) else np.nan,
        })
        return res

    except Exception as e:
        res["status"] = f"ERROR: {e}"
        return res


## --- 2. Execution Block ---
if __name__ == "__main__":
    freeze_support()
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(RESULTS_PLOT_DIR): os.makedirs(RESULTS_PLOT_DIR, exist_ok=True)

    # Load External Data (Update paths as needed)
    pv_summary = pd.read_csv(PV_SUMMARY_CSV).set_index('customer_id')
    pv_summary.index = pv_summary.index.astype(str)
    eligible_customer_ids = set(pv_summary.index)
    if not eligible_customer_ids:
        raise ValueError(
            f"No customer IDs found in {PV_SUMMARY_CSV}. "
            "At least one matching ID is required for analysis."
        )
    import data.envdata as meteo
    import data.fast_load_smart_meter as re_data
    weather_df = pd.read_csv(WEATHER_CSV, parse_dates=['timestamp'], index_col='timestamp')
    weather_df = _prepare_weather_for_merge(weather_df)

    partner_type = PARTNER_TYPE
    batch_size = BATCH_SIZE
    total_customers_hint = len(eligible_customer_ids)
    print(
        f"Eligible customers from {PV_SUMMARY_CSV}: {total_customers_hint} "
        "(only these IDs will be analyzed)"
    )

    data_gen = re_data.load_all_data_parallel_generator(
        DATA_DIR,
        partner_type=partner_type,
        batch_size=batch_size,
    )

    results = []
    processed = 0
    processed_customers = set()
    batch_counter = 0

    for batch_df in data_gen:
        batch_counter += 1
        batch_df["ID"] = batch_df["ID"].astype(str)
        batch_df = batch_df[batch_df["ID"].isin(eligible_customer_ids)]
        if batch_df.empty:
            continue
        for customer_id, customer_data in batch_df.groupby("ID", sort=False):
            pv_info = pv_summary.loc[customer_id]
            analysis = analyze_battery_residential_v7(customer_data.set_index("DT_UTC"), weather_df, pv_info)
            
            # Save strictly Residential High-Prob Plots
            if analysis["has_battery"] == "Yes" and analysis["profiles"] is not None:
                dark, sun = analysis["profiles"]
                plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), facecolor=PLOT_BACKGROUND_COLOR)
                ax = plt.gca()
                ax.set_facecolor(PLOT_BACKGROUND_COLOR)
                x = [t.hour + t.minute/60 for t in dark.index]
                plt.plot(
                    x,
                    dark.values,
                    linestyle='--',
                    color=PLOT_DARK_DAY_COLOR,
                    alpha=PLOT_DARK_ALPHA,
                    label='Grid consumption (Dark Day baseline)',
                )
                plt.plot(
                    x,
                    sun.values,
                    color=PLOT_SUNNY_DAY_COLOR,
                    label='Grid consumption (Sunny Day)',
                )
                plt.fill_between(
                    x,
                    sun.values,
                    dark.values,
                    color=PLOT_SUNNY_DAY_COLOR,
                    alpha=PLOT_FILL_ALPHA,
                    label='Inferred battery contribution',
                )
                plt.title(
                    f"RESIDENTIAL ID: {customer_id} | Prob: {analysis['battery_prob']}%",
                    color=PLOT_TEXT_COLOR,
                )
                plt.xlabel("Local time (hour)", color=PLOT_TEXT_COLOR)
                plt.ylabel("Energy / interval (kWh)", color=PLOT_TEXT_COLOR)
                ax.tick_params(colors=PLOT_TEXT_COLOR)
                for spine in ax.spines.values():
                    spine.set_color(PLOT_TEXT_COLOR)
                legend = plt.legend(facecolor=PLOT_BACKGROUND_COLOR, edgecolor=PLOT_TEXT_COLOR)
                for txt in legend.get_texts():
                    txt.set_color(PLOT_TEXT_COLOR)
                plt.savefig(f"{PLOT_DIR}/{customer_id}.png")
                plt.savefig(f"{RESULTS_PLOT_DIR}/{customer_id}.png")
                plt.close()

            results.append({"ID": customer_id, **{k: v for k, v in analysis.items() if k != 'profiles'}})
            processed += 1
            processed_customers.add(customer_id)
            if processed % PROGRESS_PRINT_EVERY == 0:
                if total_customers_hint is not None and total_customers_hint > 0:
                    print(
                        f"Progress: {len(processed_customers)}/{total_customers_hint} "
                        f"unique customers ({processed} analyses)"
                    )
                else:
                    print(f"Progress: {len(processed_customers)} unique customers ({processed} analyses)")
        del batch_df
        if batch_counter % GC_EVERY_N_BATCHES == 0:
            gc.collect()

    # Final Output
    final_df = pd.DataFrame(results)
    unique_processed = len(processed_customers)
    print("\n--- FINAL RESIDENTIAL SUMMARY ---")
    print(f"Processed unique customers: {unique_processed}")
    print(f"Total analyses: {len(final_df)} | Found: {(final_df['has_battery']=='Yes').sum()}")
    print(f"Avg Prob: {final_df['battery_prob'].mean():.2f}%")
    final_df.to_csv(RESULTS_CSV, index=False)