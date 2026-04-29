"""Legacy-compatible AC/HP modeled-dataset builders migrated into re_nilm."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

# Shared legacy constants
TEMP_F_THRESHOLD = 55.0
MIN_SAMPLES_CORR = 10
EXPECTED_FREQ = "15min"


def safe_corr(x: pd.Series, y: pd.Series, min_samples: int = MIN_SAMPLES_CORR):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < min_samples:
        return None, len(tmp)
    if tmp["x"].nunique() <= 1 or tmp["y"].nunique() <= 1:
        return None, len(tmp)
    return float(tmp["x"].corr(tmp["y"])), len(tmp)


def safe_mean(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) > 0 else np.nan


def safe_std(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std()) if len(x) > 1 else np.nan


def safe_autocorr(x: pd.Series, lag: int):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= lag + 2:
        return np.nan
    if x.nunique() <= 1:
        return np.nan
    return float(x.autocorr(lag=lag))


def bounded_balance(a: float, b: float, eps: float = 1e-8):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float((a - b) / (abs(a) + abs(b) + eps))


def convert_f_to_c_if_needed(df: pd.DataFrame, threshold_f: float = TEMP_F_THRESHOLD) -> pd.DataFrame:
    max_temp_by_user = df.groupby("id_customer", sort=False)["temp"].max()
    users_to_convert = max_temp_by_user[max_temp_by_user > threshold_f].index.tolist()
    mask = df["id_customer"].isin(users_to_convert) & df["temp"].notna()
    df.loc[mask, "temp"] = (df.loc[mask, "temp"] - 32.0) * 5.0 / 9.0
    return df


def compute_time_quality(df_curve: pd.DataFrame, expected_freq: str = EXPECTED_FREQ):
    if df_curve.empty:
        return {
            "n_rows": 0,
            "winter_rows": 0,
            "summer_rows": 0,
            "max_gap_days": np.nan,
            "coverage_ratio": 0.0,
            "start": pd.NaT,
            "end": pd.NaT,
        }

    d = df_curve.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    d = d.drop_duplicates(subset=["dt_utc"], keep="first")

    if d.empty:
        return {
            "n_rows": 0,
            "winter_rows": 0,
            "summer_rows": 0,
            "max_gap_days": np.nan,
            "coverage_ratio": 0.0,
            "start": pd.NaT,
            "end": pd.NaT,
        }

    months = d["dt_utc"].dt.month
    winter_rows = int(months.isin([12, 1, 2]).sum())
    summer_rows = int(months.isin([6, 7, 8]).sum())

    dt = d["dt_utc"]
    deltas = dt.diff().dropna()
    max_gap_days = float(deltas.max().total_seconds() / (3600 * 24)) if len(deltas) > 0 else 0.0

    full_idx = pd.date_range(start=dt.min(), end=dt.max(), freq=expected_freq)
    expected_n = len(full_idx)
    observed_n = len(dt)
    coverage_ratio = float(observed_n / expected_n) if expected_n > 0 else 0.0

    return {
        "n_rows": observed_n,
        "winter_rows": winter_rows,
        "summer_rows": summer_rows,
        "max_gap_days": max_gap_days,
        "coverage_ratio": coverage_ratio,
        "start": dt.min(),
        "end": dt.max(),
    }


def curve_is_time_usable(
    df_curve: pd.DataFrame,
    curve_name: str,
    min_total_rows: int,
    min_rows_winter: int,
    min_rows_summer: int,
    max_gap_days: float,
    min_coverage_ratio: float,
):
    q = compute_time_quality(df_curve)

    if q["n_rows"] < min_total_rows:
        return False, f"{curve_name}_too_few_rows", q
    if q["winter_rows"] < min_rows_winter:
        return False, f"{curve_name}_too_few_winter_rows", q
    if q["summer_rows"] < min_rows_summer:
        return False, f"{curve_name}_too_few_summer_rows", q
    if pd.notna(q["max_gap_days"]) and q["max_gap_days"] > max_gap_days:
        return False, f"{curve_name}_max_gap_too_large", q
    if q["coverage_ratio"] < min_coverage_ratio:
        return False, f"{curve_name}_coverage_too_low", q
    return True, "ok", q


# AC-specific legacy constants/functions
AC_CORR_THRESHOLD = 0.20
AC_TEMP_HOT = 25.0
AC_TEMP_COLD = 10.0
AC_DAY_RAD_THRESHOLD = 50.0
AC_MIN_DAY_ROWS = 100

AC_MIN_TOTAL_ROWS = 500
AC_MIN_ROWS_SUMMER = 200
AC_MAX_GAP_DAYS = 21
AC_MIN_COVERAGE_RATIO = 0.30

AC_TOT_MIN_TOTAL_ROWS = 500
AC_TOT_MIN_ROWS_SUMMER = 0
AC_TOT_MAX_GAP_DAYS = 90
AC_TOT_MIN_COVERAGE_RATIO = 0.10

AC_LABEL_TO_INT = {"no_ac": 0, "has_ac": 1}


def infer_ac_label(ac_df: pd.DataFrame):
    ac_df = ac_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    if ac_df.empty:
        return None, "empty_ac_curve", None

    usable, reason, quality = curve_is_time_usable(
        ac_df,
        curve_name="ac",
        min_total_rows=AC_MIN_TOTAL_ROWS,
        min_rows_winter=0,
        min_rows_summer=AC_MIN_ROWS_SUMMER,
        max_gap_days=AC_MAX_GAP_DAYS,
        min_coverage_ratio=AC_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return None, reason, quality

    corr_all, _ = safe_corr(ac_df["value_kw_mean"], ac_df["temp"])
    if corr_all is not None and corr_all >= AC_CORR_THRESHOLD:
        return "has_ac", "all_year_corr", quality

    hot_df = ac_df[ac_df["temp"] > AC_TEMP_HOT]
    corr_hot, _ = safe_corr(hot_df["value_kw_mean"], hot_df["temp"])
    if corr_hot is not None and corr_hot >= AC_CORR_THRESHOLD:
        return "has_ac", "hot_check", quality

    summer_df = ac_df[ac_df["dt_utc"].dt.month.isin([6, 7, 8])]
    mean_summer_ac = safe_mean(summer_df["value_kw_mean"])
    if not pd.isna(mean_summer_ac) and mean_summer_ac > 0.05:
        return "has_ac", "summer_mean_positive", quality

    return None, "weak_ac_signal", quality


def ac_tot_curve_is_usable(tot_df: pd.DataFrame):
    tot_df = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    usable, reason, quality = curve_is_time_usable(
        tot_df,
        curve_name="tot",
        min_total_rows=AC_TOT_MIN_TOTAL_ROWS,
        min_rows_winter=0,
        min_rows_summer=AC_TOT_MIN_ROWS_SUMMER,
        max_gap_days=AC_TOT_MAX_GAP_DAYS,
        min_coverage_ratio=AC_TOT_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return False, reason, quality
    if int(tot_df["temp"].notna().sum()) < MIN_SAMPLES_CORR:
        return False, "tot_too_few_temp_rows", quality
    return True, "ok", quality


def extract_ac_tot_features(tot_df: pd.DataFrame):
    d = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    d = d.drop_duplicates(subset=["dt_utc"], keep="first")
    if d.empty:
        return None

    d = d.copy()
    d["month"] = d["dt_utc"].dt.month
    d["hour"] = d["dt_utc"].dt.hour
    d["load"] = pd.to_numeric(d["value_kw_mean"], errors="coerce")
    d["temp_num"] = pd.to_numeric(d["temp"], errors="coerce")
    d["rad_num"] = pd.to_numeric(d["glob_rad"], errors="coerce")

    used_daytime_only = False
    n_day_rows = 0

    day_mask = d["rad_num"].notna() & (d["rad_num"] > AC_DAY_RAD_THRESHOLD)
    d_day = d.loc[day_mask].copy()
    n_day_rows = len(d_day)
    if n_day_rows >= AC_MIN_DAY_ROWS:
        d_feat = d_day
        used_daytime_only = True
    else:
        d_feat = d.copy()

    if d_feat.empty:
        return None

    winter_mask = d_feat["month"].isin([12, 1, 2])
    summer_mask = d_feat["month"].isin([6, 7, 8])
    hot_mask = d_feat["temp_num"] > AC_TEMP_HOT
    cold_mask = d_feat["temp_num"] < AC_TEMP_COLD

    corr_temp_all, _ = safe_corr(d_feat["load"], d_feat["temp_num"])
    corr_temp_hot, _ = safe_corr(d_feat.loc[hot_mask, "load"], d_feat.loc[hot_mask, "temp_num"])
    corr_rad_all, _ = safe_corr(d_feat["load"], d_feat["rad_num"])

    mean_summer = safe_mean(d_feat.loc[summer_mask, "load"])
    mean_winter = safe_mean(d_feat.loc[winter_mask, "load"])
    mean_hot = safe_mean(d_feat.loc[hot_mask, "load"])
    mean_cold = safe_mean(d_feat.loc[cold_mask, "load"])

    season_balance = bounded_balance(mean_summer, mean_winter)
    thermal_balance = bounded_balance(mean_hot, mean_cold)

    total_load = safe_mean(d_feat["load"])
    summer_share = float(mean_summer / (total_load + 1e-8)) if not pd.isna(mean_summer) and not pd.isna(total_load) else np.nan

    day_rows_all = d.loc[d["rad_num"].notna() & (d["rad_num"] > AC_DAY_RAD_THRESHOLD), "load"]
    night_rows_all = d.loc[d["rad_num"].notna() & (d["rad_num"] <= AC_DAY_RAD_THRESHOLD), "load"]
    mean_day_load = safe_mean(day_rows_all)
    mean_night_load = safe_mean(night_rows_all)
    daytime_share = bounded_balance(mean_day_load, mean_night_load)

    load_mean = safe_mean(d_feat["load"])
    load_std = safe_std(d_feat["load"])
    coeff_var = float(load_std / (abs(load_mean) + 1e-8)) if not pd.isna(load_mean) and not pd.isna(load_std) else np.nan

    acf_1h = safe_autocorr(d_feat["load"], lag=4)
    acf_24h = safe_autocorr(d_feat["load"], lag=96)

    afternoon_mask = d_feat["hour"].isin([14, 15, 16, 17, 18])
    mean_afternoon = safe_mean(d_feat.loc[afternoon_mask, "load"])
    afternoon_peak_ratio = float(mean_afternoon / (total_load + 1e-8)) if not pd.isna(mean_afternoon) and not pd.isna(total_load) else np.nan

    summer_d = d_feat.loc[summer_mask].copy()
    if not summer_d.empty and summer_d["load"].notna().any():
        peak_summer_hour = float(summer_d.groupby("hour")["load"].mean().idxmax())
    else:
        peak_summer_hour = np.nan

    spring_mask = d_feat["month"].isin([3, 4, 5])
    mean_spring = safe_mean(d_feat.loc[spring_mask, "load"])
    summer_vs_spring = float(mean_summer / (mean_spring + 1e-8)) if not pd.isna(mean_summer) and not pd.isna(mean_spring) else np.nan
    hot_load_ratio = float(mean_hot / (total_load + 1e-8)) if not pd.isna(mean_hot) and not pd.isna(total_load) else np.nan

    q = compute_time_quality(d)
    return {
        "corr_temp_all": corr_temp_all,
        "corr_temp_hot": corr_temp_hot,
        "corr_rad_all": corr_rad_all,
        "season_balance": season_balance,
        "thermal_balance": thermal_balance,
        "summer_share": summer_share,
        "daytime_share": daytime_share,
        "coeff_var": coeff_var,
        "acf_1h": acf_1h,
        "acf_24h": acf_24h,
        "afternoon_peak_ratio": afternoon_peak_ratio,
        "peak_summer_hour": peak_summer_hour,
        "summer_vs_spring": summer_vs_spring,
        "hot_load_ratio": hot_load_ratio,
        "n_rows": q["n_rows"],
        "summer_rows": q["summer_rows"],
        "winter_rows": q["winter_rows"],
        "max_gap_days": q["max_gap_days"],
        "coverage_ratio": q["coverage_ratio"],
        "used_daytime_only": int(used_daytime_only),
        "n_day_rows": int(n_day_rows),
        "n_feature_rows": int(len(d_feat)),
    }


def build_ac_modeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = df.groupby("id_customer", sort=False)
    for user_id, user_df in grouped:
        ac_df = user_df[user_df["type"] == "AC"]
        tot_df = user_df[user_df["type"] == "TOT"]
        source_values = user_df["source"].dropna().astype(str).unique().tolist()
        source_name = source_values[0] if source_values else "UNKNOWN"

        has_ac = not ac_df.empty
        has_tot = not tot_df.empty

        if has_ac:
            label, _, _ = infer_ac_label(ac_df)
            if label is None or not has_tot:
                continue
            tot_ok, _, _ = ac_tot_curve_is_usable(tot_df)
            if not tot_ok:
                continue
            feats = extract_ac_tot_features(tot_df)
            if feats is None:
                continue
            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = "has_ac"
            feats["target"] = AC_LABEL_TO_INT["has_ac"]
            records.append(feats)
        elif has_tot:
            no_ac_sources = {"dataport", "eco", "refit"}
            if str(source_name).lower() not in no_ac_sources:
                continue
            tot_ok, _, _ = ac_tot_curve_is_usable(tot_df)
            if not tot_ok:
                continue
            feats = extract_ac_tot_features(tot_df)
            if feats is None:
                continue
            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = "no_ac"
            feats["target"] = AC_LABEL_TO_INT["no_ac"]
            records.append(feats)

    return pd.DataFrame(records)


# HP-specific legacy constants/functions
HP_CORR_THRESHOLD = 0.20
HP_TEMP_COLD = 10.0
HP_TEMP_HOT = 28.0
HP_NIGHT_RAD_THRESHOLD = 20.0
HP_MIN_NIGHT_ROWS = 100

HP_MIN_TOTAL_ROWS = 1000
HP_MIN_ROWS_WINTER = 500
HP_MIN_ROWS_SUMMER = 500
HP_MAX_GAP_DAYS = 21
HP_MIN_COVERAGE_RATIO = 0.50

HP_TOT_MIN_TOTAL_ROWS = 500
HP_TOT_MIN_ROWS_WINTER = 0
HP_TOT_MIN_ROWS_SUMMER = 0
HP_TOT_MAX_GAP_DAYS = 90
HP_TOT_MIN_COVERAGE_RATIO = 0.10

HP_LABEL_TO_INT = {"no_hp": 0, "winter_hp": 1, "summer_hp": 2}


def infer_hp_season_label(hp_df: pd.DataFrame):
    hp_df = hp_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    if hp_df.empty:
        return None, "empty_hp_curve", None

    usable, usable_reason, quality = curve_is_time_usable(
        hp_df,
        curve_name="hp",
        min_total_rows=HP_MIN_TOTAL_ROWS,
        min_rows_winter=HP_MIN_ROWS_WINTER,
        min_rows_summer=HP_MIN_ROWS_SUMMER,
        max_gap_days=HP_MAX_GAP_DAYS,
        min_coverage_ratio=HP_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return None, usable_reason, quality

    corr_all, _ = safe_corr(hp_df["value_kw_mean"], hp_df["temp"])
    if corr_all is not None and corr_all <= -HP_CORR_THRESHOLD:
        return "winter_hp", "all_year_corr", quality
    if corr_all is not None and corr_all >= HP_CORR_THRESHOLD:
        return "summer_hp", "all_year_corr", quality

    cold_df = hp_df[hp_df["temp"] < HP_TEMP_COLD]
    hot_df = hp_df[hp_df["temp"] > HP_TEMP_HOT]
    corr_cold, _ = safe_corr(cold_df["value_kw_mean"], cold_df["temp"])
    corr_hot, _ = safe_corr(hot_df["value_kw_mean"], hot_df["temp"])

    winter_ok = corr_cold is not None and corr_cold <= -HP_CORR_THRESHOLD
    summer_ok = corr_hot is not None and corr_hot >= HP_CORR_THRESHOLD

    if winter_ok and summer_ok:
        if abs(corr_cold) >= abs(corr_hot):
            return "winter_hp", "cold_check_stronger_than_hot", quality
        return "summer_hp", "hot_check_stronger_than_cold", quality
    if winter_ok:
        return "winter_hp", "cold_check", quality
    if summer_ok:
        return "summer_hp", "hot_check", quality
    return None, "weak_after_temperature_checks", quality


def hp_tot_curve_is_usable(tot_df: pd.DataFrame):
    tot_df = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    usable, reason, quality = curve_is_time_usable(
        tot_df,
        curve_name="tot",
        min_total_rows=HP_TOT_MIN_TOTAL_ROWS,
        min_rows_winter=HP_TOT_MIN_ROWS_WINTER,
        min_rows_summer=HP_TOT_MIN_ROWS_SUMMER,
        max_gap_days=HP_TOT_MAX_GAP_DAYS,
        min_coverage_ratio=HP_TOT_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return False, reason, quality
    valid_temp_rows = int(tot_df["temp"].notna().sum())
    if valid_temp_rows < MIN_SAMPLES_CORR:
        return False, "tot_too_few_temp_rows", quality
    return True, "ok", quality


def extract_hp_tot_features(tot_df: pd.DataFrame):
    d = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    d = d.drop_duplicates(subset=["dt_utc"], keep="first")
    if d.empty:
        return None

    d = d.copy()
    d["month"] = d["dt_utc"].dt.month
    d["load"] = pd.to_numeric(d["value_kw_mean"], errors="coerce")
    d["temp_num"] = pd.to_numeric(d["temp"], errors="coerce")
    d["rad_num"] = pd.to_numeric(d["glob_rad"], errors="coerce")

    night_mask = d["rad_num"].notna() & (d["rad_num"] <= HP_NIGHT_RAD_THRESHOLD)
    d_night = d.loc[night_mask].copy()
    n_night_rows = len(d_night)
    if len(d_night) < HP_MIN_NIGHT_ROWS:
        return None
    d_feat = d_night

    winter_mask = d_feat["month"].isin([12, 1, 2])
    summer_mask = d_feat["month"].isin([6, 7, 8])
    cold_mask = d_feat["temp_num"] < HP_TEMP_COLD
    hot_mask = d_feat["temp_num"] > HP_TEMP_HOT

    corr_temp_all, _ = safe_corr(d_feat["load"], d_feat["temp_num"])
    corr_temp_cold, _ = safe_corr(d_feat.loc[cold_mask, "load"], d_feat.loc[cold_mask, "temp_num"])
    corr_temp_hot, _ = safe_corr(d_feat.loc[hot_mask, "load"], d_feat.loc[hot_mask, "temp_num"])

    mean_winter = safe_mean(d_feat.loc[winter_mask, "load"])
    mean_summer = safe_mean(d_feat.loc[summer_mask, "load"])
    mean_cold = safe_mean(d_feat.loc[cold_mask, "load"])
    mean_hot = safe_mean(d_feat.loc[hot_mask, "load"])

    season_balance = bounded_balance(mean_summer, mean_winter)
    thermal_balance = bounded_balance(mean_hot, mean_cold)

    load_mean = safe_mean(d_feat["load"])
    load_std = safe_std(d_feat["load"])
    coeff_var = np.nan
    if not pd.isna(load_mean) and not pd.isna(load_std):
        coeff_var = float(load_std / (abs(load_mean) + 1e-8))

    acf_1h = safe_autocorr(d_feat["load"], lag=4)
    acf_24h = safe_autocorr(d_feat["load"], lag=96)

    q = compute_time_quality(d)
    return {
        "corr_temp_all": corr_temp_all,
        "corr_temp_cold": corr_temp_cold,
        "corr_temp_hot": corr_temp_hot,
        "season_balance": season_balance,
        "thermal_balance": thermal_balance,
        "coeff_var": coeff_var,
        "acf_1h": acf_1h,
        "acf_24h": acf_24h,
        "n_rows": q["n_rows"],
        "winter_rows": q["winter_rows"],
        "summer_rows": q["summer_rows"],
        "max_gap_days": q["max_gap_days"],
        "coverage_ratio": q["coverage_ratio"],
        "used_night_only": 1,
        "n_night_rows": int(n_night_rows),
        "n_feature_rows": int(len(d_feat)),
    }


def build_hp_modeled_dataset(df: pd.DataFrame):
    records = []

    hp_reason_counts = Counter()
    tot_reason_counts = Counter()
    skipped_unlabeled_counts = Counter()

    grouped = df.groupby("id_customer", sort=False)
    for user_id, user_df in grouped:
        hp_df = user_df[user_df["type"] == "HP"]
        tot_df = user_df[user_df["type"] == "TOT"]
        source_values = user_df["source"].dropna().astype(str).unique().tolist()
        source_name = source_values[0] if len(source_values) > 0 else "UNKNOWN"

        has_hp = not hp_df.empty
        has_tot = not tot_df.empty

        if has_hp:
            season_label, hp_reason, _ = infer_hp_season_label(hp_df)
            if season_label is None:
                hp_reason_counts[hp_reason] += 1
                continue
            if not has_tot:
                continue
            tot_ok, tot_reason, _ = hp_tot_curve_is_usable(tot_df)
            if not tot_ok:
                tot_reason_counts[tot_reason] += 1
                continue
            feats = extract_hp_tot_features(tot_df)
            if feats is None:
                tot_reason_counts["tot_too_few_night_rows"] += 1
                continue
            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = season_label
            feats["target"] = HP_LABEL_TO_INT[season_label]
            records.append(feats)
            continue

        if (not has_hp) and has_tot:
            tot_ok, tot_reason, _ = hp_tot_curve_is_usable(tot_df)
            if not tot_ok:
                tot_reason_counts[tot_reason] += 1
                continue
            feats = extract_hp_tot_features(tot_df)
            if feats is None:
                tot_reason_counts["tot_too_few_night_rows"] += 1
                continue
            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = "no_hp"
            feats["target"] = HP_LABEL_TO_INT["no_hp"]
            records.append(feats)
            continue

        skipped_unlabeled_counts["users_without_hp_and_without_tot"] += 1

    return pd.DataFrame(records)
