"""Shared load-curve feature functions used by all detectors.

These replace the nearly-identical extract_tot_features() implementations that
existed independently in ac_actrainingfunctions.py and hp_detection_functions.py.
The key difference between AC and HP is the row-filtering step:
  - HP: filter to night-only rows (global_rad <= night_rad_threshold)
  - AC: filter to daytime rows (global_rad > day_rad_threshold)
Both use the same underlying math here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_SAMPLES_CORR = 10
_EPS = 1e-8


# ── Primitive statistics ──────────────────────────────────────────────────────

def safe_corr(x: pd.Series, y: pd.Series, min_samples: int = _MIN_SAMPLES_CORR):
    """Pearson correlation of two series; returns (float, n) or (None, n)."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(tmp)
    if n < min_samples or tmp["x"].nunique() <= 1 or tmp["y"].nunique() <= 1:
        return None, n
    return float(tmp["x"].corr(tmp["y"])), n


def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) > 0 else np.nan


def safe_std(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std()) if len(x) > 1 else np.nan


def safe_autocorr(x: pd.Series, lag: int) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= lag + 2 or x.nunique() <= 1:
        return np.nan
    return float(x.autocorr(lag=lag))


def bounded_balance(a: float, b: float, eps: float = _EPS) -> float:
    """Signed normalised difference: (a-b) / (|a|+|b|+eps). Range (-1, 1)."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float((a - b) / (abs(a) + abs(b) + eps))


# ── Composite features ────────────────────────────────────────────────────────

def corr_with_temperature(
    load: pd.Series,
    temp: pd.Series,
    *,
    hot_thresh: float = 28.0,
    cold_thresh: float = 10.0,
    min_samples: int = _MIN_SAMPLES_CORR,
) -> dict[str, float]:
    """Correlation of load with temperature — overall, hot-only, cold-only.

    Returns dict with keys: corr_temp_all, corr_temp_hot, corr_temp_cold.
    """
    corr_all, _ = safe_corr(load, temp, min_samples)
    hot_mask = temp > hot_thresh
    cold_mask = temp < cold_thresh
    corr_hot, _ = safe_corr(load[hot_mask], temp[hot_mask], min_samples)
    corr_cold, _ = safe_corr(load[cold_mask], temp[cold_mask], min_samples)
    return {
        "corr_temp_all": corr_all if corr_all is not None else np.nan,
        "corr_temp_hot": corr_hot if corr_hot is not None else np.nan,
        "corr_temp_cold": corr_cold if corr_cold is not None else np.nan,
    }


def season_balance(load: pd.Series, timestamps: pd.Series) -> float:
    """Signed normalised difference of mean summer vs mean winter load.

    Positive → more load in summer (AC signal).
    Negative → more load in winter (HP signal).
    """
    months = pd.to_datetime(timestamps).dt.month
    summer = safe_mean(load[months.isin([6, 7, 8])])
    winter = safe_mean(load[months.isin([12, 1, 2])])
    return bounded_balance(summer, winter)


def thermal_balance(
    load: pd.Series,
    temp: pd.Series,
    hot_thresh: float = 28.0,
    cold_thresh: float = 10.0,
) -> float:
    """Mean load on hot days vs cold days — positive for AC, negative for HP."""
    hot = safe_mean(load[temp > hot_thresh])
    cold = safe_mean(load[temp < cold_thresh])
    return bounded_balance(hot, cold)


def daytime_share(
    load: pd.Series,
    global_rad: pd.Series,
    rad_thresh: float = 50.0,
) -> float:
    """Fraction of load falling during daylight (positive → more load during day)."""
    rad = pd.to_numeric(global_rad, errors="coerce")
    day_mask = rad.notna() & (rad > rad_thresh)
    return bounded_balance(safe_mean(load[day_mask]), safe_mean(load[~day_mask]))


def autocorrelations(load: pd.Series, lags: list[int] | None = None) -> dict[str, float]:
    """Autocorrelation at multiple lags. Returns dict of acf_<lag_steps> → float."""
    if lags is None:
        lags = [4, 96]   # 1h and 24h at 15-min resolution
    return {f"acf_{lag}steps": safe_autocorr(load, lag) for lag in lags}


def coeff_var(load: pd.Series) -> float:
    """Coefficient of variation (std/mean) of the load."""
    m = safe_mean(load)
    s = safe_std(load)
    if pd.isna(m) or pd.isna(s):
        return np.nan
    return float(s / (abs(m) + _EPS))


# ── HP feature bundle (night-only) ────────────────────────────────────────────

def extract_hp_features(
    load: pd.Series,
    temp: pd.Series,
    global_rad: pd.Series,
    timestamps: pd.Series,
    night_rad_threshold: float = 20.0,
    min_night_rows: int = 100,
) -> dict[str, float] | None:
    """Extract HP-detection features using night-only rows.

    Mirrors hp_detection_functions.py:extract_tot_features() logic.
    Returns None if there are insufficient night rows.
    """
    rad = pd.to_numeric(global_rad, errors="coerce")
    # Legacy hp_detection_functions uses <= for the night radiation cutoff.
    night_mask = rad.notna() & (rad <= night_rad_threshold)
    n_night = int(night_mask.sum())
    if n_night < min_night_rows:
        return None

    lo = load[night_mask]
    te = temp[night_mask]
    ts = timestamps[night_mask]

    feats = corr_with_temperature(lo, te)
    feats["season_balance"] = season_balance(lo, ts)
    feats["thermal_balance"] = thermal_balance(lo, te)
    feats["coeff_var"] = coeff_var(lo)
    acf = autocorrelations(lo, lags=[4, 96])
    feats["acf_1h"] = acf["acf_4steps"]
    feats["acf_24h"] = acf["acf_96steps"]
    feats["n_night_rows"] = n_night
    return feats


# ── AC feature bundle (daytime-focused) ───────────────────────────────────────

def extract_ac_features(
    load: pd.Series,
    temp: pd.Series,
    global_rad: pd.Series,
    timestamps: pd.Series,
    day_rad_threshold: float = 50.0,
    min_day_rows: int = 100,
    hot_thresh: float = 25.0,   # matches TEMP_HOT=25.0 in ac_actrainingfunctions.py
    cold_thresh: float = 10.0,
) -> dict[str, float] | None:
    """Extract AC-detection features matching ac_actrainingfunctions.py:extract_tot_features().

    All seasonal and time-of-day features are computed on the daytime subset
    (global_rad > day_rad_threshold), falling back to the full curve when fewer
    than min_day_rows are available. This matches the original extract_tot_features
    behaviour where d_feat is the daytime-filtered DataFrame used for everything.

    daytime_share is the exception: it compares day vs night load from the full
    input curve, exactly as the original code does with its full 'd' variable.
    """
    rad = pd.to_numeric(global_rad, errors="coerce")
    day_mask = rad.notna() & (rad > day_rad_threshold)
    n_day = int(day_mask.sum())
    used_daytime = n_day >= min_day_rows

    # d_feat equivalent: daytime rows (or full curve as fallback)
    lo = load[day_mask] if used_daytime else load
    te = temp[day_mask] if used_daytime else temp
    ra = rad[day_mask] if used_daytime else rad
    ts = timestamps[day_mask] if used_daytime else timestamps

    dt_lo = pd.to_datetime(ts)
    months_lo = dt_lo.dt.month
    hours_lo = dt_lo.dt.hour

    summer_mask = months_lo.isin([6, 7, 8])
    winter_mask = months_lo.isin([12, 1, 2])
    spring_mask = months_lo.isin([3, 4, 5])
    hot_mask = te > hot_thresh
    cold_mask = te < cold_thresh

    corr_temp_all, _ = safe_corr(lo, te)
    corr_temp_hot, _ = safe_corr(lo[hot_mask], te[hot_mask])
    corr_rad_all, _ = safe_corr(lo, ra)

    mean_summer = safe_mean(lo[summer_mask])
    mean_winter = safe_mean(lo[winter_mask])
    mean_spring = safe_mean(lo[spring_mask])
    mean_hot = safe_mean(lo[hot_mask])
    mean_cold = safe_mean(lo[cold_mask])
    total_load = safe_mean(lo)

    season_balance_val = bounded_balance(mean_summer, mean_winter)
    thermal_balance_val = bounded_balance(mean_hot, mean_cold)

    summer_share_val = float(mean_summer / (abs(total_load) + _EPS)) if not np.isnan(mean_summer) and not np.isnan(total_load) else np.nan
    summer_vs_spring_val = float(mean_summer / (mean_spring + _EPS)) if not np.isnan(mean_summer) and not np.isnan(mean_spring) else np.nan
    hot_load_ratio_val = float(mean_hot / (abs(total_load) + _EPS)) if not np.isnan(mean_hot) and not np.isnan(total_load) else np.nan

    # daytime_share uses the full curve (day vs night split), not the daytime subset
    day_rows_all = load[rad.notna() & (rad > day_rad_threshold)]
    night_rows_all = load[rad.notna() & (rad <= day_rad_threshold)]
    daytime_share_val = bounded_balance(safe_mean(day_rows_all), safe_mean(night_rows_all))

    coeff_var_val = coeff_var(lo)
    acf = autocorrelations(lo, lags=[4, 96])

    afternoon_mask = hours_lo.isin([14, 15, 16, 17, 18])
    mean_afternoon = safe_mean(lo[afternoon_mask])
    afternoon_peak_ratio_val = float(mean_afternoon / (abs(total_load) + _EPS)) if not np.isnan(mean_afternoon) and not np.isnan(total_load) else np.nan

    summer_load = lo[summer_mask]
    summer_hours = hours_lo[summer_mask]
    if not summer_load.empty and summer_load.notna().any():
        tmp = pd.DataFrame({"load": summer_load.values, "hour": summer_hours.values})
        peak_summer_hour_val = float(tmp.groupby("hour")["load"].mean().idxmax())
    else:
        peak_summer_hour_val = np.nan

    return {
        "corr_temp_all": corr_temp_all if corr_temp_all is not None else np.nan,
        "corr_temp_hot": corr_temp_hot if corr_temp_hot is not None else np.nan,
        "corr_rad_all": corr_rad_all if corr_rad_all is not None else np.nan,
        "season_balance": season_balance_val,
        "thermal_balance": thermal_balance_val,
        "summer_share": summer_share_val,
        "daytime_share": daytime_share_val,
        "coeff_var": coeff_var_val,
        "acf_1h": acf["acf_4steps"],
        "acf_24h": acf["acf_96steps"],
        "afternoon_peak_ratio": afternoon_peak_ratio_val,
        "peak_summer_hour": peak_summer_hour_val,
        "summer_vs_spring": summer_vs_spring_val,
        "hot_load_ratio": hot_load_ratio_val,
        "used_daytime_only": int(used_daytime),
        "n_day_rows": n_day,
    }
