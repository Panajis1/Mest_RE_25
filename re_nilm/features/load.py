"""Shared load-curve feature functions used by all detectors.

These replace the nearly-identical extract_tot_features() implementations that
existed independently in ac_actrainingfunctions.py and hp_detection_functions.py.
The key difference between AC and HP is the row-filtering step:
  - HP: filter to night-only rows (global_rad < night_rad_threshold)
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
    night_mask = rad.notna() & (rad < night_rad_threshold)
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
    min_day_rows: int = 50,
    hot_thresh: float = 28.0,
) -> dict[str, float] | None:
    """Extract AC-detection features matching ac_actrainingfunctions.py:extract_tot_features().

    Uses daytime rows for correlated features; falls back to full curve when
    insufficient daytime data is available.
    """
    rad = pd.to_numeric(global_rad, errors="coerce")
    day_mask = rad.notna() & (rad > day_rad_threshold)
    n_day = int(day_mask.sum())
    used_daytime = n_day >= min_day_rows

    lo = load[day_mask] if used_daytime else load
    te = temp[day_mask] if used_daytime else temp
    ts = timestamps[day_mask] if used_daytime else timestamps

    dt = pd.to_datetime(timestamps)
    months = dt.dt.month
    hours = dt.dt.hour

    corr_feats = corr_with_temperature(lo, te, hot_thresh=hot_thresh)

    # Radiation correlation (AC follows sun)
    corr_rad, _ = safe_corr(lo, rad[day_mask] if used_daytime else rad)
    corr_rad_all = corr_rad if corr_rad is not None else np.nan

    summer_mask = months.isin([6, 7, 8])
    spring_mask = months.isin([3, 4, 5])
    hot_mask_all = temp > hot_thresh

    mean_summer = safe_mean(load[summer_mask])
    mean_total = safe_mean(load)
    mean_spring = safe_mean(load[spring_mask])
    mean_hot = safe_mean(load[hot_mask_all])

    # Summer share: how much of mean annual load falls in summer
    summer_share_val = float(mean_summer / (abs(mean_total) + _EPS)) if not np.isnan(mean_summer) and not np.isnan(mean_total) else np.nan

    # Afternoon peak ratio: 14-18h mean / total mean
    afternoon_mask = hours.isin([14, 15, 16, 17, 18])
    mean_afternoon = safe_mean(load[afternoon_mask])
    afternoon_peak_ratio = float(mean_afternoon / (abs(mean_total) + _EPS)) if not np.isnan(mean_afternoon) and not np.isnan(mean_total) else np.nan

    # Peak summer hour
    summer_load = load[summer_mask]
    summer_hours = hours[summer_mask]
    if not summer_load.empty and summer_load.notna().any():
        tmp = pd.DataFrame({"load": summer_load.values, "hour": summer_hours.values})
        peak_summer_hour = float(tmp.groupby("hour")["load"].mean().idxmax())
    else:
        peak_summer_hour = np.nan

    # Summer vs spring ratio
    summer_vs_spring = float(mean_summer / (mean_spring + _EPS)) if not np.isnan(mean_summer) and not np.isnan(mean_spring) else np.nan

    # Hot load ratio: mean on hot days / total mean
    hot_load_ratio = float(mean_hot / (abs(mean_total) + _EPS)) if not np.isnan(mean_hot) and not np.isnan(mean_total) else np.nan

    feats: dict[str, float] = {
        "corr_temp_all": corr_feats["corr_temp_all"],
        "corr_temp_hot": corr_feats["corr_temp_hot"],
        "corr_rad_all": corr_rad_all,
        "season_balance": season_balance(lo, ts),
        "thermal_balance": thermal_balance(lo, te, hot_thresh=hot_thresh),
        "summer_share": summer_share_val,
        "daytime_share": daytime_share(load, global_rad, day_rad_threshold),
        "coeff_var": coeff_var(lo),
    }
    acf = autocorrelations(lo, lags=[4, 96])
    feats["acf_1h"] = acf["acf_4steps"]
    feats["acf_24h"] = acf["acf_96steps"]
    feats["afternoon_peak_ratio"] = afternoon_peak_ratio
    feats["peak_summer_hour"] = peak_summer_hour
    feats["summer_vs_spring"] = summer_vs_spring
    feats["hot_load_ratio"] = hot_load_ratio
    feats["used_daytime_only"] = int(used_daytime)
    feats["n_day_rows"] = n_day
    return feats
