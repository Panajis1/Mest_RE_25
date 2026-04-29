"""Legacy-compatible PV capacity helpers migrated from model/pv_detection.py."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _fit_simple_slope(x: np.ndarray, y: np.ndarray):
    """
    Fit a simple linear regression y = alpha + beta x and return (beta, se_beta).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return 0.0, np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0, np.nan
    beta = ((x - x_mean) * (y - y_mean)).sum() / denom
    alpha = y_mean - beta * x_mean
    residuals = y - (alpha + beta * x)
    dof = max(n - 2, 1)
    sigma2 = (residuals ** 2).sum() / dof
    se_beta = float(np.sqrt(sigma2 / denom))
    return float(beta), se_beta


def _compute_demand_radiation_correction(df: pd.DataFrame) -> float:
    """
    Estimate the demand-radiation confound from nighttime demand-temperature
    sensitivity and midday radiation-temperature correlation.

    Returns beta_demand_rad: the portion of the Net_KWH regression slope
    attributable to demand response (positive means demand *decreases* with
    radiation, i.e. correction should be subtracted from -beta_net).
    """
    if "t_2m_C" not in df.columns:
        return 0.0

    night_mask = (df["hour"] >= 22) | (df["hour"] < 5)
    night = df[night_mask].dropna(subset=["t_2m_C", "CONSO_KWH"])
    if night.shape[0] < 30:
        return 0.0

    beta_demand_temp, _ = _fit_simple_slope(
        night["t_2m_C"].to_numpy(),
        night["CONSO_KWH"].to_numpy(),
    )

    midday_mask = (df["hour"] >= 10) & (df["hour"] < 16)
    midday = df[midday_mask].dropna(subset=["t_2m_C", "global_rad_W"])
    if midday.shape[0] < 30:
        return 0.0

    rad = midday["global_rad_W"].to_numpy()
    temp = midday["t_2m_C"].to_numpy()
    if rad.std() < 1e-6 or temp.std() < 1e-6:
        return 0.0

    beta_temp_vs_rad, _ = _fit_simple_slope(rad, temp)

    return float(beta_demand_temp * beta_temp_vs_rad)


def _capacity_and_sc_from_data(
    cust_df: pd.DataFrame, cust_days: pd.DataFrame
) -> dict:
    """
    Compute PV capacity proxy and self-consumption share for a single customer.

    v2 changes:
      - Floor: export-only (no sc_base_kW addition), 95th percentile, STC
        normalized via concurrent irradiance at near-peak export moments.
      - Combiner: regression-primary; floor serves only as a lower-bound clamp.
      - Optional demand-radiation correction from nighttime temperature model.
    """
    _empty = {
        "beta_net": 0.0,
        "beta_net_se": np.nan,
        "beta_export": 0.0,
        "beta_export_se": np.nan,
        "demand_rad_correction": 0.0,
        "pv_capacity_proxy": 0.0,
        "pv_capacity_regression_kwp": 0.0,
        "pv_capacity_floor_kwp": 0.0,
        "pv_capacity_hybrid_kwp": 0.0,
        "sc_share": np.nan,
    }
    if cust_df.empty or cust_days.empty:
        return _empty

    df = cust_df.dropna(subset=["global_rad_W"]).copy()
    if df.empty:
        return _empty

    df["date"] = df["DT_UTC"].dt.date
    df["hour"] = df["DT_UTC"].dt.hour
    if "rad_bucket" in cust_days.columns:
        bucket_map = (
            cust_days[["date", "rad_bucket"]]
            .dropna()
            .drop_duplicates()
            .set_index("date")["rad_bucket"]
        )
        df["rad_bucket"] = df["date"].map(bucket_map)
    else:
        df["rad_bucket"] = np.nan

    df["Net_KWH"] = df["CONSO_KWH"] - df["PROD_KWH"]

    # --- Regression capacity ---
    full_beta_net, full_beta_net_se = _fit_simple_slope(
        df["global_rad_W"].to_numpy(),
        df["Net_KWH"].to_numpy(),
    )

    midday_mask = (df["hour"] >= 10) & (df["hour"] < 16)
    sunny_mask = df["rad_bucket"] == "high"
    reg_df = df[midday_mask & sunny_mask].copy()
    if reg_df.empty:
        reg_beta_net, reg_beta_net_se = full_beta_net, full_beta_net_se
    else:
        reg_beta_net, reg_beta_net_se = _fit_simple_slope(
            reg_df["global_rad_W"].to_numpy(),
            reg_df["Net_KWH"].to_numpy(),
        )

    beta_net = reg_beta_net
    beta_net_se = reg_beta_net_se

    # Export slope vs radiation
    df_export = df[df["PROD_KWH"] > 0]
    if df_export.empty:
        beta_export, beta_export_se = 0.0, np.nan
    else:
        beta_export, beta_export_se = _fit_simple_slope(
            df_export["global_rad_W"].to_numpy(),
            df_export["PROD_KWH"].to_numpy(),
        )

    # Within-customer demand-radiation correction (section 2c of plan)
    demand_rad_correction = _compute_demand_radiation_correction(df)

    raw_pv_slope = max(0.0, -beta_net)
    corrected_pv_slope = max(0.0, raw_pv_slope - demand_rad_correction)
    regression_capacity_kwp = corrected_pv_slope * STC_FACTOR

    # --- Physical floor: export-only, no sc_base_kW (section 2a) ---
    export_kW = df["PROD_KWH"] * 4.0

    export_vals = export_kW.to_numpy()
    export_vals = export_vals[np.isfinite(export_vals)]
    if export_vals.size > 0:
        export_peak_kW = float(np.nanpercentile(export_vals, 95.0))
    else:
        export_peak_kW = 0.0

    # Reference irradiance at near-peak export moments
    if export_peak_kW > 0.0:
        peak_threshold = export_peak_kW * 0.9
        peak_mask = export_kW >= peak_threshold
        if peak_mask.any():
            g_ref_vals = df.loc[peak_mask, "global_rad_W"].to_numpy()
            g_ref_vals = g_ref_vals[np.isfinite(g_ref_vals)]
            if g_ref_vals.size > 0:
                g_ref = float(np.nanmean(g_ref_vals))
            else:
                g_ref = float(df["global_rad_W"].quantile(0.95))
        else:
            g_ref = float(df["global_rad_W"].quantile(0.95))
    else:
        g_ref = np.nan

    if export_peak_kW > 0.0 and np.isfinite(g_ref) and g_ref > 0.0:
        floor_capacity_kwp = float(export_peak_kW * 1000.0 / g_ref)
    else:
        floor_capacity_kwp = 0.0

    # --- Combiner: regression-primary, floor as lower clamp (section 2b) ---
    if regression_capacity_kwp >= floor_capacity_kwp:
        hybrid_capacity_kwp = regression_capacity_kwp
    elif regression_capacity_kwp >= floor_capacity_kwp * 0.7:
        hybrid_capacity_kwp = regression_capacity_kwp
    else:
        hybrid_capacity_kwp = floor_capacity_kwp

    pv_capacity_proxy = hybrid_capacity_kwp / STC_FACTOR

    # Self-consumption share
    days = cust_days.copy()
    hi = days[days["rad_bucket"] == "high"]
    lo = days[days["rad_bucket"] == "low"]

    if hi.empty or lo.empty:
        sc_share = np.nan
    else:
        base_import = lo["Conso_midday"].mean()
        sunny_import = hi["Conso_midday"].mean()
        if pd.isna(base_import) or pd.isna(sunny_import):
            sc_share = np.nan
        else:
            s_imp = max(0.0, base_import - sunny_import)
            exported_midday = hi["Prod_midday"].mean()
            denom = s_imp + max(exported_midday, 0.0)
            if denom <= 0:
                sc_share = np.nan
            else:
                sc_share = float(s_imp / denom)

    return {
        "beta_net": float(full_beta_net),
        "beta_net_se": float(full_beta_net_se),
        "beta_export": beta_export,
        "beta_export_se": beta_export_se,
        "demand_rad_correction": demand_rad_correction,
        "pv_capacity_proxy": pv_capacity_proxy,
        "pv_capacity_regression_kwp": float(regression_capacity_kwp),
        "pv_capacity_floor_kwp": float(floor_capacity_kwp),
        "pv_capacity_hybrid_kwp": float(hybrid_capacity_kwp),
        "sc_share": sc_share,
    }


def _bootstrap_capacity_and_sc(
    cust_df: pd.DataFrame,
    cust_days: pd.DataFrame,
    n_bootstrap: int = 200,
    rng: Optional[np.random.Generator] = None,
    stratify_by_month: bool = True,
    block_size: int = 1,
):
    """
    Block-bootstrap by day to obtain distributions of capacity proxy and
    self-consumption share for a single customer.
    Uses precomputed date->row indices and iloc instead of merge for speed.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_bootstrap <= 0:
        return None, None

    # Precompute date -> row indices (avoids 2 merges per bootstrap iteration)
    date_to_idx_df = cust_df.groupby("date", sort=False).indices
    date_to_idx_days = cust_days.groupby("date", sort=False).indices

    # Restrict to dates present in both 15-min and daily data
    unique_dates = np.array([
        d for d in cust_days["date"].dropna().unique()
        if d in date_to_idx_df and d in date_to_idx_days
    ])
    if unique_dates.size < 3:
        return None, None

    # Optional: month information for stratified sampling (from restricted dates).
    # Pre-sort once so we don't re-sort inside the bootstrap loop.
    unique_dates_sorted = np.array(sorted(unique_dates))
    if stratify_by_month:
        month_df = pd.DataFrame({"date": unique_dates})
        month_df["month"] = pd.to_datetime(month_df["date"]).dt.to_period("M")
        month_to_dates = {
            m: np.array(sorted(grp["date"].values))
            for m, grp in month_df.groupby("month")
        }
    else:
        month_to_dates = None

    def _sample_dates_once() -> np.ndarray:
        """
        Sample a list of dates with replacement, optionally stratified by month
        and optionally in multi-day blocks. Uses pre-sorted month/unique arrays.
        """
        if stratify_by_month and month_to_dates is not None:
            sampled_chunks = []
            for _, dates_m_sorted in month_to_dates.items():
                n_days_m = len(dates_m_sorted)
                if n_days_m == 0:
                    continue

                if block_size <= 1 or n_days_m < block_size:
                    sampled_m = rng.choice(dates_m_sorted, size=n_days_m, replace=True)
                else:
                    if n_days_m <= block_size:
                        sampled_m = rng.choice(dates_m_sorted, size=n_days_m, replace=True)
                    else:
                        blocks = [
                            dates_m_sorted[i: i + block_size]
                            for i in range(0, n_days_m - block_size + 1)
                        ]
                        n_blocks = int(np.ceil(n_days_m / block_size))
                        idx = rng.integers(0, len(blocks), size=n_blocks)
                        sampled_blocks = [blocks[i] for i in idx]
                        sampled_m = np.concatenate(sampled_blocks)
                        if sampled_m.size > n_days_m:
                            sampled_m = sampled_m[:n_days_m]
                sampled_chunks.append(sampled_m)

            if not sampled_chunks:
                return unique_dates
            return np.concatenate(sampled_chunks)
        dates_sorted = unique_dates_sorted
        n_days = len(dates_sorted)
        if block_size <= 1 or n_days < block_size:
            return rng.choice(dates_sorted, size=n_days, replace=True)
        if n_days <= block_size:
            return rng.choice(dates_sorted, size=n_days, replace=True)
        blocks = [
            dates_sorted[i: i + block_size]
            for i in range(0, n_days - block_size + 1)
        ]
        n_blocks = int(np.ceil(n_days / block_size))
        idx = rng.integers(0, len(blocks), size=n_blocks)
        sampled_blocks = [blocks[i] for i in idx]
        sampled = np.concatenate(sampled_blocks)
        if sampled.size > n_days:
            sampled = sampled[:n_days]
        return sampled

    cap_samples: list[float] = []
    sc_samples: list[float] = []

    for _ in range(n_bootstrap):
        sampled_dates = _sample_dates_once()
        # Index-based resampling: duplicate dates => duplicated row indices
        idx_df = np.concatenate([date_to_idx_df[d] for d in sampled_dates])
        idx_days = np.concatenate([date_to_idx_days[d] for d in sampled_dates])
        boot_df = cust_df.iloc[idx_df].reset_index(drop=True)
        boot_days = cust_days.iloc[idx_days].reset_index(drop=True)

        res = _capacity_and_sc_from_data(boot_df, boot_days)
        cap_samples.append(res["pv_capacity_proxy"])
        sc_samples.append(res["sc_share"])

    return np.array(cap_samples), np.array(sc_samples)


# (kWh/15min)/(W/m²) to kWp: ×4 for 15min→power, ×1000 for STC irradiance
STC_FACTOR = 4000.0
