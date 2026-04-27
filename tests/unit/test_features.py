"""Unit tests for re_nilm/features/load.py and weather.py using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from re_nilm.features.load import (
    autocorrelations,
    bounded_balance,
    coeff_var,
    corr_with_temperature,
    daytime_share,
    extract_ac_features,
    extract_hp_features,
    safe_autocorr,
    safe_corr,
    season_balance,
    thermal_balance,
)
from re_nilm.features.weather import (
    compute_daily_radiation,
    compute_temperature_bands,
    cooling_degree_days,
    heating_degree_days,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(n: int = 2 * 365 * 96, seed: int = 0) -> pd.DatetimeIndex:
    """Two years of 15-min timestamps starting 2022-01-01 UTC."""
    return pd.date_range("2022-01-01", periods=n, freq="15min")


# ---------------------------------------------------------------------------
# safe_corr
# ---------------------------------------------------------------------------

def test_safe_corr_perfect():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    r, n = safe_corr(x, x)
    assert r == pytest.approx(1.0)
    assert n == 10


def test_safe_corr_constant_returns_none():
    x = pd.Series([1.0] * 20)
    y = pd.Series(range(20), dtype=float)
    r, n = safe_corr(x, y)
    assert r is None


def test_safe_corr_too_few_samples():
    x = pd.Series([1.0, 2.0])
    y = pd.Series([3.0, 4.0])
    r, n = safe_corr(x, y, min_samples=10)
    assert r is None


# ---------------------------------------------------------------------------
# bounded_balance
# ---------------------------------------------------------------------------

def test_bounded_balance_symmetric():
    assert bounded_balance(1.0, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_bounded_balance_positive():
    result = bounded_balance(2.0, 1.0)
    assert result > 0


def test_bounded_balance_nan_propagation():
    assert np.isnan(bounded_balance(np.nan, 1.0))


# ---------------------------------------------------------------------------
# season_balance
# ---------------------------------------------------------------------------

def test_season_balance_summer_heavy():
    ts = _make_ts()
    # High load in summer months, low in winter
    months = ts.month
    load = pd.Series(np.where(months.isin([6, 7, 8]), 2.0, 0.5))
    sb = season_balance(load, pd.Series(ts))
    assert sb > 0, "Summer-heavy load should give positive season balance"


def test_season_balance_winter_heavy():
    ts = _make_ts()
    months = ts.month
    load = pd.Series(np.where(months.isin([12, 1, 2]), 2.0, 0.5))
    sb = season_balance(load, pd.Series(ts))
    assert sb < 0, "Winter-heavy load should give negative season balance"


# ---------------------------------------------------------------------------
# coeff_var
# ---------------------------------------------------------------------------

def test_coeff_var_constant_series():
    cv = coeff_var(pd.Series([1.0] * 100))
    assert cv == pytest.approx(0.0, abs=1e-3)


def test_coeff_var_positive():
    rng = np.random.default_rng(42)
    cv = coeff_var(pd.Series(rng.uniform(0.5, 2.0, 200)))
    assert cv > 0


# ---------------------------------------------------------------------------
# daytime_share
# ---------------------------------------------------------------------------

def test_daytime_share_all_night():
    n = 200
    load = pd.Series([1.0] * n)
    rad = pd.Series([0.0] * n)
    ds = daytime_share(load, rad)
    assert ds < 0 or np.isnan(ds)


def test_daytime_share_all_day():
    # When all rows are daytime there are no night rows → bounded_balance returns NaN.
    n = 200
    load = pd.Series([1.0] * n)
    rad = pd.Series([500.0] * n)
    ds = daytime_share(load, rad)
    assert np.isnan(ds)


def test_daytime_share_mixed():
    n = 200
    load = pd.Series([2.0] * (n // 2) + [0.5] * (n // 2))
    rad = pd.Series([500.0] * (n // 2) + [0.0] * (n // 2))
    ds = daytime_share(load, rad)
    assert ds > 0


# ---------------------------------------------------------------------------
# autocorrelations
# ---------------------------------------------------------------------------

def test_autocorrelations_periodic():
    # Periodic signal with period 96 steps (24h) should have high acf at lag 96
    t = np.arange(500)
    signal = pd.Series(np.sin(2 * np.pi * t / 96.0))
    acf = autocorrelations(signal, lags=[96])
    assert acf["acf_96steps"] > 0.8


# ---------------------------------------------------------------------------
# extract_hp_features
# ---------------------------------------------------------------------------

def test_extract_hp_features_returns_dict():
    rng = np.random.default_rng(1)
    n = 2 * 365 * 96
    ts = _make_ts(n)
    load = pd.Series(rng.uniform(0.5, 3.0, n))
    temp = pd.Series(rng.uniform(-5, 20, n))
    rad = pd.Series(rng.uniform(0, 50, n))  # all near-night conditions
    rad[:] = 5.0

    feats = extract_hp_features(load, temp, rad, pd.Series(ts))
    assert feats is not None
    assert "corr_temp_all" in feats
    assert "coeff_var" in feats
    assert "acf_1h" in feats
    assert feats["n_night_rows"] > 0


def test_extract_hp_features_too_few_night_rows_returns_none():
    n = 50
    ts = _make_ts(n)
    load = pd.Series(np.ones(n))
    temp = pd.Series(np.ones(n) * 10)
    rad = pd.Series(np.ones(n) * 500)  # always day — no night rows
    result = extract_hp_features(load, temp, rad, pd.Series(ts), min_night_rows=100)
    assert result is None


# ---------------------------------------------------------------------------
# extract_ac_features
# ---------------------------------------------------------------------------

def test_extract_ac_features_returns_all_keys():
    rng = np.random.default_rng(2)
    n = 2 * 365 * 96
    ts = _make_ts(n)
    load = pd.Series(rng.uniform(0.5, 3.0, n))
    temp = pd.Series(rng.uniform(5, 35, n))
    rad = pd.Series(rng.uniform(0, 600, n))

    feats = extract_ac_features(load, temp, rad, pd.Series(ts))
    assert feats is not None
    expected_keys = [
        "corr_temp_all", "corr_temp_hot", "corr_rad_all",
        "season_balance", "thermal_balance", "summer_share",
        "daytime_share", "coeff_var", "acf_1h", "acf_24h",
        "afternoon_peak_ratio", "peak_summer_hour", "summer_vs_spring", "hot_load_ratio",
    ]
    for k in expected_keys:
        assert k in feats, f"Missing feature: {k}"


# ---------------------------------------------------------------------------
# weather features
# ---------------------------------------------------------------------------

def test_compute_daily_radiation_buckets():
    dt = pd.date_range("2022-06-01", periods=96 * 3, freq="15min")
    # Three days with different radiation levels
    rad = np.zeros(len(dt))
    # Day 1: dark (max ~50 W/m²)
    rad[:96] = 50
    # Day 2: cloudy (max ~200)
    rad[96:192] = 200
    # Day 3: sunny (max ~600)
    rad[192:] = 600

    weather = pd.DataFrame({"dt_utc": dt, "global_rad_W": rad})
    daily = compute_daily_radiation(weather)
    assert len(daily) == 3
    buckets = list(daily["rad_bucket"].astype(float))
    assert buckets[0] == 0  # dark
    assert buckets[1] == 1  # cloudy
    assert buckets[2] == 2  # sunny


def test_cdd_hdd():
    temps = pd.Series([15.0, 18.0, 22.0, 25.0])
    cdd = cooling_degree_days(temps, base=18.0)
    hdd = heating_degree_days(temps, base=18.0)
    assert cdd.iloc[0] == 0.0  # 15 < 18
    assert cdd.iloc[3] == pytest.approx(7.0)
    assert hdd.iloc[0] == pytest.approx(3.0)
    assert hdd.iloc[3] == 0.0
