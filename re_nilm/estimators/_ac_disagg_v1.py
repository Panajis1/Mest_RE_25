"""Legacy-compatible AC disaggregation helpers and model class."""

from __future__ import annotations

import sys as _sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

EPS = 1e-6
AC_ON_THRESHOLD_KW = 0.30
DAY_RAD_THRESHOLD = 50.0
LAGS = [1, 4, 8, 96]
ROLL_WINDOWS = [4, 8, 96]
SUMMER_MONTHS = [6, 7, 8]
RANDOM_STATE = 42


def _compute_user_profile(tot_df: pd.DataFrame) -> dict:
    x = pd.to_numeric(tot_df["tot_kw"], errors="coerce")
    rad = pd.to_numeric(tot_df.get("glob_rad", pd.Series(dtype=float)), errors="coerce")

    q10 = float(x.quantile(0.10)) if len(x) else np.nan
    q50 = float(x.quantile(0.50)) if len(x) else np.nan
    q95 = float(x.quantile(0.95)) if len(x) else np.nan
    mean = float(x.mean()) if len(x) else np.nan
    std = float(x.std()) if len(x) > 1 else np.nan

    day = x[rad.notna() & (rad > DAY_RAD_THRESHOLD)]
    night = x[rad.notna() & (rad <= DAY_RAD_THRESHOLD)]
    day_mean = float(day.mean()) if len(day) else np.nan
    night_mean = float(night.mean()) if len(night) else np.nan

    baseline = q10
    load_factor = mean / (q95 + EPS) if pd.notna(mean) and pd.notna(q95) else np.nan
    spikiness = std / (abs(mean) + EPS) if pd.notna(std) and pd.notna(mean) else np.nan
    baseline_share = baseline / (mean + EPS) if pd.notna(baseline) and pd.notna(mean) else np.nan
    day_night_ratio = day_mean / (night_mean + EPS) if pd.notna(day_mean) and pd.notna(night_mean) else np.nan
    frac_above = float((x > (baseline + AC_ON_THRESHOLD_KW)).mean()) if len(x) else np.nan

    return {
        "user_tot_mean": mean,
        "user_tot_std": std,
        "user_tot_p10": q10,
        "user_tot_p50": q50,
        "user_tot_p95": q95,
        "user_baseline_kw": baseline,
        "user_day_mean": day_mean,
        "user_night_mean": night_mean,
        "user_day_night_ratio": day_night_ratio,
        "user_load_factor": load_factor,
        "user_spikiness": spikiness,
        "user_baseline_share": baseline_share,
        "user_frac_above_baseline": frac_above,
    }


def _add_calendar_and_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["dt_utc"].dt.hour.astype("int16")
    df["minute"] = df["dt_utc"].dt.minute.astype("int16")
    df["quarter_index"] = (df["hour"] * 4 + df["minute"] // 15).astype("int16")
    df["dow"] = df["dt_utc"].dt.dayofweek.astype("int16")
    df["month"] = df["dt_utc"].dt.month.astype("int16")

    df["qidx_sin"] = np.sin(2 * np.pi * df["quarter_index"] / 96.0).astype("float32")
    df["qidx_cos"] = np.cos(2 * np.pi * df["quarter_index"] / 96.0).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")

    df["cdd_28"] = np.maximum(0.0, df["temp"].to_numpy(dtype="float32") - 28.0).astype("float32")
    df["is_day"] = (df["glob_rad"].notna() & (df["glob_rad"] > DAY_RAD_THRESHOLD)).astype("int8")
    df["is_summer"] = df["month"].isin(SUMMER_MONTHS).astype("int8")

    afternoon_mask = ((df["hour"] >= 14) & (df["hour"] < 18)).astype("float32")
    df["cdd28_x_afternoon"] = (df["cdd_28"] * afternoon_mask).astype("float32")

    tot = df["tot_kw"].astype("float32")
    for lag in LAGS:
        df[f"tot_lag_{lag}"] = tot.shift(lag).astype("float32")

    shifted = tot.shift(1)
    for w in ROLL_WINDOWS:
        df[f"tot_rollmean_{w}"] = shifted.rolling(w, min_periods=1).mean().astype("float32")
        df[f"tot_rollstd_{w}"] = shifted.rolling(w, min_periods=2).std().astype("float32")

    df["tot_diff_1"] = tot.diff(1).astype("float32")
    df["tot_diff_4"] = tot.diff(4).astype("float32")
    df["tot_minus_baseline"] = (tot - df["user_baseline_kw"].astype("float32")).astype("float32")
    df["tot_over_mean"] = (tot / (df["user_tot_mean"].astype("float32") + EPS)).astype("float32")
    df["tot_over_p95"] = (tot / (df["user_tot_p95"].astype("float32") + EPS)).astype("float32")
    df["cdd28_x_day"] = (df["cdd_28"] * df["is_day"]).astype("float32")
    return df


class AcTwoStageModel:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_depth=6,
            min_samples_leaf=80, l2_regularization=1.0, random_state=random_state,
        )
        self.reg = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=350, max_depth=8,
            min_samples_leaf=40, l2_regularization=0.5, random_state=random_state,
        )
        self.feature_cols: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], row_weight_col: str = "sample_weight") -> "AcTwoStageModel":
        self.feature_cols = list(feature_cols)
        X = self.imputer.fit_transform(train_df[self.feature_cols])
        y_on = train_df["ac_on"].astype(int).to_numpy()
        w = train_df[row_weight_col].to_numpy(dtype=float) if row_weight_col in train_df.columns else np.ones(len(train_df))

        self.clf.fit(X, y_on, sample_weight=w)

        month = train_df["month"].to_numpy() if "month" in train_df.columns else train_df["dt_utc"].dt.month.to_numpy()
        on_summer_mask = (y_on == 1) & np.isin(month, SUMMER_MONTHS)
        if not np.any(on_summer_mask):
            on_summer_mask = y_on == 1
        if not np.any(on_summer_mask):
            raise ValueError("No AC_on=1 rows in training data.")

        self.reg.fit(
            X[on_summer_mask],
            train_df.loc[on_summer_mask, "ac_kw"].to_numpy(dtype=float),
            sample_weight=w[on_summer_mask],
        )
        return self

    def predict(self, df: pd.DataFrame, apply_month_gating: bool = True) -> pd.DataFrame:
        if self.feature_cols is None:
            raise RuntimeError("Model not fitted.")

        X = self.imputer.transform(df[self.feature_cols])
        p_on = self.clf.predict_proba(X)[:, 1]
        ac_kw_direct = np.clip(self.reg.predict(X), 0.0, None)

        if apply_month_gating:
            month = df["dt_utc"].dt.month.to_numpy()
            is_summer = np.isin(month, SUMMER_MONTHS)
            p_on_adj = np.where(is_summer, p_on, 0.0)
        else:
            p_on_adj = p_on

        on_pred = (p_on_adj >= 0.5).astype(float)
        tot_kw = df["tot_kw"].to_numpy(dtype=float)
        ac_kw_pred = on_pred * ac_kw_direct
        ac_kw_pred = np.minimum(ac_kw_pred, tot_kw)

        ac_ratio_pred = np.where(tot_kw > 0, ac_kw_pred / (tot_kw + EPS), 0.0)
        ac_ratio_pred = np.clip(ac_ratio_pred, 0.0, 1.0)

        out = df.copy()
        out["ac_on_prob"] = p_on_adj.astype("float32")
        out["ac_ratio_pred"] = ac_ratio_pred.astype("float32")
        out["ac_kw_pred"] = ac_kw_pred.astype("float32")
        return out


# Keep old pickle import paths resolvable without retraining artifacts.
_sys.modules.setdefault("model.ac_disaggregation", _sys.modules[__name__])
_sys.modules.setdefault("ac_disaggregation", _sys.modules[__name__])
