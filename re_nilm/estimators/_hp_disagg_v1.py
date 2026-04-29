"""Legacy-compatible HP disaggregation helpers and model class."""

from __future__ import annotations

from dataclasses import dataclass
import sys as _sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
NIGHT_RAD_THRESHOLD = 20.0
HP_ON_THRESHOLD_KW = 0.20
EPS = 1e-6
LAGS = [1, 4, 8, 96]
ROLL_WINDOWS = [4, 8, 96]


@dataclass
class _CorrAccumulator:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return
        xv = np.asarray(x[mask], dtype=float)
        yv = np.asarray(y[mask], dtype=float)
        self.n += int(len(xv))
        self.sum_x += float(xv.sum())
        self.sum_y += float(yv.sum())
        self.sum_x2 += float(np.square(xv).sum())
        self.sum_y2 += float(np.square(yv).sum())
        self.sum_xy += float((xv * yv).sum())

    def corr(self) -> float:
        if self.n < 2:
            return np.nan
        num = self.n * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        den = np.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        if den <= 0:
            return np.nan
        return float(num / den)


def _compute_user_profile(tot_df: pd.DataFrame) -> dict[str, float]:
    x = pd.to_numeric(tot_df["tot_kw"], errors="coerce")
    rad = pd.to_numeric(tot_df["glob_rad"], errors="coerce")

    q05 = float(x.quantile(0.05)) if len(x) else np.nan
    q10 = float(x.quantile(0.10)) if len(x) else np.nan
    q50 = float(x.quantile(0.50)) if len(x) else np.nan
    q95 = float(x.quantile(0.95)) if len(x) else np.nan
    mean = float(x.mean()) if len(x) else np.nan
    std = float(x.std()) if len(x) > 1 else np.nan

    night = x[rad.notna() & (rad <= NIGHT_RAD_THRESHOLD)]
    day = x[rad.notna() & (rad > NIGHT_RAD_THRESHOLD)]
    night_mean = float(night.mean()) if len(night) else np.nan
    day_mean = float(day.mean()) if len(day) else np.nan

    baseline = q10
    load_factor = mean / (q95 + EPS) if pd.notna(mean) and pd.notna(q95) else np.nan
    spikiness = std / (abs(mean) + EPS) if pd.notna(std) and pd.notna(mean) else np.nan
    baseline_share = baseline / (mean + EPS) if pd.notna(baseline) and pd.notna(mean) else np.nan
    day_night_ratio = day_mean / (night_mean + EPS) if pd.notna(day_mean) and pd.notna(night_mean) else np.nan
    frac_above_baseline = float((x > (baseline + HP_ON_THRESHOLD_KW)).mean()) if len(x) else np.nan

    lf_n = np.clip(load_factor if pd.notna(load_factor) else 0.0, 0.0, 1.0)
    fab_n = np.clip(frac_above_baseline if pd.notna(frac_above_baseline) else 0.0, 0.0, 1.0)
    spiky_n = np.clip((spikiness if pd.notna(spikiness) else 0.0) / 2.5, 0.0, 1.0)
    dnr = day_night_ratio if pd.notna(day_night_ratio) else 1.0
    dnr_dev = np.clip(abs(dnr - 1.0), 0.0, 2.0) / 2.0

    scientific_like_score = 0.50 * lf_n + 0.30 * fab_n - 0.15 * spiky_n - 0.05 * dnr_dev
    scientific_like_score = float(np.clip(scientific_like_score, 0.0, 1.0))

    if scientific_like_score >= 0.60:
        profile_class = "scientific_like"
    elif scientific_like_score <= 0.35:
        profile_class = "dataport_like"
    else:
        profile_class = "uncertain"

    return {
        "user_tot_mean": mean,
        "user_tot_std": std,
        "user_tot_p05": q05,
        "user_tot_p50": q50,
        "user_tot_p95": q95,
        "user_baseline_kw": baseline,
        "user_night_mean": night_mean,
        "user_day_mean": day_mean,
        "user_day_night_ratio": day_night_ratio,
        "user_load_factor": load_factor,
        "user_spikiness": spikiness,
        "user_frac_above_baseline": frac_above_baseline,
        "user_baseline_share": baseline_share,
        "scientific_like_score": scientific_like_score,
        "profile_class": profile_class,
    }


def _add_calendar_and_dynamic_features(merged: pd.DataFrame) -> pd.DataFrame:
    merged["hour"] = merged["dt_utc"].dt.hour.astype("int16")
    merged["minute"] = merged["dt_utc"].dt.minute.astype("int16")
    merged["quarter_index"] = (merged["hour"] * 4 + (merged["minute"] // 15)).astype("int16")
    merged["dow"] = merged["dt_utc"].dt.dayofweek.astype("int16")
    merged["month"] = merged["dt_utc"].dt.month.astype("int16")

    merged["qidx_sin"] = np.sin(2 * np.pi * merged["quarter_index"] / 96.0).astype("float32")
    merged["qidx_cos"] = np.cos(2 * np.pi * merged["quarter_index"] / 96.0).astype("float32")
    merged["dow_sin"] = np.sin(2 * np.pi * merged["dow"] / 7.0).astype("float32")
    merged["dow_cos"] = np.cos(2 * np.pi * merged["dow"] / 7.0).astype("float32")
    merged["month_sin"] = np.sin(2 * np.pi * (merged["month"] - 1) / 12.0).astype("float32")
    merged["month_cos"] = np.cos(2 * np.pi * (merged["month"] - 1) / 12.0).astype("float32")

    merged["hdd_18"] = np.maximum(0.0, 18.0 - merged["temp"].to_numpy(dtype="float32")).astype("float32")
    merged["is_night"] = (merged["glob_rad"].notna() & (merged["glob_rad"] <= NIGHT_RAD_THRESHOLD)).astype("int8")

    tot = merged["tot_kw"].astype("float32")
    for lag in LAGS:
        merged[f"tot_lag_{lag}"] = tot.shift(lag).astype("float32")

    shifted = tot.shift(1)
    for w in ROLL_WINDOWS:
        merged[f"tot_rollmean_{w}"] = shifted.rolling(window=w, min_periods=1).mean().astype("float32")
        merged[f"tot_rollstd_{w}"] = shifted.rolling(window=w, min_periods=2).std().astype("float32")

    merged["tot_diff_1"] = tot.diff(1).astype("float32")
    merged["tot_diff_4"] = tot.diff(4).astype("float32")
    merged["tot_minus_baseline"] = (tot - merged["user_baseline_kw"].astype("float32")).astype("float32")
    merged["tot_over_mean"] = (tot / (merged["user_tot_mean"].astype("float32") + EPS)).astype("float32")
    merged["tot_over_p95"] = (tot / (merged["user_tot_p95"].astype("float32") + EPS)).astype("float32")
    merged["hdd_x_night"] = (merged["hdd_18"] * merged["is_night"]).astype("float32")
    return merged


class ScientificTwoStageHPModel:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_depth=6,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self.reg = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=350,
            max_depth=8,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self.feature_cols: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], row_weight_col: str = "sample_weight") -> "ScientificTwoStageHPModel":
        self.feature_cols = list(feature_cols)
        X = self.imputer.fit_transform(train_df[self.feature_cols])

        y_on = train_df["hp_on"].astype(int).to_numpy()
        w = train_df[row_weight_col].to_numpy(dtype=float) if row_weight_col in train_df.columns else np.ones(len(train_df))
        self.clf.fit(X, y_on, sample_weight=w)

        on_mask = y_on == 1
        if not np.any(on_mask):
            raise ValueError("No positive hp_on rows available for stage-2 regression.")

        self.reg.fit(X[on_mask], train_df.loc[on_mask, "hp_ratio"].to_numpy(dtype=float), sample_weight=w[on_mask])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols is None:
            raise RuntimeError("Model has not been fitted.")

        X = self.imputer.transform(df[self.feature_cols])
        p_on = self.clf.predict_proba(X)[:, 1]
        ratio_on = np.clip(self.reg.predict(X), 0.0, 1.0)

        score = np.clip(df["scientific_like_score"].to_numpy(dtype=float), 0.0, 1.0)
        p_on_adj = np.power(np.clip(p_on, 0.0, 1.0), 0.95)
        ratio_factor = 0.70 + 0.55 * score
        ratio_on_adj = ratio_on * ratio_factor
        ratio_on_adj = ratio_on_adj * 4.0

        ratio_cap = np.clip(4.5 * score + 0.12, 0.12, 0.95)
        ratio_on_adj = np.clip(ratio_on_adj, 0.0, ratio_cap)
        hp_ratio_pred = np.clip(p_on_adj * ratio_on_adj, 0.0, ratio_cap)
        hp_kw_pred = np.clip(hp_ratio_pred * df["tot_kw"].to_numpy(dtype=float), 0.0, None)

        out = df.copy()
        out["hp_on_prob"] = p_on_adj
        out["hp_ratio_on_pred"] = ratio_on_adj
        out["hp_ratio_pred"] = hp_ratio_pred
        out["hp_kw_pred"] = hp_kw_pred
        return out


# Keep old pickle import paths resolvable without retraining artifacts.
_sys.modules.setdefault("hp_model.disaggregation_functions", _sys.modules[__name__])
_sys.modules.setdefault("disaggregation_functions", _sys.modules[__name__])
