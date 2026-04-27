
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import gc
import os
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from hp_detection_functions import (
    TEMP_F_THRESHOLD,
    convert_f_to_c_if_needed,
    infer_hp_season_label,
    tot_curve_is_usable,
)

INPUT_FILE = "all_sources_load_with_weather.parquet"
RANDOM_STATE = 42
NIGHT_RAD_THRESHOLD = 20.0
PV_FRAC_THRESHOLD = 0.03
PV_POWER_THRESHOLD = -0.1
USE_QUALITY_FILTERS = True
HP_ON_THRESHOLD_KW = 0.20
EPS = 1e-6

LAGS = [1, 4, 8, 96]
ROLL_WINDOWS = [4, 8, 96]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_memory(tag: str = "") -> None:
    if psutil is None:
        log(f"[MEM] {tag} | psutil not available")
        return
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    log(f"[MEM] {tag} | RAM: {mem_mb:.2f} MB")



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


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_users: list[str]
    test_users: list[str]
    split_reports: dict[str, pd.DataFrame]
    split_info: dict[str, object]


def has_strong_pv_signal(
    tot_df: pd.DataFrame,
    frac_threshold: float = PV_FRAC_THRESHOLD,
    power_threshold: float = PV_POWER_THRESHOLD,
) -> bool:
    vals = pd.to_numeric(tot_df["value_kw_mean"], errors="coerce").dropna()
    if len(vals) == 0:
        return False
    return float((vals < power_threshold).mean()) >= frac_threshold


def load_raw_df(input_file: str = INPUT_FILE) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / input_file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    log(f"[LOAD] Reading internal parquet: {file_path}")
    needed_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df = pd.read_parquet(file_path, columns=needed_cols)
    df = df.loc[df["type"].isin(["TOT", "HP"])].copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"]).copy()
    df["id_customer"] = df["id_customer"].astype("string")
    df["source"] = df["source"].astype("string")
    df["value_kw_mean"] = pd.to_numeric(df["value_kw_mean"], errors="coerce").astype("float32")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce").astype("float32")
    df["glob_rad"] = pd.to_numeric(df["glob_rad"], errors="coerce").astype("float32")
    df = df.dropna(subset=["id_customer", "value_kw_mean"]).copy()
    df = convert_f_to_c_if_needed(df, threshold_f=TEMP_F_THRESHOLD)
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce").astype("float32")
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)
    log(f"[LOAD] Internal rows kept: {len(df):,} | users: {df['id_customer'].nunique():,}")
    return df


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

    # Soft "scientific-like" score, kept continuous and not too aggressive.
    lf_n = np.clip(load_factor if pd.notna(load_factor) else 0.0, 0.0, 1.0)
    fab_n = np.clip(frac_above_baseline if pd.notna(frac_above_baseline) else 0.0, 0.0, 1.0)
    spiky_n = np.clip((spikiness if pd.notna(spikiness) else 0.0) / 2.5, 0.0, 1.0)
    base_n = np.clip(baseline_share if pd.notna(baseline_share) else 0.0, 0.0, 1.2) / 1.2
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
    # Caller must pass data already sorted by dt_utc.
    # Avoid extra reset/copy here to reduce memory pressure in external inference.
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


def build_aligned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    total_users = df["id_customer"].nunique()
    log(f"[BUILD] Building aligned dataset for {total_users:,} users...")

    for i, (user_id, user_df) in enumerate(df.groupby("id_customer", sort=False), start=1):
        if i == 1 or i % 25 == 0 or i == total_users:
            log(f"[BUILD] Processed users: {i:,}/{total_users:,}")
        tot_df = user_df[user_df["type"] == "TOT"].copy()
        hp_df = user_df[user_df["type"] == "HP"].copy()
        if tot_df.empty or hp_df.empty:
            continue
        if has_strong_pv_signal(tot_df):
            continue

        source_values = sorted(user_df["source"].dropna().astype(str).unique().tolist())
        source_name = source_values[0] if source_values else "UNKNOWN"

        if USE_QUALITY_FILTERS:
            tot_ok, _, _ = tot_curve_is_usable(tot_df)
            hp_label, _, _ = infer_hp_season_label(hp_df)
            if (not tot_ok) or (hp_label is None):
                continue
        else:
            hp_label = "hp_unknown_season"

        if hp_label != "winter_hp":
            continue

        tot_df = (
            tot_df[["id_customer", "dt_utc", "value_kw_mean", "temp", "glob_rad"]]
            .drop_duplicates(subset=["dt_utc"], keep="first")
            .rename(columns={"value_kw_mean": "tot_kw"})
            .sort_values("dt_utc")
        )

        hp_df = (
            hp_df[["id_customer", "dt_utc", "value_kw_mean"]]
            .drop_duplicates(subset=["dt_utc"], keep="first")
            .rename(columns={"value_kw_mean": "hp_kw"})
            .sort_values("dt_utc")
        )

        merged = tot_df.merge(hp_df, on=["id_customer", "dt_utc"], how="inner")
        if merged.empty:
            continue

        merged = merged.loc[merged["tot_kw"].abs() > EPS].copy().reset_index(drop=True)
        if merged.empty:
            continue

        merged["source_name"] = source_name
        merged["hp_season_label"] = hp_label

        profile = _compute_user_profile(merged[["tot_kw", "glob_rad"]].copy())
        for k, v in profile.items():
            merged[k] = v

        merged["hp_ratio"] = (merged["hp_kw"] / (merged["tot_kw"] + EPS)).clip(0.0, 1.0)
        merged["hp_on"] = (merged["hp_kw"] >= HP_ON_THRESHOLD_KW).astype(int)
        merged = _add_calendar_and_dynamic_features(merged)
        records.append(merged)

    if not records:
        raise ValueError("No aligned TOT-HP users available after filtering.")

    out = pd.concat(records, axis=0, ignore_index=True)
    log(f"[BUILD] Final aligned rows: {len(out):,} | users: {out['id_customer'].nunique():,}")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "id_customer",
        "dt_utc",
        "source_name",
        "hp_season_label",
        "profile_class",
        "hp_kw",
        "hp_ratio",
        "hp_on",
    }
    return [c for c in df.columns if c not in excluded]


def build_user_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df[["id_customer", "source_name", "hp_season_label"]]
        .drop_duplicates()
        .sort_values(["id_customer"])
        .reset_index(drop=True)
    )
    return meta


def summarize_sources_and_labels(split_df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_level = split_df[["id_customer", "source_name", "hp_season_label"]].drop_duplicates()
    source_users = user_level["source_name"].value_counts(dropna=False).rename_axis("source_name").reset_index(name="n_users")
    source_rows = split_df["source_name"].value_counts(dropna=False).rename_axis("source_name").reset_index(name="n_rows")
    label_users = user_level["hp_season_label"].value_counts(dropna=False).rename_axis("hp_season_label").reset_index(name="n_users")
    label_rows = split_df["hp_season_label"].value_counts(dropna=False).rename_axis("hp_season_label").reset_index(name="n_rows")
    log(f"\n[{split_name}] Sources by users\n{source_users.to_string(index=False)}")
    log(f"\n[{split_name}] Sources by rows\n{source_rows.to_string(index=False)}")
    return source_users, source_rows, label_users, label_rows


def split_random_by_user(aligned_df: pd.DataFrame, test_size: float = 0.20) -> SplitResult:
    user_meta = build_user_metadata(aligned_df)
    train_meta, test_meta = train_test_split(
        user_meta,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_users = sorted(train_meta["id_customer"].astype(str).tolist())
    test_users = sorted(test_meta["id_customer"].astype(str).tolist())

    train_df = aligned_df[aligned_df["id_customer"].isin(train_users)].copy()
    test_df = aligned_df[aligned_df["id_customer"].isin(test_users)].copy()

    train_source_users, train_source_rows, train_label_users, train_label_rows = summarize_sources_and_labels(train_df, "TRAIN")
    test_source_users, test_source_rows, test_label_users, test_label_rows = summarize_sources_and_labels(test_df, "TEST")

    split_reports = {
        "train_source_users": train_source_users,
        "train_source_rows": train_source_rows,
        "train_label_users": train_label_users,
        "train_label_rows": train_label_rows,
        "test_source_users": test_source_users,
        "test_source_rows": test_source_rows,
        "test_label_users": test_label_users,
        "test_label_rows": test_label_rows,
    }
    split_info = {
        "split_type": "random_user",
        "n_total_users": aligned_df["id_customer"].nunique(),
        "n_train_users": len(train_users),
        "n_test_users": len(test_users),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
    }
    return SplitResult(train_df, test_df, train_users, test_users, split_reports, split_info)


def add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    user_counts = out["id_customer"].value_counts()
    user_weight = out["id_customer"].map(lambda u: 1.0 / float(user_counts[u]))
    out["sample_weight"] = user_weight / user_weight.mean()
    return out


class ScientificTwoStageHPModel:
    """
    Train only on scientific-like winter HP users.
    Stage 1: P(HP_on | X)
    Stage 2: hp_ratio | HP_on
    Final prediction uses a *mild* profile-aware cap/factor to avoid dataport explosion
    without crushing scientific-like behavior.
    """

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

        # Global scale calibration for SCIENTIFIC-only training.
        # The model already learns the ON/OFF pattern well, but tends to under-predict
        # the HP ratio magnitude. We therefore correct only the final ratio scale here.
        p_on_adj = np.power(np.clip(p_on, 0.0, 1.0), 0.95)
        ratio_factor = 0.70 + 0.55 * score      # 0.70 .. 1.25
        ratio_on_adj = ratio_on * ratio_factor

        SCALE_ALPHA = 4.0
        ratio_on_adj = ratio_on_adj * SCALE_ALPHA

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


def energy_kwh_from_kw_15min(power_kw: np.ndarray) -> float:
    return float(np.nansum(power_kw) * 0.25)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(np.std(y_true), 0.0):
        return np.nan
    return float(r2_score(y_true, y_pred))


def _safe_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(np.std(y_true), 0.0):
        return np.nan
    return float(explained_variance_score(y_true, y_pred))


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    mbe_kw = float(np.mean(y_pred - y_true))
    mean_true = float(np.mean(y_true))
    energy_true_kwh = energy_kwh_from_kw_15min(y_true)
    energy_pred_kwh = energy_kwh_from_kw_15min(y_pred)
    signed_energy_error_kwh = energy_pred_kwh - energy_true_kwh
    abs_energy_error_kwh = abs(signed_energy_error_kwh)
    return {
        "n": int(len(y_true)),
        "mae_kw": float(mae),
        "rmse_kw": float(rmse),
        "medae_kw": float(medae),
        "r2": _safe_r2(y_true, y_pred),
        "explained_variance": _safe_explained_variance(y_true, y_pred),
        "mbe_kw": mbe_kw,
        "nmbe": float(mbe_kw / mean_true) if mean_true > 0 else np.nan,
        "nmae": float(mae / mean_true) if mean_true > 0 else np.nan,
        "nrmse": float(rmse / mean_true) if mean_true > 0 else np.nan,
        "smape": _smape(y_true, y_pred),
        "mean_true_kw": mean_true,
        "mean_pred_kw": float(np.mean(y_pred)),
        "std_true_kw": float(np.std(y_true)),
        "std_pred_kw": float(np.std(y_pred)),
        "p50_true_kw": float(np.median(y_true)),
        "p50_pred_kw": float(np.median(y_pred)),
        "max_true_kw": float(np.max(y_true)) if len(y_true) else np.nan,
        "max_pred_kw": float(np.max(y_pred)) if len(y_pred) else np.nan,
        "true_energy_kwh": energy_true_kwh,
        "pred_energy_kwh": energy_pred_kwh,
        "signed_energy_error_kwh": signed_energy_error_kwh,
        "abs_energy_error_kwh": abs_energy_error_kwh,
        "rel_energy_error": float(abs_energy_error_kwh / energy_true_kwh) if energy_true_kwh > 0 else np.nan,
    }


def compute_ratio_metrics_dict(y_true_ratio: np.ndarray, y_pred_ratio: np.ndarray) -> dict[str, float]:
    y_true_ratio = np.asarray(y_true_ratio, dtype=float)
    y_pred_ratio = np.asarray(y_pred_ratio, dtype=float)
    mae = mean_absolute_error(y_true_ratio, y_pred_ratio)
    rmse = np.sqrt(mean_squared_error(y_true_ratio, y_pred_ratio))
    return {
        "n": int(len(y_true_ratio)),
        "mae_ratio": float(mae),
        "rmse_ratio": float(rmse),
        "medae_ratio": float(median_absolute_error(y_true_ratio, y_pred_ratio)),
        "r2_ratio": _safe_r2(y_true_ratio, y_pred_ratio),
        "explained_variance_ratio": _safe_explained_variance(y_true_ratio, y_pred_ratio),
        "mbe_ratio": float(np.mean(y_pred_ratio - y_true_ratio)),
        "smape_ratio": _smape(y_true_ratio, y_pred_ratio),
        "mean_true_ratio": float(np.mean(y_true_ratio)),
        "mean_pred_ratio": float(np.mean(y_pred_ratio)),
        "p50_true_ratio": float(np.median(y_true_ratio)),
        "p50_pred_ratio": float(np.median(y_pred_ratio)),
        "max_true_ratio": float(np.max(y_true_ratio)) if len(y_true_ratio) else np.nan,
        "max_pred_ratio": float(np.max(y_pred_ratio)) if len(y_pred_ratio) else np.nan,
    }


def compute_on_metrics_dict(y_true_on: np.ndarray, y_prob_on: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true_on = np.asarray(y_true_on, dtype=int)
    y_prob_on = np.asarray(y_prob_on, dtype=float)
    y_pred_on = (y_prob_on >= threshold).astype(int)
    out = {
        "n": int(len(y_true_on)),
        "threshold": float(threshold),
        "mean_true_on": float(np.mean(y_true_on)),
        "mean_pred_prob_on": float(np.mean(y_prob_on)),
        "mean_pred_on_at_threshold": float(np.mean(y_pred_on)),
        "precision_on": float(precision_score(y_true_on, y_pred_on, zero_division=0)),
        "recall_on": float(recall_score(y_true_on, y_pred_on, zero_division=0)),
        "f1_on": float(f1_score(y_true_on, y_pred_on, zero_division=0)),
    }
    if len(np.unique(y_true_on)) > 1:
        out["roc_auc_on"] = float(roc_auc_score(y_true_on, y_prob_on))
    else:
        out["roc_auc_on"] = np.nan
    return out


def evaluate_predictions(test_pred_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kw_metrics = compute_metrics_dict(test_pred_df["hp_kw"].to_numpy(), test_pred_df["hp_kw_pred"].to_numpy())
    ratio_metrics = compute_ratio_metrics_dict(test_pred_df["hp_ratio"].to_numpy(), test_pred_df["hp_ratio_pred"].to_numpy())
    on_metrics = compute_on_metrics_dict(test_pred_df["hp_on"].to_numpy(), test_pred_df["hp_on_prob"].to_numpy())

    by_user_rows = []
    for user_id, g in test_pred_df.groupby("id_customer", sort=False):
        row = {"id_customer": user_id, "n_rows_test": len(g), "source_name": g["source_name"].iloc[0]}
        row.update(compute_metrics_dict(g["hp_kw"].to_numpy(), g["hp_kw_pred"].to_numpy()))
        row.update(compute_ratio_metrics_dict(g["hp_ratio"].to_numpy(), g["hp_ratio_pred"].to_numpy()))
        row.update(compute_on_metrics_dict(g["hp_on"].to_numpy(), g["hp_on_prob"].to_numpy()))
        by_user_rows.append(row)
    by_user_df = pd.DataFrame(by_user_rows).sort_values(["rmse_kw", "id_customer"]).reset_index(drop=True)

    by_source_rows = []
    for source_name, g in test_pred_df.groupby("source_name", sort=False):
        row = {"source_name": source_name, "n_rows_test": len(g), "n_users_test": g["id_customer"].nunique()}
        row.update(compute_metrics_dict(g["hp_kw"].to_numpy(), g["hp_kw_pred"].to_numpy()))
        row.update(compute_ratio_metrics_dict(g["hp_ratio"].to_numpy(), g["hp_ratio_pred"].to_numpy()))
        row.update(compute_on_metrics_dict(g["hp_on"].to_numpy(), g["hp_on_prob"].to_numpy()))
        by_source_rows.append(row)
    by_source_df = pd.DataFrame(by_source_rows).reset_index(drop=True)

    by_hp_rows = []
    for hp_label, g in test_pred_df.groupby("hp_season_label", sort=False):
        row = {"hp_season_label": hp_label, "n_rows_test": len(g), "n_users_test": g["id_customer"].nunique()}
        row.update(compute_metrics_dict(g["hp_kw"].to_numpy(), g["hp_kw_pred"].to_numpy()))
        row.update(compute_ratio_metrics_dict(g["hp_ratio"].to_numpy(), g["hp_ratio_pred"].to_numpy()))
        row.update(compute_on_metrics_dict(g["hp_on"].to_numpy(), g["hp_on_prob"].to_numpy()))
        by_hp_rows.append(row)
    by_hp_season_df = pd.DataFrame(by_hp_rows).reset_index(drop=True)

    return kw_metrics, ratio_metrics, on_metrics, by_user_df, by_source_df, by_hp_season_df


def _print_metric_block(title: str, metrics: dict[str, float], ordered_keys: list[str]) -> None:
    log(f"\n[{title}]")
    for key in ordered_keys:
        val = metrics.get(key, np.nan)
        if isinstance(val, (float, np.floating)):
            log(f"{key:26s}: {float(val):.6f}")
        else:
            log(f"{key:26s}: {val}")


def print_kw_metrics(metrics: dict[str, float], title: str) -> None:
    ordered = [
        "n", "mae_kw", "rmse_kw", "medae_kw", "r2", "explained_variance",
        "mbe_kw", "nmbe", "nmae", "nrmse", "smape",
        "mean_true_kw", "mean_pred_kw", "std_true_kw", "std_pred_kw",
        "p50_true_kw", "p50_pred_kw", "max_true_kw", "max_pred_kw",
        "true_energy_kwh", "pred_energy_kwh", "signed_energy_error_kwh",
        "abs_energy_error_kwh", "rel_energy_error",
    ]
    _print_metric_block(title, metrics, ordered)


def print_ratio_metrics(metrics: dict[str, float], title: str) -> None:
    ordered = [
        "n", "mae_ratio", "rmse_ratio", "medae_ratio", "r2_ratio",
        "explained_variance_ratio", "mbe_ratio", "smape_ratio",
        "mean_true_ratio", "mean_pred_ratio", "p50_true_ratio", "p50_pred_ratio",
        "max_true_ratio", "max_pred_ratio",
    ]
    _print_metric_block(title, metrics, ordered)


def print_on_metrics(metrics: dict[str, float], title: str) -> None:
    ordered = [
        "n", "threshold", "mean_true_on", "mean_pred_prob_on",
        "mean_pred_on_at_threshold", "roc_auc_on", "precision_on", "recall_on", "f1_on",
    ]
    _print_metric_block(title, metrics, ordered)


def print_source_summary(by_source_df: pd.DataFrame) -> None:
    if by_source_df.empty:
        return
    cols = [
        "source_name", "n_users_test", "n_rows_test", "mae_kw", "rmse_kw",
        "r2", "mbe_kw", "rel_energy_error", "mae_ratio", "r2_ratio", "roc_auc_on",
    ]
    log("\n[TEST BY SOURCE]")
    log(by_source_df[cols].to_string(index=False))


def print_user_summary(by_user_df: pd.DataFrame) -> None:
    if by_user_df.empty:
        return
    cols = [
        "mae_kw", "rmse_kw", "r2", "mbe_kw", "rel_energy_error",
        "mae_ratio", "rmse_ratio", "r2_ratio", "mbe_ratio",
    ]
    summary = by_user_df[cols].describe(percentiles=[0.25, 0.5, 0.75]).transpose()[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    log("\n[TEST BY USER - summary]")
    log(summary.to_string())


def print_evaluation_report(
    kw_metrics: dict[str, float],
    ratio_metrics: dict[str, float],
    on_metrics: dict[str, float],
    by_source_df: pd.DataFrame,
    by_user_df: pd.DataFrame,
) -> None:
    print_kw_metrics(kw_metrics, "TEST - GLOBAL kW")
    print_ratio_metrics(ratio_metrics, "TEST - GLOBAL ratio")
    print_on_metrics(on_metrics, "TEST - GLOBAL ON")
    print_source_summary(by_source_df)
    print_user_summary(by_user_df)


def save_outputs(
    prefix: str,
    split_info: dict[str, object],
    split_reports: dict[str, pd.DataFrame],
    test_pred_df: pd.DataFrame,
    by_user_df: pd.DataFrame,
    by_source_df: pd.DataFrame,
    by_hp_season_df: pd.DataFrame,
    kw_metrics: dict[str, float],
    ratio_metrics: dict[str, float],
    on_metrics: dict[str, float],
) -> None:
    base_dir = Path(__file__).resolve().parent
    metrics_row: dict[str, object] = {}
    metrics_row.update(split_info)
    metrics_row.update(kw_metrics)
    metrics_row.update(ratio_metrics)
    metrics_row.update(on_metrics)
    pd.DataFrame([metrics_row]).to_csv(base_dir / f"{prefix}_metrics.csv", index=False)
    test_pred_df.to_parquet(base_dir / f"{prefix}_test_predictions.parquet", index=False)
    by_user_df.to_csv(base_dir / f"{prefix}_test_by_user.csv", index=False)
    by_source_df.to_csv(base_dir / f"{prefix}_test_by_source.csv", index=False)
    by_hp_season_df.to_csv(base_dir / f"{prefix}_test_by_hp_season.csv", index=False)
    for key, value in split_reports.items():
        value.to_csv(base_dir / f"{prefix}_{key}.csv", index=False)


# ---------- external data helpers ----------

def _find_first_existing(columns: Iterable[str], candidates: list[str]) -> str:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def load_winter_hp_ids(label_dir: Path) -> set[str]:
    csv_paths = sorted(label_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {label_dir}")

    ids: set[str] = set()
    for path in csv_paths:
        df = pd.read_csv(path)
        id_col = _find_first_existing(df.columns, ["id_customer", "customer_id", "id", "customer"])
        label_col = _find_first_existing(df.columns, ["label", "target_name", "pred_target_name"])
        tmp = df[[id_col, label_col]].copy()
        tmp[id_col] = tmp[id_col].astype(str)
        tmp[label_col] = tmp[label_col].astype(str).str.strip().str.lower()
        ids.update(tmp.loc[tmp[label_col] == "winter_hp", id_col].unique().tolist())
    return ids


def load_external_weather_15min(dt_min=None, dt_max=None) -> pd.DataFrame:
    from envdata import env_data

    log("[WEATHER] Loading weather from envdata.py ...")
    _, avg_df = env_data()

    weather = avg_df.copy()
    if "timestamp" not in weather.columns:
        weather = weather.reset_index()
        if "timestamp" not in weather.columns:
            if "index" in weather.columns:
                weather = weather.rename(columns={"index": "timestamp"})
            else:
                weather = weather.rename(columns={weather.columns[0]: "timestamp"})

    log(f"[WEATHER] Raw columns after index reset/rename: {list(weather.columns)}")

    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
    weather = weather.dropna(subset=["timestamp"]).copy()

    rename_map = {}
    if "t_2m_C" in weather.columns:
        rename_map["t_2m_C"] = "temp"
    if "global_rad_W" in weather.columns:
        rename_map["global_rad_W"] = "glob_rad"
    weather = weather.rename(columns=rename_map)

    needed = ["timestamp", "temp", "glob_rad"]
    missing = [c for c in needed if c not in weather.columns]
    if missing:
        raise ValueError(
            f"Weather data missing required columns: {missing}. "
            f"Available columns: {list(weather.columns)}"
        )

    weather = weather[needed].copy()
    weather = weather.rename(columns={"timestamp": "dt_utc"})
    weather["temp"] = pd.to_numeric(weather["temp"], errors="coerce").astype("float32")
    weather["glob_rad"] = pd.to_numeric(weather["glob_rad"], errors="coerce").astype("float32")

    if dt_min is not None:
        dt_min = pd.to_datetime(dt_min, utc=True)
        weather = weather[weather["dt_utc"] >= dt_min]
    if dt_max is not None:
        dt_max = pd.to_datetime(dt_max, utc=True)
        weather = weather[weather["dt_utc"] <= dt_max]

    weather_15 = (
        weather.set_index("dt_utc")[["temp", "glob_rad"]]
        .sort_index()
        .resample("15min")
        .interpolate(method="time")
        .reset_index()
    )

    log(
        f"[WEATHER] Weather 15min ready: rows={len(weather_15)}, "
        f"range={weather_15['dt_utc'].min()} -> {weather_15['dt_utc'].max()}"
    )
    return weather_15


def _build_external_feature_table_for_parquet(
    parquet_path: Path,
    winter_ids: set[str],
    weather_15: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    id_col = _find_first_existing(df.columns, ["customer_id", "id_customer", "id"])
    dt_col = _find_first_existing(df.columns, ["DT_UTC", "dt_utc", "timestamp"])
    load_col = _find_first_existing(
        df.columns,
        ["net_consumption_kwh_15min", "CONSO_KWH", "consumption_kwh_15min"],
    )

    tmp = df[[id_col, dt_col, load_col]].copy()
    del df

    tmp = tmp.rename(columns={id_col: "id_customer", dt_col: "dt_utc", load_col: "tot_kw"})
    tmp["id_customer"] = tmp["id_customer"].astype(str)
    tmp = tmp[tmp["id_customer"].isin(winter_ids)]
    if tmp.empty:
        log(f"[EXTERNAL]   no winter_hp users in {parquet_path.name}, skipping.")
        return pd.DataFrame()

    tmp["dt_utc"] = pd.to_datetime(tmp["dt_utc"], utc=True, errors="coerce")
    tmp["tot_kw"] = pd.to_numeric(tmp["tot_kw"], errors="coerce").astype("float32")
    tmp = tmp.dropna(subset=["dt_utc", "tot_kw"])
    if tmp.empty:
        log(f"[EXTERNAL]   all rows invalid after datetime/numeric parsing, skipping.")
        return pd.DataFrame()

    tmp = tmp.sort_values(["id_customer", "dt_utc"]).reset_index(drop=True)
    tmp = tmp.merge(weather_15, on="dt_utc", how="left", sort=False)

    tmp = tmp[["id_customer", "dt_utc", "tot_kw", "temp", "glob_rad"]]
    tmp["temp"] = pd.to_numeric(tmp["temp"], errors="coerce").astype("float32")
    tmp["glob_rad"] = pd.to_numeric(tmp["glob_rad"], errors="coerce").astype("float32")

    records: list[pd.DataFrame] = []
    for j, (user_id, g0) in enumerate(tmp.groupby("id_customer", sort=False), start=1):
        if j == 1 or j % 100 == 0:
            log(f"[EXTERNAL]   users processed in {parquet_path.name}: {j}")

        if len(g0) < 96 * 5:
            continue

        g = pd.DataFrame({
            "id_customer": pd.Series([str(user_id)] * len(g0), dtype="string"),
            "dt_utc": g0["dt_utc"].to_numpy(copy=False),
            "tot_kw": g0["tot_kw"].to_numpy(dtype="float32", copy=False),
            "temp": g0["temp"].to_numpy(dtype="float32", copy=False),
            "glob_rad": g0["glob_rad"].to_numpy(dtype="float32", copy=False),
        })

        g["source_name"] = "external_pvforecast"
        g["hp_season_label"] = "unknown_external"

        profile = _compute_user_profile(g[["tot_kw", "glob_rad"]])
        for k, v in profile.items():
            if isinstance(v, (float, int, np.floating, np.integer)) and pd.notna(v):
                g[k] = np.float32(v)
            else:
                g[k] = v

        g = _add_calendar_and_dynamic_features(g)

        for col in g.select_dtypes(include=["float64"]).columns:
            g[col] = g[col].astype("float32")

        records.append(g)

    del tmp

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, axis=0, ignore_index=True)
    log(
        f"[EXTERNAL] Completed feature build for {parquet_path.name}: "
        f"kept_users={out['id_customer'].nunique()}, kept_rows={len(out)}"
    )
    return out


def build_external_feature_table(
    pv_forecast_dir: Path,
    winter_ids: set[str],
    weather_15: pd.DataFrame,
) -> pd.DataFrame:
    parquet_paths = sorted(pv_forecast_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {pv_forecast_dir}")

    records: list[pd.DataFrame] = []
    total_rows_kept = 0
    total_users_kept = 0

    weather_15 = weather_15.sort_values("dt_utc").reset_index(drop=True)
    weather_15["temp"] = pd.to_numeric(weather_15["temp"], errors="coerce").astype("float32")
    weather_15["glob_rad"] = pd.to_numeric(weather_15["glob_rad"], errors="coerce").astype("float32")

    for i, path in enumerate(parquet_paths, start=1):
        log(f"[EXTERNAL] Reading parquet {i}/{len(parquet_paths)}: {path.name}")
        part_df = _build_external_feature_table_for_parquet(path, winter_ids, weather_15)
        if part_df.empty:
            continue
        records.append(part_df)
        total_rows_kept += len(part_df)
        total_users_kept += part_df["id_customer"].nunique()
        log(
            f"[EXTERNAL] Cumulative totals: users={total_users_kept}, rows={total_rows_kept}"
        )

    if not records:
        raise ValueError("No eligible external users found after filtering.")

    log(f"[EXTERNAL] Concatenating {len(records)} user chunks...")
    out = pd.concat(records, axis=0, ignore_index=True)
    log(f"[EXTERNAL] Final external feature table: rows={len(out)}, users={out['id_customer'].nunique()}")
    return out


def stream_external_predictions_to_parquet(
    model: ScientificTwoStageHPModel,
    pv_forecast_dir: Path,
    winter_ids: set[str],
    weather_15: pd.DataFrame,
    output_dir: Path,
) -> Path:
    parquet_paths = sorted(pv_forecast_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {pv_forecast_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_parquet in output_dir.glob("*.parquet"):
        old_parquet.unlink()

    weather_15 = weather_15.sort_values("dt_utc").reset_index(drop=True)
    weather_15["temp"] = pd.to_numeric(weather_15["temp"], errors="coerce").astype("float32")
    weather_15["glob_rad"] = pd.to_numeric(weather_15["glob_rad"], errors="coerce").astype("float32")

    kept_files = 0
    total_rows = 0
    total_users = 0

    for i, path in enumerate(parquet_paths, start=1):
        log(f"[STREAM] Processing parquet {i}/{len(parquet_paths)}: {path.name}")
        external_df = _build_external_feature_table_for_parquet(path, winter_ids, weather_15)
        if external_df.empty:
            gc.collect()
            continue

        pred_df = model.predict(external_df)
        chunk_path = output_dir / f"predictions_part_{i:04d}.parquet"
        pred_df.to_parquet(chunk_path, index=False)

        kept_files += 1
        total_rows += len(pred_df)
        total_users += pred_df["id_customer"].nunique()
        log(
            f"[STREAM] Saved {chunk_path.name}: rows={len(pred_df)}, "
            f"users={pred_df['id_customer'].nunique()}, cumulative_rows={total_rows}"
        )

        del external_df
        del pred_df
        gc.collect()

    if kept_files == 0:
        raise ValueError("No eligible external users found after filtering.")

    log(
        f"[STREAM] Completed streaming predictions: chunk_files={kept_files}, "
        f"cumulative_users={total_users}, cumulative_rows={total_rows}"
    )
    return output_dir

def summarize_external_prediction_chunks(predictions_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunk_paths = sorted(predictions_dir.glob("*.parquet"))
    if not chunk_paths:
        raise FileNotFoundError(f"No parquet prediction chunks found in {predictions_dir}")

    total_rows = 0
    all_users: set[str] = set()
    sum_tot_kw = 0.0
    sum_hp_kw_pred = 0.0
    sum_hp_ratio_pred = 0.0
    sum_hp_on_prob = 0.0
    corr_temp = _CorrAccumulator()
    corr_hdd = _CorrAccumulator()
    user_profiles: dict[str, tuple[str, float]] = {}
    monthly_acc: dict[str, dict[str, object]] = {}

    usecols = [
        "id_customer", "dt_utc", "tot_kw", "hp_kw_pred", "hp_ratio_pred", "hp_on_prob",
        "temp", "hdd_18", "profile_class", "scientific_like_score",
    ]

    for path in chunk_paths:
        log(f"[SUMMARY] Reading chunk: {path.name}")
        df = pd.read_parquet(path, columns=usecols)
        if df.empty:
            continue

        total_rows += len(df)
        all_users.update(df["id_customer"].astype(str).unique().tolist())
        sum_tot_kw += float(pd.to_numeric(df["tot_kw"], errors="coerce").fillna(0).sum())
        sum_hp_kw_pred += float(pd.to_numeric(df["hp_kw_pred"], errors="coerce").fillna(0).sum())
        sum_hp_ratio_pred += float(pd.to_numeric(df["hp_ratio_pred"], errors="coerce").fillna(0).sum())
        sum_hp_on_prob += float(pd.to_numeric(df["hp_on_prob"], errors="coerce").fillna(0).sum())

        corr_temp.update(
            pd.to_numeric(df["hp_kw_pred"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["temp"], errors="coerce").to_numpy(dtype=float),
        )
        corr_hdd.update(
            pd.to_numeric(df["hp_kw_pred"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["hdd_18"], errors="coerce").to_numpy(dtype=float),
        )

        profile_df = df[["id_customer", "profile_class", "scientific_like_score"]].drop_duplicates()
        for row in profile_df.itertuples(index=False):
            user_profiles[str(row.id_customer)] = (str(row.profile_class), float(row.scientific_like_score))

        df["month"] = (
            pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .dt.to_period("M")
            .astype(str)
        )

        month_grouped = (
            df.groupby("month", dropna=False)
            .agg(
                n_rows=("tot_kw", "size"),
                mean_tot_kw_sum=("tot_kw", "sum"),
                mean_hp_kw_pred_sum=("hp_kw_pred", "sum"),
                mean_hp_ratio_pred_sum=("hp_ratio_pred", "sum"),
                mean_hp_on_prob_sum=("hp_on_prob", "sum"),
                mean_temp_sum=("temp", "sum"),
            )
            .reset_index()
        )

        for row in month_grouped.itertuples(index=False):
            slot = monthly_acc.setdefault(str(row.month), {
                "n_rows": 0,
                "sum_tot_kw": 0.0,
                "sum_hp_kw_pred": 0.0,
                "sum_hp_ratio_pred": 0.0,
                "sum_hp_on_prob": 0.0,
                "sum_temp": 0.0,
                "users": set(),
            })
            slot["n_rows"] += int(row.n_rows)
            slot["sum_tot_kw"] += float(row.mean_tot_kw_sum)
            slot["sum_hp_kw_pred"] += float(row.mean_hp_kw_pred_sum)
            slot["sum_hp_ratio_pred"] += float(row.mean_hp_ratio_pred_sum)
            slot["sum_hp_on_prob"] += float(row.mean_hp_on_prob_sum)
            slot["sum_temp"] += float(row.mean_temp_sum)

        for month, ids in df.groupby("month", dropna=False)["id_customer"]:
            monthly_acc[str(month)]["users"].update(ids.astype(str).unique().tolist())

        del df
        gc.collect()

    if total_rows == 0:
        raise ValueError(f"All parquet prediction chunks were empty in {predictions_dir}")

    global_summary = pd.DataFrame([{
        "n_rows": total_rows,
        "n_users": len(all_users),
        "mean_tot_kw": sum_tot_kw / total_rows,
        "mean_hp_kw_pred": sum_hp_kw_pred / total_rows,
        "mean_hp_ratio_pred": sum_hp_ratio_pred / total_rows,
        "mean_hp_on_prob": sum_hp_on_prob / total_rows,
        "corr_hpkw_temp": corr_temp.corr(),
        "corr_hpkw_hdd18": corr_hdd.corr(),
    }])

    by_profile_rows = []
    profile_buckets: dict[str, list[float]] = {}
    for _, (profile_class, score) in user_profiles.items():
        profile_buckets.setdefault(profile_class, []).append(score)
    for profile_class, scores in profile_buckets.items():
        by_profile_rows.append({
            "profile_class": profile_class,
            "n_users": len(scores),
            "mean_score": float(np.mean(scores)) if scores else np.nan,
        })
    by_profile = pd.DataFrame(by_profile_rows).sort_values("n_users", ascending=False).reset_index(drop=True)

    monthly_rows = []
    for month, acc in sorted(monthly_acc.items()):
        n_rows = int(acc["n_rows"])
        monthly_rows.append({
            "month": month,
            "n_rows": n_rows,
            "n_users": len(acc["users"]),
            "mean_tot_kw": acc["sum_tot_kw"] / n_rows,
            "mean_hp_kw_pred": acc["sum_hp_kw_pred"] / n_rows,
            "mean_hp_ratio_pred": acc["sum_hp_ratio_pred"] / n_rows,
            "mean_hp_on_prob": acc["sum_hp_on_prob"] / n_rows,
            "mean_temp": acc["sum_temp"] / n_rows,
        })
    monthly = pd.DataFrame(monthly_rows)

    return global_summary, by_profile, monthly


def combine_prediction_chunks_to_parquet(predictions_dir: Path, output_parquet_path: Path) -> Path:
    chunk_paths = sorted(predictions_dir.glob("*.parquet"))
    if not chunk_paths:
        raise FileNotFoundError(f"No parquet prediction chunks found in {predictions_dir}")

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet_path.exists():
        output_parquet_path.unlink()

    frames = []
    for path in chunk_paths:
        frames.append(pd.read_parquet(path))
    out = pd.concat(frames, axis=0, ignore_index=True)
    out.to_parquet(output_parquet_path, index=False)

    del frames
    del out
    gc.collect()
    return output_parquet_path


def save_external_outputs_from_chunks(
    prefix: str,
    predictions_dir: Path,
    global_summary: pd.DataFrame,
    by_profile: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    base_dir = Path(__file__).resolve().parent
    final_predictions_parquet = combine_prediction_chunks_to_parquet(
        predictions_dir=predictions_dir,
        output_parquet_path=base_dir / f"{prefix}_predictions.parquet",
    )
    log(f"[SAVE] Final predictions parquet written to: {final_predictions_parquet}")
    global_summary.to_csv(base_dir / f"{prefix}_sanity_global.csv", index=False)
    by_profile.to_csv(base_dir / f"{prefix}_sanity_by_profile.csv", index=False)
    monthly.to_csv(base_dir / f"{prefix}_sanity_monthly.csv", index=False)


def run_external_sanity_checks(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp = pred_df.copy()
    tmp["month"] = (
        pd.to_datetime(tmp["dt_utc"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )

    global_summary = pd.DataFrame([{
        "n_rows": len(tmp),
        "n_users": tmp["id_customer"].nunique(),
        "mean_tot_kw": float(tmp["tot_kw"].mean()),
        "mean_hp_kw_pred": float(tmp["hp_kw_pred"].mean()),
        "mean_hp_ratio_pred": float(tmp["hp_ratio_pred"].mean()),
        "mean_hp_on_prob": float(tmp["hp_on_prob"].mean()),
        "corr_hpkw_temp": float(tmp[["hp_kw_pred", "temp"]].corr().iloc[0, 1]) if tmp["temp"].notna().sum() > 2 else np.nan,
        "corr_hpkw_hdd18": float(tmp[["hp_kw_pred", "hdd_18"]].corr().iloc[0, 1]) if tmp["hdd_18"].notna().sum() > 2 else np.nan,
    }])

    by_profile = (
        tmp[["id_customer", "profile_class", "scientific_like_score"]]
        .drop_duplicates()
        .groupby("profile_class")
        .agg(
            n_users=("id_customer", "nunique"),
            mean_score=("scientific_like_score", "mean"),
        )
        .reset_index()
        .sort_values("n_users", ascending=False)
        .reset_index(drop=True)
    )

    monthly = (
        tmp.groupby("month")
        .agg(
            n_rows=("tot_kw", "size"),
            n_users=("id_customer", "nunique"),
            mean_tot_kw=("tot_kw", "mean"),
            mean_hp_kw_pred=("hp_kw_pred", "mean"),
            mean_hp_ratio_pred=("hp_ratio_pred", "mean"),
            mean_hp_on_prob=("hp_on_prob", "mean"),
            mean_temp=("temp", "mean"),
        )
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )

    return global_summary, by_profile, monthly


def save_external_outputs(prefix: str, pred_df: pd.DataFrame, global_summary: pd.DataFrame, by_profile: pd.DataFrame, monthly: pd.DataFrame) -> None:
    base_dir = Path(__file__).resolve().parent
    pred_df.to_parquet(base_dir / f"{prefix}_predictions.parquet", index=False)
    global_summary.to_csv(base_dir / f"{prefix}_sanity_global.csv", index=False)
    by_profile.to_csv(base_dir / f"{prefix}_sanity_by_profile.csv", index=False)
    monthly.to_csv(base_dir / f"{prefix}_sanity_monthly.csv", index=False)