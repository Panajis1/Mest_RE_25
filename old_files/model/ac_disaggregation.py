"""
AC Disaggregation Functions (Optimized)
=========================================
Improvements over v1:
  1. Cache: aligned_df, model, feature_cols saved to disk
             → second run skips training entirely
  2. Parallel streaming: parquet files processed with ProcessPoolExecutor
             → ~N_WORKERS x speedup on streaming step
  3. Memory: float32 throughout, early column pruning

Structure mirrors HP's disaggregation_functions.py but adapted for AC.
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
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

from ac_actrainingfunctions import (
    TEMP_F_THRESHOLD,
    convert_f_to_c_if_needed,
    tot_curve_is_usable,
)

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE   = "all_sources_load_with_weather.parquet"
RANDOM_STATE = 42
EPS          = 1e-6

AC_ON_THRESHOLD_KW = 0.30
TEMP_HARD_CUTOFF   = 20.0
TEMP_SOFT_LOW      = 20.0
TEMP_SOFT_HIGH     = 25.0
DAY_RAD_THRESHOLD  = 50.0
LAGS               = [1, 4, 8, 96]
ROLL_WINDOWS       = [4, 8, 96]
SUMMER_MONTHS      = [6, 7, 8]

# ── Cache ────────────────────────────────────────────────────
CACHE_DIR          = Path(__file__).resolve().parent / "ac_disagg_cache"
ALIGNED_CACHE      = CACHE_DIR / "aligned_df.parquet"
MODEL_CACHE        = CACHE_DIR / "ac_disagg_model.pkl"
FEATURE_COLS_CACHE = CACHE_DIR / "feature_cols.json"

FORCE_REBUILD_ALIGNED = False   # True → rebuild aligned_df from scratch
FORCE_REBUILD_MODEL   = False   # True → retrain model from scratch

# ── Parallel streaming ───────────────────────────────────────
N_STREAM_WORKERS = 4


# ============================================================
# LOGGING
# ============================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_memory(tag: str = "") -> None:
    if psutil is None:
        return
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    log(f"[MEM] {tag} | RAM: {mem_mb:.0f} MB")


# ============================================================
# CACHE UTILS
# ============================================================

def load_or_build_aligned(raw_df: pd.DataFrame) -> pd.DataFrame:
    if not FORCE_REBUILD_ALIGNED and ALIGNED_CACHE.exists():
        log(f"[CACHE] Loading aligned_df: {ALIGNED_CACHE}")
        return pd.read_parquet(ALIGNED_CACHE)
    log("[CACHE] Building aligned_df from scratch...")
    aligned = build_aligned_dataset(raw_df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(ALIGNED_CACHE, index=False)
    log(f"[CACHE] Saved → {ALIGNED_CACHE}")
    return aligned


def load_or_fit_model(aligned_df: pd.DataFrame, feature_cols: list[str]) -> "AcTwoStageModel":
    if not FORCE_REBUILD_MODEL and MODEL_CACHE.exists():
        log(f"[CACHE] Loading model: {MODEL_CACHE}")
        return joblib.load(MODEL_CACHE)
    log("[CACHE] Training model from scratch...")
    df_w = add_sample_weights(aligned_df)
    model = AcTwoStageModel()
    model.fit(df_w, feature_cols)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_CACHE)
    log(f"[CACHE] Saved → {MODEL_CACHE}")
    return model


def save_feature_cols(feature_cols: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLS_CACHE, "w") as f:
        json.dump(feature_cols, f)


def load_feature_cols() -> list[str] | None:
    if FEATURE_COLS_CACHE.exists():
        with open(FEATURE_COLS_CACHE) as f:
            return json.load(f)
    return None


# ============================================================
# USER PROFILE
# ============================================================

def _compute_user_profile(tot_df: pd.DataFrame) -> dict:
    x   = pd.to_numeric(tot_df["tot_kw"], errors="coerce")
    rad = pd.to_numeric(tot_df.get("glob_rad", pd.Series(dtype=float)), errors="coerce")

    q10  = float(x.quantile(0.10)) if len(x) else np.nan
    q50  = float(x.quantile(0.50)) if len(x) else np.nan
    q95  = float(x.quantile(0.95)) if len(x) else np.nan
    mean = float(x.mean())         if len(x) else np.nan
    std  = float(x.std())          if len(x) > 1 else np.nan

    day        = x[rad.notna() & (rad > DAY_RAD_THRESHOLD)]
    night      = x[rad.notna() & (rad <= DAY_RAD_THRESHOLD)]
    day_mean   = float(day.mean())   if len(day)   else np.nan
    night_mean = float(night.mean()) if len(night) else np.nan

    baseline        = q10
    load_factor     = mean / (q95 + EPS)         if pd.notna(mean) and pd.notna(q95)         else np.nan
    spikiness       = std  / (abs(mean) + EPS)   if pd.notna(std)  and pd.notna(mean)        else np.nan
    baseline_share  = baseline / (mean + EPS)    if pd.notna(baseline) and pd.notna(mean)    else np.nan
    day_night_ratio = day_mean / (night_mean + EPS) if pd.notna(day_mean) and pd.notna(night_mean) else np.nan
    frac_above      = float((x > (baseline + AC_ON_THRESHOLD_KW)).mean()) if len(x) else np.nan

    return {
        "user_tot_mean":            mean,
        "user_tot_std":             std,
        "user_tot_p10":             q10,
        "user_tot_p50":             q50,
        "user_tot_p95":             q95,
        "user_baseline_kw":         baseline,
        "user_day_mean":            day_mean,
        "user_night_mean":          night_mean,
        "user_day_night_ratio":     day_night_ratio,
        "user_load_factor":         load_factor,
        "user_spikiness":           spikiness,
        "user_baseline_share":      baseline_share,
        "user_frac_above_baseline": frac_above,
    }


# ============================================================
# CALENDAR + DYNAMIC FEATURES
# ============================================================

def _add_calendar_and_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]          = df["dt_utc"].dt.hour.astype("int16")
    df["minute"]        = df["dt_utc"].dt.minute.astype("int16")
    df["quarter_index"] = (df["hour"] * 4 + df["minute"] // 15).astype("int16")
    df["dow"]           = df["dt_utc"].dt.dayofweek.astype("int16")
    df["month"]         = df["dt_utc"].dt.month.astype("int16")

    df["qidx_sin"]  = np.sin(2 * np.pi * df["quarter_index"] / 96.0).astype("float32")
    df["qidx_cos"]  = np.cos(2 * np.pi * df["quarter_index"] / 96.0).astype("float32")
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7.0).astype("float32")
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7.0).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")

    df["cdd_28"]    = np.maximum(0.0, df["temp"].to_numpy(dtype="float32") - 28.0).astype("float32")
    df["is_day"]    = (df["glob_rad"].notna() & (df["glob_rad"] > DAY_RAD_THRESHOLD)).astype("int8")
    df["is_summer"] = df["month"].isin(SUMMER_MONTHS).astype("int8")

    afternoon_mask          = ((df["hour"] >= 14) & (df["hour"] < 18)).astype("float32")
    df["cdd28_x_afternoon"] = (df["cdd_28"] * afternoon_mask).astype("float32")

    tot = df["tot_kw"].astype("float32")
    for lag in LAGS:
        df[f"tot_lag_{lag}"] = tot.shift(lag).astype("float32")

    shifted = tot.shift(1)
    for w in ROLL_WINDOWS:
        df[f"tot_rollmean_{w}"] = shifted.rolling(w, min_periods=1).mean().astype("float32")
        df[f"tot_rollstd_{w}"]  = shifted.rolling(w, min_periods=2).std().astype("float32")

    df["tot_diff_1"]         = tot.diff(1).astype("float32")
    df["tot_diff_4"]         = tot.diff(4).astype("float32")
    df["tot_minus_baseline"] = (tot - df["user_baseline_kw"].astype("float32")).astype("float32")
    df["tot_over_mean"]      = (tot / (df["user_tot_mean"].astype("float32") + EPS)).astype("float32")
    df["tot_over_p95"]       = (tot / (df["user_tot_p95"].astype("float32") + EPS)).astype("float32")
    df["cdd28_x_day"]        = (df["cdd_28"] * df["is_day"]).astype("float32")
    return df


# ============================================================
# BUILD ALIGNED DATASET
# ============================================================

def build_aligned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records     = []
    total_users = df["id_customer"].nunique()
    log(f"[BUILD] {total_users:,} users...")

    for i, (user_id, user_df) in enumerate(df.groupby("id_customer", sort=False), start=1):
        if i == 1 or i % 25 == 0 or i == total_users:
            log(f"[BUILD] {i:,}/{total_users:,}")

        tot_df = user_df[user_df["type"] == "TOT"].copy()
        ac_df  = user_df[user_df["type"] == "AC"].copy()
        if tot_df.empty or ac_df.empty:
            continue

        source  = str(user_df["source"].dropna().iloc[0]) if not user_df["source"].dropna().empty else "UNKNOWN"
        tot_ok, _, _ = tot_curve_is_usable(tot_df)
        if not tot_ok:
            continue

        tot_df = (
            tot_df[["id_customer", "dt_utc", "value_kw_mean", "temp", "glob_rad"]]
            .drop_duplicates("dt_utc", keep="first")
            .rename(columns={"value_kw_mean": "tot_kw"})
            .sort_values("dt_utc")
        )
        ac_df = (
            ac_df[["dt_utc", "value_kw_mean"]]
            .drop_duplicates("dt_utc", keep="first")
            .rename(columns={"value_kw_mean": "ac_kw"})
            .sort_values("dt_utc")
        )

        merged = tot_df.merge(ac_df, on="dt_utc", how="inner")
        if merged.empty:
            continue
        merged = merged[merged["tot_kw"].abs() > EPS].reset_index(drop=True)
        if merged.empty:
            continue

        merged["source_name"] = source
        profile = _compute_user_profile(merged[["tot_kw", "glob_rad"]])
        for k, v in profile.items():
            merged[k] = v

        merged["ac_ratio"] = (merged["ac_kw"] / (merged["tot_kw"] + EPS)).clip(0.0, 1.0)
        merged["ac_on"]    = (merged["ac_kw"] >= AC_ON_THRESHOLD_KW).astype(int)
        merged = _add_calendar_and_dynamic_features(merged)

        for col in merged.select_dtypes("float64").columns:
            merged[col] = merged[col].astype("float32")

        records.append(merged)

    if not records:
        raise ValueError("No aligned TOT+AC users found.")

    out = pd.concat(records, ignore_index=True)
    log(f"[BUILD] Done: {len(out):,} rows | {out['id_customer'].nunique():,} users")
    return out


# ============================================================
# FEATURE COLUMNS / SPLIT / WEIGHTS
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "id_customer", "dt_utc", "source_name",
        "ac_kw", "ac_ratio", "ac_on", "sample_weight",
    }
    return [c for c in df.columns if c not in excluded]


def split_by_user(aligned_df: pd.DataFrame, test_size: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    users    = aligned_df["id_customer"].unique()
    train_u, test_u = train_test_split(users, test_size=test_size, random_state=RANDOM_STATE)
    train_df = aligned_df[aligned_df["id_customer"].isin(train_u)].copy()
    test_df  = aligned_df[aligned_df["id_customer"].isin(test_u)].copy()
    log(f"[SPLIT] Train: {train_df['id_customer'].nunique()} | Test: {test_df['id_customer'].nunique()} users")
    return train_df, test_df


def add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["id_customer"].value_counts()
    w = df["id_customer"].map(lambda u: 1.0 / float(counts[u]))
    df = df.copy()
    df["sample_weight"] = w / w.mean()
    return df


# ============================================================
# TWO-STAGE MODEL
# ============================================================

class AcTwoStageModel:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.imputer      = SimpleImputer(strategy="median")
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
        X    = self.imputer.fit_transform(train_df[self.feature_cols])
        y_on = train_df["ac_on"].astype(int).to_numpy()
        w    = train_df[row_weight_col].to_numpy(dtype=float) if row_weight_col in train_df.columns else np.ones(len(train_df))

        # Stage 1: classifier trained on all timesteps
        self.clf.fit(X, y_on, sample_weight=w)

        # Stage 2: regressor trained on summer ON timesteps only
        # Rationale: non-summer AC usage in Dataport (Texas) is a different regime
        # from Swiss summer AC — mixing in winter ON rows adds noise to the regressor.
        month = train_df["month"].to_numpy() if "month" in train_df.columns else train_df["dt_utc"].dt.month.to_numpy()
        on_summer_mask = (y_on == 1) & np.isin(month, SUMMER_MONTHS)

        if not np.any(on_summer_mask):
            # Fallback: use all ON rows if no summer ON rows available
            log("[WARN] No summer ON rows found, falling back to all ON rows for regressor.")
            on_summer_mask = y_on == 1

        if not np.any(on_summer_mask):
            raise ValueError("No AC_on=1 rows in training data.")

        # Train regressor to predict ac_kw directly (more stable than ratio)
        # ac_ratio = AC/TOT is unstable: same AC kW gives different ratio depending on TOT
        self.reg.fit(
            X[on_summer_mask],
            train_df.loc[on_summer_mask, "ac_kw"].to_numpy(dtype=float),
            sample_weight=w[on_summer_mask],
        )
        log(f"[FIT] Classifier: {y_on.sum():,} ON rows | Regressor: {on_summer_mask.sum():,} summer ON rows")
        return self

    def predict(self, df: pd.DataFrame, apply_month_gating: bool = True) -> pd.DataFrame:
        """
        apply_month_gating=True  : restrict predictions to Jun/Jul/Aug (use for RE inference)
        apply_month_gating=False : no month filter (use for hold-out validation on Dataport)
        """
        if self.feature_cols is None:
            raise RuntimeError("Model not fitted.")

        X        = self.imputer.transform(df[self.feature_cols])
        p_on     = self.clf.predict_proba(X)[:, 1]
        # Predict ac_kw directly — clipped to [0, inf)
        ac_kw_direct = np.clip(self.reg.predict(X), 0.0, None)

        if apply_month_gating:
            # Month-based gating for Swiss RE data:
            # AC physically implausible outside Jun/Jul/Aug in Switzerland
            month = df["dt_utc"].dt.month.to_numpy()
            is_summer = np.isin(month, SUMMER_MONTHS)
            p_on_adj = np.where(is_summer, p_on, 0.0)
        else:
            # No gating for Dataport hold-out validation
            # (Dataport is US data — AC used year-round in some regions)
            p_on_adj = p_on

        # Only predict AC where classifier says ON
        on_pred    = (p_on_adj >= 0.5).astype(float)
        tot_kw     = df["tot_kw"].to_numpy(dtype=float)

        # Gate by ON prediction, then clip to [0, TOT]
        ac_kw_pred = on_pred * ac_kw_direct
        ac_kw_pred = np.minimum(ac_kw_pred, tot_kw)  # never exceed TOT

        # Compute ratio for logging/output purposes only
        ac_ratio_pred = np.where(tot_kw > 0, ac_kw_pred / (tot_kw + EPS), 0.0)
        ac_ratio_pred = np.clip(ac_ratio_pred, 0.0, 1.0)

        out = df.copy()
        out["ac_on_prob"]    = p_on_adj.astype("float32")
        out["ac_ratio_pred"] = ac_ratio_pred.astype("float32")
        out["ac_kw_pred"]    = ac_kw_pred.astype("float32")
        return out


# ============================================================
# METRICS
# ============================================================

def _safe_r2(y_true, y_pred):
    if len(y_true) < 2 or np.allclose(np.std(y_true), 0):
        return np.nan
    return float(r2_score(y_true, y_pred))


def energy_kwh(power_kw: np.ndarray) -> float:
    return float(np.nansum(power_kw) * 0.25)


def compute_kw_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae       = mean_absolute_error(y_true, y_pred)
    rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mbe       = float(np.mean(y_pred - y_true))
    mean_true = float(np.mean(y_true))
    e_true    = energy_kwh(y_true)
    e_pred    = energy_kwh(y_pred)
    return {
        "n":                int(len(y_true)),
        "mae_kw":           float(mae),
        "rmse_kw":          rmse,
        "medae_kw":         float(median_absolute_error(y_true, y_pred)),
        "r2":               _safe_r2(y_true, y_pred),
        "mbe_kw":           mbe,
        "nmae":             mae / mean_true if mean_true > 0 else np.nan,
        "nmbe":             mbe / mean_true if mean_true > 0 else np.nan,
        "mean_true_kw":     mean_true,
        "mean_pred_kw":     float(np.mean(y_pred)),
        "true_energy_kwh":  e_true,
        "pred_energy_kwh":  e_pred,
        "rel_energy_error": abs(e_pred - e_true) / e_true if e_true > 0 else np.nan,
    }


def compute_on_metrics(y_true_on: np.ndarray, y_prob_on: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred_on = (y_prob_on >= threshold).astype(int)
    out = {
        "n":                 int(len(y_true_on)),
        "precision_on":      float(precision_score(y_true_on, y_pred_on, zero_division=0)),
        "recall_on":         float(recall_score(y_true_on, y_pred_on, zero_division=0)),
        "f1_on":             float(f1_score(y_true_on, y_pred_on, zero_division=0)),
        "mean_true_on":      float(np.mean(y_true_on)),
        "mean_pred_prob_on": float(np.mean(y_prob_on)),
    }
    if len(np.unique(y_true_on)) > 1:
        out["roc_auc_on"] = float(roc_auc_score(y_true_on, y_prob_on))
    else:
        out["roc_auc_on"] = np.nan
    return out


def evaluate_predictions(test_pred_df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    kw_m = compute_kw_metrics(test_pred_df["ac_kw"].to_numpy(), test_pred_df["ac_kw_pred"].to_numpy())
    on_m = compute_on_metrics(test_pred_df["ac_on"].to_numpy(),  test_pred_df["ac_on_prob"].to_numpy())

    by_user = []
    for uid, g in test_pred_df.groupby("id_customer", sort=False):
        row = {"id_customer": uid, "source_name": g["source_name"].iloc[0]}
        row.update(compute_kw_metrics(g["ac_kw"].to_numpy(), g["ac_kw_pred"].to_numpy()))
        row.update(compute_on_metrics(g["ac_on"].to_numpy(),  g["ac_on_prob"].to_numpy()))
        by_user.append(row)

    return kw_m, on_m, pd.DataFrame(by_user).sort_values("rmse_kw").reset_index(drop=True)


def print_evaluation_report(kw_m: dict, on_m: dict, by_user_df: pd.DataFrame) -> None:
    log("\n" + "=" * 60)
    log("AC DISAGGREGATION — Hold-out Evaluation")
    log("=" * 60)
    for section, metrics in [("kW", kw_m), ("ON", on_m)]:
        log(f"\n[{section} metrics]")
        for k, v in metrics.items():
            log(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    log("\n[Per-user summary]")
    cols = ["mae_kw", "rmse_kw", "r2", "mbe_kw", "rel_energy_error"]
    log(by_user_df[cols].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
    log("=" * 60)


# ============================================================
# EXTERNAL DATA HELPERS
# ============================================================

def load_raw_df(input_file: str = INPUT_FILE) -> pd.DataFrame:
    file_path = Path(input_file)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / input_file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    log(f"[LOAD] Reading: {file_path}")
    needed = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df = pd.read_parquet(file_path, columns=needed)
    df = df[df["type"].isin(["TOT", "AC"])].copy()
    df["dt_utc"]        = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df["id_customer"]   = df["id_customer"].astype("string")
    df["value_kw_mean"] = pd.to_numeric(df["value_kw_mean"], errors="coerce").astype("float32")
    df["temp"]          = pd.to_numeric(df["temp"],          errors="coerce").astype("float32")
    df["glob_rad"]      = pd.to_numeric(df["glob_rad"],      errors="coerce").astype("float32")
    df = df.dropna(subset=["dt_utc", "id_customer", "value_kw_mean"])
    df = convert_f_to_c_if_needed(df, threshold_f=TEMP_F_THRESHOLD)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)
    log(f"[LOAD] {len(df):,} rows | {df['id_customer'].nunique():,} users")
    return df


def load_ac_customer_ids(label_file: Path) -> set[str]:
    csv_paths = [label_file]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files in {label_dir}")
    ids: set[str] = set()
    for path in csv_paths:
        df        = pd.read_csv(path)
        id_col    = next((c for c in ["id_customer", "customer_id", "id"] if c in df.columns), None)
        label_col = next((c for c in ["label", "pred_target_name", "target_name"] if c in df.columns), None)
        if id_col is None or label_col is None:
            continue
        df[id_col]    = df[id_col].astype(str)
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        ids.update(df.loc[df[label_col] == "has_ac", id_col].unique())
    log(f"[AC IDs] {len(ids):,} has_ac customers")
    return ids


def load_weather_15min(dt_min=None, dt_max=None) -> pd.DataFrame:
    from envdata import env_data

    log("[WEATHER] Loading weather...")
    _, avg_df = env_data()

    weather = avg_df.reset_index() if "timestamp" not in avg_df.columns else avg_df.copy()
    if "timestamp" not in weather.columns:
        weather = weather.rename(columns={weather.columns[0]: "timestamp"})

    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
    weather = weather.dropna(subset=["timestamp"])
    weather = weather.rename(columns={"t_2m_C": "temp", "global_rad_W": "glob_rad"})
    weather = weather[["timestamp", "temp", "glob_rad"]].rename(columns={"timestamp": "dt_utc"})
    weather["temp"]     = pd.to_numeric(weather["temp"],     errors="coerce").astype("float32")
    weather["glob_rad"] = pd.to_numeric(weather["glob_rad"], errors="coerce").astype("float32")

    if dt_min is not None:
        weather = weather[weather["dt_utc"] >= pd.to_datetime(dt_min, utc=True)]
    if dt_max is not None:
        weather = weather[weather["dt_utc"] <= pd.to_datetime(dt_max, utc=True)]

    weather_15 = (
        weather.set_index("dt_utc")[["temp", "glob_rad"]]
        .sort_index()
        .resample("15min")
        .interpolate(method="time")
        .reset_index()
    )
    log(f"[WEATHER] {len(weather_15):,} rows")
    return weather_15


# ============================================================
# FEATURE BUILDING (per parquet file)
# ============================================================

def build_external_features_for_parquet(
    parquet_path: Path,
    ac_ids: set[str],
    weather_15: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    id_col   = next((c for c in ["ID", "id_customer", "customer_id"] if c in df.columns), None)
    dt_col   = next((c for c in ["DT_UTC", "dt_utc", "timestamp"]    if c in df.columns), None)
    load_col = next((c for c in ["CONSO_KWH", "net_consumption_kwh_15min", "consumption_kwh_15min"] if c in df.columns), None)

    if not all([id_col, dt_col, load_col]):
        return pd.DataFrame()

    tmp = df[[id_col, dt_col, load_col]].copy()
    del df

    tmp = tmp.rename(columns={id_col: "id_customer", dt_col: "dt_utc", load_col: "tot_kw"})
    tmp["id_customer"] = tmp["id_customer"].astype(str)
    tmp = tmp[tmp["id_customer"].isin(ac_ids)]
    if tmp.empty:
        return pd.DataFrame()

    tmp["dt_utc"] = pd.to_datetime(tmp["dt_utc"], utc=True, errors="coerce")
    tmp["tot_kw"] = pd.to_numeric(tmp["tot_kw"], errors="coerce").astype("float32") * 4.0
    tmp = tmp.dropna(subset=["dt_utc", "tot_kw"])
    tmp = tmp.sort_values(["id_customer", "dt_utc"]).reset_index(drop=True)
    tmp = tmp.merge(weather_15, on="dt_utc", how="left")

    records = []
    for user_id, g in tmp.groupby("id_customer", sort=False):
        if len(g) < 96 * 5:
            continue
        g = g.copy()
        g["source_name"] = "external_re"
        profile = _compute_user_profile(g[["tot_kw", "glob_rad"]])
        for k, v in profile.items():
            g[k] = np.float32(v) if isinstance(v, (float, int)) and pd.notna(v) else v
        g = _add_calendar_and_dynamic_features(g)
        for col in g.select_dtypes("float64").columns:
            g[col] = g[col].astype("float32")
        records.append(g)

    del tmp
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ============================================================
# PARALLEL STREAM WORKER (top-level → picklable)
# ============================================================

def _stream_worker(args: tuple) -> tuple[int, int]:
    i, parquet_path_str, ac_ids, weather_15, output_dir_str, feature_cols, model = args
    parquet_path = Path(parquet_path_str)
    output_dir   = Path(output_dir_str)

    ext_df = build_external_features_for_parquet(parquet_path, ac_ids, weather_15)
    if ext_df.empty:
        return i, 0

    model.feature_cols = feature_cols
    pred_df    = model.predict(ext_df)
    chunk_path = output_dir / f"ac_predictions_part_{i:04d}.parquet"

    save_cols = ["id_customer", "dt_utc", "tot_kw", "temp", "glob_rad",
                 "ac_on_prob", "ac_ratio_pred", "ac_kw_pred"]
    pred_df[save_cols].to_parquet(chunk_path, index=False)
    return i, len(pred_df)


# ============================================================
# STREAM PREDICTIONS (parallel)
# ============================================================

def stream_predictions_to_parquet(
    model: AcTwoStageModel,
    parquet_dir: Path,
    ac_ids: set[str],
    weather_15: pd.DataFrame,
    output_dir: Path,
    n_workers: int = N_STREAM_WORKERS,
) -> Path:
    parquet_paths = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.parquet"):
        old.unlink()

    log(f"[STREAM] {len(parquet_paths)} files | {n_workers} workers")

    args_list = [
        (i, str(p), ac_ids, weather_15, str(output_dir), model.feature_cols, model)
        for i, p in enumerate(parquet_paths, 1)
    ]

    total_rows = 0
    kept       = 0

    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_stream_worker, a): a[0] for a in args_list}
            for future in concurrent.futures.as_completed(futures):
                try:
                    i, n_rows = future.result()
                    if n_rows > 0:
                        kept       += 1
                        total_rows += n_rows
                        log(f"[STREAM] Part {i:04d}: {n_rows:,} rows | total: {total_rows:,}")
                except Exception as e:
                    log(f"[STREAM] Worker error: {e}")
    else:
        for args in args_list:
            i, n_rows = _stream_worker(args)
            if n_rows > 0:
                kept       += 1
                total_rows += n_rows
                log(f"[STREAM] {i}/{len(parquet_paths)}: {n_rows:,} rows | total: {total_rows:,}")
            gc.collect()

    if kept == 0:
        raise ValueError("No eligible AC customers found.")

    log(f"[STREAM] Done: {kept} files | {total_rows:,} rows")
    return output_dir



# ============================================================
# SWISS CALIBRATION
# ============================================================

def compute_calibration_factor(
    parquet_dir: Path,
    ac_ids: set[str],
    weather_15: pd.DataFrame,
    model_mean_pred_kw: float,
) -> float:
    """
    Estimate a scaling factor to calibrate model predictions
    to Swiss AC consumption levels.

    Method:
      For each has_ac customer, compute:
        summer_excess = mean_TOT(Jun-Aug) - mean_TOT(Apr-May)
      This excess is assumed to be driven by AC usage.
      The calibration factor = mean(summer_excess) / model_mean_pred_kw

    Parameters
    ----------
    parquet_dir       : directory with RE parquet files
    ac_ids            : set of has_ac customer IDs
    weather_15        : weather dataframe (for merging)
    model_mean_pred_kw: mean ac_kw_pred from the uncalibrated model

    Returns
    -------
    calibration_factor: float >= 0
    """
    parquet_paths = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_paths:
        return 1.0

    summer_excess_list = []

    for path in parquet_paths:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        id_col   = next((c for c in ["ID", "id_customer", "customer_id"] if c in df.columns), None)
        dt_col   = next((c for c in ["DT_UTC", "dt_utc", "timestamp"]    if c in df.columns), None)
        load_col = next((c for c in ["CONSO_KWH", "net_consumption_kwh_15min"] if c in df.columns), None)

        if not all([id_col, dt_col, load_col]):
            continue

        tmp = df[[id_col, dt_col, load_col]].copy()
        del df

        tmp = tmp.rename(columns={id_col: "id_customer", dt_col: "dt_utc", load_col: "tot_kw"})
        tmp["id_customer"] = tmp["id_customer"].astype(str)
        tmp = tmp[tmp["id_customer"].isin(ac_ids)]
        if tmp.empty:
            continue

        tmp["dt_utc"] = pd.to_datetime(tmp["dt_utc"], utc=True, errors="coerce")
        tmp["tot_kw"] = pd.to_numeric(tmp["tot_kw"], errors="coerce").astype("float32") * 4.0
        tmp = tmp.dropna(subset=["dt_utc", "tot_kw"])
        tmp["month"]  = tmp["dt_utc"].dt.month

        for cid, g in tmp.groupby("id_customer"):
            summer     = g[g["month"].isin([6, 7, 8])]["tot_kw"]
            shoulder   = g[g["month"].isin([4, 5])]["tot_kw"]

            if len(summer) < 96 * 10 or len(shoulder) < 96 * 10:
                continue

            excess = float(summer.mean()) - float(shoulder.mean())
            if excess > 0:
                summer_excess_list.append(excess)

        del tmp
        gc.collect()

    if not summer_excess_list or model_mean_pred_kw <= 0:
        log("[CALIB] Could not compute calibration factor — using 1.0")
        return 1.0

    mean_excess = float(np.mean(summer_excess_list))
    factor      = mean_excess / model_mean_pred_kw

    # Clip to reasonable range (0.5x to 10x)
    factor = float(np.clip(factor, 0.05, 10.0))
    #factor = float(np.clip(factor, 0.5, 10.0))

    log(f"[CALIB] Mean summer excess TOT (has_ac): {mean_excess:.3f} kW")
    log(f"[CALIB] Model mean pred kW:              {model_mean_pred_kw:.3f} kW")
    log(f"[CALIB] Calibration factor:              {factor:.3f}x")
    return factor


def apply_calibration_to_chunks(
    predictions_dir: Path,
    calibration_factor: float,
) -> None:
    """
    Apply calibration factor to ac_kw_pred in all prediction chunks.
    Overwrites chunks in-place.
    """
    if abs(calibration_factor - 1.0) < 1e-4:
        log("[CALIB] Factor ~1.0 — skipping calibration")
        return

    chunk_paths = sorted(predictions_dir.glob("*.parquet"))
    log(f"[CALIB] Applying factor {calibration_factor:.3f} to {len(chunk_paths)} chunks...")

    for path in chunk_paths:
        df = pd.read_parquet(path)
        df["ac_kw_pred"] = (df["ac_kw_pred"] * calibration_factor).clip(upper=df["tot_kw"]).astype("float32")
        # Recompute ratio
        df["ac_ratio_pred"] = (df["ac_kw_pred"] / (df["tot_kw"] + 1e-6)).clip(0, 1).astype("float32")
        df.to_parquet(path, index=False)

    log("[CALIB] Done.")

# ============================================================
# SUMMARISE
# ============================================================

def summarize_predictions(predictions_dir: Path) -> pd.DataFrame:
    chunk_paths = sorted(predictions_dir.glob("*.parquet"))
    if not chunk_paths:
        raise FileNotFoundError(f"No prediction chunks in {predictions_dir}")

    rows = []
    for path in chunk_paths:
        df = pd.read_parquet(path)
        for uid, g in df.groupby("id_customer"):
            rows.append({
                "id_customer":     uid,
                "n_timesteps":     len(g),
                "ac_energy_kwh":   energy_kwh(g["ac_kw_pred"].to_numpy()),
                "mean_ac_kw_pred": float(g["ac_kw_pred"].mean()),
                "max_ac_kw_pred":  float(g["ac_kw_pred"].max()),
                "p95_ac_kw_pred":  float(np.percentile(g["ac_kw_pred"], 95)),
                "mean_ac_on_prob": float(g["ac_on_prob"].mean()),
                "frac_ac_on":      float((g["ac_on_prob"] >= 0.5).mean()),
            })
        del df
        gc.collect()

    return pd.DataFrame(rows).sort_values("ac_energy_kwh", ascending=False).reset_index(drop=True)


# ============================================================
# STANDALONE
# ============================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    log("=" * 60)
    log("AC Disaggregation — Internal Validation")
    log("=" * 60)

    raw     = load_raw_df()
    aligned = load_or_build_aligned(raw)
    del raw

    log(f"\nAligned: {len(aligned):,} rows | {aligned['id_customer'].nunique()} users")
    log(f"AC ON fraction: {aligned['ac_on'].mean():.3f}")

    train_df, test_df = split_by_user(aligned)
    feature_cols      = get_feature_columns(train_df)
    save_feature_cols(feature_cols)

    model = load_or_fit_model(aligned, feature_cols)
    del aligned, train_df

    log("\nPredicting on test set...")
    test_pred = model.predict(test_df)
    kw_m, on_m, by_user_df = evaluate_predictions(test_pred)
    print_evaluation_report(kw_m, on_m, by_user_df)