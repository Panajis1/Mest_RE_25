"""
AC Detection: Training Functions & Internal Validation
Mirror of trainingfunctions_internalvalidation.py but for AC (air conditioner) detection.

Ground truth: Dataport households that have type=AC metering → label = "has_ac"
              Dataport households with only type=TOT (no AC rows) → label = "no_ac"

Key differences from HP:
  - AC is a summer/heat load  → positive temp correlation expected
  - Features emphasize summer months, daytime hours, solar radiation
  - Night-only features NOT used (AC runs during the day)
  - Label is binary: has_ac / no_ac  (no seasonal sub-type like winter_hp/summer_hp)
  - AC label is inferred from presence of type=AC rows with strong summer+temp signal
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "all_sources_load_with_weather.parquet"

RF_RANDOM_STATE = 42

TEMP_F_THRESHOLD = 55.0          # same as HP: convert F→C if max temp > this

CORR_THRESHOLD = 0.20            # minimum |corr| to call it meaningful
TEMP_HOT = 25.0                  # °C — "hot" threshold for AC signal
TEMP_COLD = 10.0                 # °C — cold baseline (AC should be near zero)
MIN_SAMPLES_CORR = 10

EXPECTED_FREQ = "15min"

# Daytime feature extraction (AC runs during the day, opposite of HP night-only)
USE_DAYTIME_FEATURES = True
DAY_RAD_THRESHOLD = 50.0         # W/m² — rows with glob_rad > this are "daytime"
MIN_DAY_ROWS = 100               # minimum daytime rows to compute features

# AC curve quality (from ground-truth AC sub-meter)
AC_MIN_TOTAL_ROWS = 500
AC_MIN_ROWS_SUMMER = 200         # must have summer data to confirm AC
AC_MAX_GAP_DAYS = 21
AC_MIN_COVERAGE_RATIO = 0.30

# TOT curve quality (same relaxed thresholds as HP batch)
TOT_MIN_TOTAL_ROWS = 500
TOT_MIN_ROWS_SUMMER = 0
TOT_MAX_GAP_DAYS = 90
TOT_MIN_COVERAGE_RATIO = 0.10

# Hold-out test sizes
N_TEST_AC = 8
N_TEST_NOAC = 8

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RF_RANDOM_STATE,
    n_jobs=-1,
)

LABEL_TO_INT = {
    "no_ac": 0,
    "has_ac": 1,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# AC-specific features (daytime, summer-focused, positive temp correlation)
FEATURE_COLS = [
    "corr_temp_all",       # overall temp-load correlation (positive → AC signal)
    "corr_temp_hot",       # temp-load correlation when temp > TEMP_HOT
    "corr_rad_all",        # solar radiation correlation (AC follows sun)
    "season_balance",      # (mean_summer - mean_winter) / (|summer|+|winter|) → positive for AC
    "thermal_balance",     # (mean_hot - mean_cold) / (|hot|+|cold|)
    "summer_share",        # fraction of annual load in summer months
    "daytime_share",       # fraction of load during daytime vs night
    "coeff_var",           # coefficient of variation
    "acf_1h",              # autocorrelation lag 4 (1 hour)
    "acf_24h",             # autocorrelation lag 96 (24 hours)
    # ── 새로 추가된 피처 ──────────────────────────────────
    "afternoon_peak_ratio",  # 오후 2~6시 평균 / 전체 평균 (AC는 오후에 집중)
    "peak_summer_hour",      # 여름철 최대 부하 시간대 (AC는 낮 시간대에 피크)
    "summer_vs_spring",      # 여름 평균 / 봄 평균 (계절 점프 크기)
    "hot_load_ratio",        # 더울 때 평균 부하 / 전체 평균 (온도 반응성)
]

# 임계값: prob_has_ac > 이 값이면 has_ac로 분류 (기본 0.5보다 높게)
AC_PROB_THRESHOLD = 0.55


# ============================================================
# PROGRESS
# ============================================================

class ProgressPrinter:
    def __init__(self, total: int, every: int = 25, label: str = "Progress"):
        self.total = max(int(total), 1)
        self.every = max(int(every), 1)
        self.label = label
        self.start = time.perf_counter()

    def update(self, current: int):
        if current == 1 or current % self.every == 0 or current == self.total:
            elapsed = time.perf_counter() - self.start
            rate = current / elapsed if elapsed > 0 else 0.0
            remaining = self.total - current
            eta_sec = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * current / self.total

            def fmt(s: float) -> str:
                if not np.isfinite(s):
                    return "?"
                m, sec = divmod(int(s), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"

            print(
                f"[{self.label}] {current}/{self.total} "
                f"({pct:5.1f}%) | elapsed={fmt(elapsed)} | eta={fmt(eta_sec)}"
            )


# ============================================================
# BASIC UTILS  (same helpers as HP)
# ============================================================

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
    if len(x) <= lag + 2 or x.nunique() <= 1:
        return np.nan
    return float(x.autocorr(lag=lag))


def bounded_balance(a: float, b: float, eps: float = 1e-8):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float((a - b) / (abs(a) + abs(b) + eps))


def make_random_split_seed() -> int:
    return int(time.time_ns() % (2**32 - 1))


# ============================================================
# TEMPERATURE CONVERSION  (identical to HP)
# ============================================================

def convert_f_to_c_if_needed(df: pd.DataFrame, threshold_f: float = TEMP_F_THRESHOLD) -> pd.DataFrame:
    print("\n[1] Temperature conversion F -> C")
    max_temp_by_user = df.groupby("id_customer", sort=False)["temp"].max()
    users_to_convert = max_temp_by_user[max_temp_by_user > threshold_f].index.tolist()
    print(f"Total users: {df['id_customer'].nunique()}")
    print(f"Users converted F→C: {len(users_to_convert)}")
    mask = df["id_customer"].isin(users_to_convert) & df["temp"].notna()
    df.loc[mask, "temp"] = (df.loc[mask, "temp"] - 32.0) * 5.0 / 9.0
    return df


# ============================================================
# TIME QUALITY  (identical to HP)
# ============================================================

def compute_time_quality(df_curve: pd.DataFrame, expected_freq: str = EXPECTED_FREQ):
    if df_curve.empty:
        return {"n_rows": 0, "summer_rows": 0, "winter_rows": 0,
                "max_gap_days": np.nan, "coverage_ratio": 0.0,
                "start": pd.NaT, "end": pd.NaT}
    d = df_curve.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    d = d.drop_duplicates(subset=["dt_utc"], keep="first")
    if d.empty:
        return {"n_rows": 0, "summer_rows": 0, "winter_rows": 0,
                "max_gap_days": np.nan, "coverage_ratio": 0.0,
                "start": pd.NaT, "end": pd.NaT}
    months = d["dt_utc"].dt.month
    summer_rows = int(months.isin([6, 7, 8]).sum())
    winter_rows = int(months.isin([12, 1, 2]).sum())
    dt = d["dt_utc"]
    deltas = dt.diff().dropna()
    max_gap_days = float(deltas.max().total_seconds() / 86400) if len(deltas) > 0 else 0.0
    full_idx = pd.date_range(start=dt.min(), end=dt.max(), freq=expected_freq)
    coverage_ratio = float(len(dt) / len(full_idx)) if len(full_idx) > 0 else 0.0
    return {"n_rows": len(dt), "summer_rows": summer_rows, "winter_rows": winter_rows,
            "max_gap_days": max_gap_days, "coverage_ratio": coverage_ratio,
            "start": dt.min(), "end": dt.max()}


def curve_is_time_usable(
    df_curve, curve_name, min_total_rows, min_rows_summer,
    max_gap_days, min_coverage_ratio,
):
    q = compute_time_quality(df_curve)
    if q["n_rows"] < min_total_rows:
        return False, f"{curve_name}_too_few_rows", q
    if q["summer_rows"] < min_rows_summer:
        return False, f"{curve_name}_too_few_summer_rows", q
    if pd.notna(q["max_gap_days"]) and q["max_gap_days"] > max_gap_days:
        return False, f"{curve_name}_max_gap_too_large", q
    if q["coverage_ratio"] < min_coverage_ratio:
        return False, f"{curve_name}_coverage_too_low", q
    return True, "ok", q


# ============================================================
# AC LABEL INFERENCE  (from ground-truth AC sub-meter)
# Mirrors infer_hp_season_label() but for AC
# ============================================================

def infer_ac_label(ac_df: pd.DataFrame):
    """
    Given a user's AC sub-meter rows, decide whether this user genuinely has AC.
    Returns ("has_ac", reason, quality)  or  (None, reason, quality).

    Logic:
      1. Basic time quality check (enough rows, summer coverage, no huge gap)
      2. Positive temp-AC correlation in hot temperatures  → confirm AC
      3. Or strong positive overall temp-AC correlation    → confirm AC
      4. Otherwise discard (sub-meter exists but signal is too weak)
    """
    ac_df = ac_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    if ac_df.empty:
        return None, "empty_ac_curve", None

    usable, reason, quality = curve_is_time_usable(
        ac_df,
        curve_name="ac",
        min_total_rows=AC_MIN_TOTAL_ROWS,
        min_rows_summer=AC_MIN_ROWS_SUMMER,
        max_gap_days=AC_MAX_GAP_DAYS,
        min_coverage_ratio=AC_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return None, reason, quality

    # Overall correlation: should be positive (more AC when hotter)
    corr_all, _ = safe_corr(ac_df["value_kw_mean"], ac_df["temp"])
    if corr_all is not None and corr_all >= CORR_THRESHOLD:
        return "has_ac", "all_year_corr", quality

    # Hot-weather check
    hot_df = ac_df[ac_df["temp"] > TEMP_HOT]
    corr_hot, _ = safe_corr(hot_df["value_kw_mean"], hot_df["temp"])
    if corr_hot is not None and corr_hot >= CORR_THRESHOLD:
        return "has_ac", "hot_check", quality

    # Mean summer AC load must be meaningfully positive
    summer_df = ac_df[ac_df["dt_utc"].dt.month.isin([6, 7, 8])]
    mean_summer_ac = safe_mean(summer_df["value_kw_mean"])
    if not pd.isna(mean_summer_ac) and mean_summer_ac > 0.05:   # > 50 W average
        return "has_ac", "summer_mean_positive", quality

    return None, "weak_ac_signal", quality


# ============================================================
# TOT CURVE QUALITY CHECK  (same as HP but exported for batch)
# ============================================================

def tot_curve_is_usable(tot_df: pd.DataFrame):
    tot_df = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    usable, reason, quality = curve_is_time_usable(
        tot_df,
        curve_name="tot",
        min_total_rows=TOT_MIN_TOTAL_ROWS,
        min_rows_summer=TOT_MIN_ROWS_SUMMER,
        max_gap_days=TOT_MAX_GAP_DAYS,
        min_coverage_ratio=TOT_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return False, reason, quality
    if int(tot_df["temp"].notna().sum()) < MIN_SAMPLES_CORR:
        return False, "tot_too_few_temp_rows", quality
    return True, "ok", quality


# ============================================================
# FEATURE EXTRACTION FROM TOT  (AC-specific, daytime focused)
# Mirrors extract_tot_features() in HP code
# ============================================================

def extract_tot_features(tot_df: pd.DataFrame):
    """
    Extract AC-detection features from a user's TOT (total load) curve.
    Uses daytime rows (high solar radiation) instead of HP's night-only rows,
    because AC load is a daytime phenomenon.
    """
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

    if USE_DAYTIME_FEATURES:
        day_mask = d["rad_num"].notna() & (d["rad_num"] > DAY_RAD_THRESHOLD)
        d_day = d.loc[day_mask].copy()
        n_day_rows = len(d_day)

        if n_day_rows >= MIN_DAY_ROWS:
            d_feat = d_day
            used_daytime_only = True
        else:
            # Fallback: use full curve (AC signal still partially visible)
            d_feat = d.copy()
    else:
        d_feat = d.copy()

    if d_feat.empty:
        return None

    winter_mask = d_feat["month"].isin([12, 1, 2])
    summer_mask = d_feat["month"].isin([6, 7, 8])
    hot_mask = d_feat["temp_num"] > TEMP_HOT
    cold_mask = d_feat["temp_num"] < TEMP_COLD

    # Correlations
    corr_temp_all, _ = safe_corr(d_feat["load"], d_feat["temp_num"])
    corr_temp_hot, _ = safe_corr(d_feat.loc[hot_mask, "load"], d_feat.loc[hot_mask, "temp_num"])
    corr_rad_all, _ = safe_corr(d_feat["load"], d_feat["rad_num"])

    # Seasonal balance (positive → more load in summer → AC signal)
    mean_summer = safe_mean(d_feat.loc[summer_mask, "load"])
    mean_winter = safe_mean(d_feat.loc[winter_mask, "load"])
    mean_hot = safe_mean(d_feat.loc[hot_mask, "load"])
    mean_cold = safe_mean(d_feat.loc[cold_mask, "load"])

    season_balance = bounded_balance(mean_summer, mean_winter)   # positive for AC
    thermal_balance = bounded_balance(mean_hot, mean_cold)       # positive for AC

    # Summer share of total annual load
    total_load = safe_mean(d_feat["load"])
    summer_share = float(mean_summer / (total_load + 1e-8)) if not pd.isna(mean_summer) and not pd.isna(total_load) else np.nan

    # Daytime share: load during daytime vs night
    day_rows_all = d.loc[d["rad_num"].notna() & (d["rad_num"] > DAY_RAD_THRESHOLD), "load"]
    night_rows_all = d.loc[d["rad_num"].notna() & (d["rad_num"] <= DAY_RAD_THRESHOLD), "load"]
    mean_day_load = safe_mean(day_rows_all)
    mean_night_load = safe_mean(night_rows_all)
    daytime_share = bounded_balance(mean_day_load, mean_night_load)   # positive → more load during day

    # Variability
    load_mean = safe_mean(d_feat["load"])
    load_std = safe_std(d_feat["load"])
    coeff_var = float(load_std / (abs(load_mean) + 1e-8)) if not pd.isna(load_mean) and not pd.isna(load_std) else np.nan

    acf_1h = safe_autocorr(d_feat["load"], lag=4)
    acf_24h = safe_autocorr(d_feat["load"], lag=96)

    # ── 새 피처: 오후 2~6시 집중도 ──────────────────────────
    afternoon_mask = d_feat["hour"].isin([14, 15, 16, 17, 18])
    mean_afternoon = safe_mean(d_feat.loc[afternoon_mask, "load"])
    afternoon_peak_ratio = float(mean_afternoon / (total_load + 1e-8)) if not pd.isna(mean_afternoon) and not pd.isna(total_load) else np.nan

    # ── 새 피처: 여름철 최대 부하 시간대 ────────────────────
    summer_d = d_feat.loc[summer_mask].copy()
    if not summer_d.empty and summer_d["load"].notna().any():
        peak_summer_hour = float(summer_d.groupby("hour")["load"].mean().idxmax())
    else:
        peak_summer_hour = np.nan

    # ── 새 피처: 여름 vs 봄 비율 ────────────────────────────
    spring_mask = d_feat["month"].isin([3, 4, 5])
    mean_spring = safe_mean(d_feat.loc[spring_mask, "load"])
    summer_vs_spring = float(mean_summer / (mean_spring + 1e-8)) if not pd.isna(mean_summer) and not pd.isna(mean_spring) else np.nan

    # ── 새 피처: 더울 때 부하 / 전체 평균 ───────────────────
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
        # Metadata
        "n_rows": q["n_rows"],
        "summer_rows": q["summer_rows"],
        "winter_rows": q["winter_rows"],
        "max_gap_days": q["max_gap_days"],
        "coverage_ratio": q["coverage_ratio"],
        "used_daytime_only": int(used_daytime_only),
        "n_day_rows": int(n_day_rows),
        "n_feature_rows": int(len(d_feat)),
    }


# ============================================================
# BUILD MODELED DATASET
# Mirrors build_modeled_dataset() in HP code
# ============================================================

def build_modeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each user:
      - If type=AC rows exist → infer_ac_label() → use as positive label
      - If only type=TOT (Dataport, no AC sub-meter) → label = "no_ac"
      - Extract TOT-based features for the classifier
    """
    print("\n[2] Building AC modeled dataset")

    records = []
    ac_reason_counts: Counter = Counter()
    tot_reason_counts: Counter = Counter()
    skipped_counts: Counter = Counter()

    grouped = df.groupby("id_customer", sort=False)
    total_users = df["id_customer"].nunique()
    progress = ProgressPrinter(total=total_users, every=25, label="user scan")

    for i, (user_id, user_df) in enumerate(grouped, start=1):
        progress.update(i)

        ac_df = user_df[user_df["type"] == "AC"]
        tot_df = user_df[user_df["type"] == "TOT"]
        source_values = user_df["source"].dropna().astype(str).unique().tolist()
        source_name = source_values[0] if source_values else "UNKNOWN"

        has_ac = not ac_df.empty
        has_tot = not tot_df.empty

        if has_ac:
            # --- Positive examples ---
            label, ac_reason, _ = infer_ac_label(ac_df)
            if label is None:
                ac_reason_counts[ac_reason] += 1
                continue

            if not has_tot:
                skipped_counts["ac_label_but_no_tot"] += 1
                continue

            tot_ok, tot_reason, _ = tot_curve_is_usable(tot_df)
            if not tot_ok:
                tot_reason_counts[tot_reason] += 1
                continue

            feats = extract_tot_features(tot_df)
            if feats is None:
                tot_reason_counts["tot_too_few_day_rows"] += 1
                continue

            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = "has_ac"
            feats["target"] = LABEL_TO_INT["has_ac"]
            records.append(feats)

        elif has_tot:
            # --- Negative examples: Dataport, ECO, REFIT TOT-only users ---
            # ECO and REFIT confirmed to have no AC (not listed in appliance list)
            NO_AC_SOURCES = {"dataport", "eco", "refit"}
            if str(source_name).lower() not in NO_AC_SOURCES:
                skipped_counts["non_eligible_tot_without_ac"] += 1
                continue

            tot_ok, tot_reason, _ = tot_curve_is_usable(tot_df)
            if not tot_ok:
                tot_reason_counts[tot_reason] += 1
                continue

            feats = extract_tot_features(tot_df)
            if feats is None:
                tot_reason_counts["tot_too_few_day_rows"] += 1
                continue

            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = "no_ac"
            feats["target"] = LABEL_TO_INT["no_ac"]
            records.append(feats)

        else:
            skipped_counts["no_ac_no_tot"] += 1

    modeled_df = pd.DataFrame(records)

    print(f"\nEligible modeled users for RF: {len(modeled_df)}")
    if not modeled_df.empty:
        print("\nClass distribution:")
        print(modeled_df["target_name"].value_counts())
        if "used_daytime_only" in modeled_df.columns:
            print("\nFeature mode summary:")
            print(
                modeled_df["used_daytime_only"]
                .value_counts(dropna=False)
                .rename(index={1: "daytime_only", 0: "full_curve_fallback"})
                .to_string()
            )

    print("\nAC discard reasons:")
    for k, v in ac_reason_counts.most_common():
        print(f"  - {k}: {v}")

    print("\nTOT discard reasons:")
    for k, v in tot_reason_counts.most_common():
        print(f"  - {k}: {v}")

    print("\nSkipped reasons:")
    for k, v in skipped_counts.most_common():
        print(f"  - {k}: {v}")

    return modeled_df


# ============================================================
# HOLD-OUT SPLIT
# ============================================================

def make_holdout_split(modeled_df: pd.DataFrame, split_seed: int):
    print("\n[3] Creating hold-out train/test split")
    counts = modeled_df["target_name"].value_counts()
    print(counts)

    n_ac = int(counts.get("has_ac", 0))
    n_noac = int(counts.get("no_ac", 0))

    if n_ac <= N_TEST_AC:
        raise ValueError(f"Too few has_ac users: {n_ac}")
    if n_noac <= N_TEST_NOAC:
        raise ValueError(f"Too few no_ac users: {n_noac}")

    rng = np.random.RandomState(split_seed)
    ac_ids = modeled_df.loc[modeled_df["target_name"] == "has_ac", "id_customer"].tolist()
    noac_ids = modeled_df.loc[modeled_df["target_name"] == "no_ac", "id_customer"].tolist()

    test_ac_ids = rng.choice(ac_ids, size=N_TEST_AC, replace=False).tolist()
    test_noac_ids = rng.choice(noac_ids, size=N_TEST_NOAC, replace=False).tolist()
    test_ids = set(test_ac_ids + test_noac_ids)

    train_df = modeled_df[~modeled_df["id_customer"].isin(test_ids)].copy()
    test_df = modeled_df[modeled_df["id_customer"].isin(test_ids)].copy()

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print("Train class distribution:")
    print(train_df["target_name"].value_counts())
    print("Test class distribution:")
    print(test_df["target_name"].value_counts())

    return train_df, test_df


# ============================================================
# TRAIN & EVALUATE
# ============================================================

def train_and_evaluate_holdout(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n[4] Training RandomForest and evaluating on hold-out set")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    labels_order = [LABEL_TO_INT["no_ac"], LABEL_TO_INT["has_ac"]]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    report = classification_report(
        y_test, y_pred,
        labels=labels_order,
        target_names=[INT_TO_LABEL[x] for x in labels_order],
        digits=3, zero_division=0,
    )

    result_df = test_df[["id_customer", "source", "target_name", "target"]].copy()
    result_df["pred_target"] = y_pred
    result_df["pred_target_name"] = result_df["pred_target"].map(INT_TO_LABEL)
    result_df["correct"] = result_df["target"] == result_df["pred_target"]

    classes = model.named_steps["rf"].classes_
    proba = model.predict_proba(X_test)
    for class_id in labels_order:
        col = f"prob_{INT_TO_LABEL[class_id]}"
        if class_id in classes:
            idx = int(np.where(classes == class_id)[0][0])
            result_df[col] = proba[:, idx]
        else:
            result_df[col] = 0.0

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("Order:", [INT_TO_LABEL[x] for x in labels_order])
    print(cm)
    print("\nClassification report:")
    print(report)

    rf_model = model.named_steps["rf"]
    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(importances.to_string(index=False))

    return model, result_df, importances


# ============================================================
# MAIN  (internal validation, mirrors HP trainingfunctions main)
# ============================================================

def main():
    import time as _time
    total_start = _time.perf_counter()

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / INPUT_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\nReading: {file_path}")
    df = pd.read_parquet(file_path)

    print("Columns:", df.columns.tolist())
    print("Types found:", sorted(df["type"].dropna().unique().tolist()))
    print("Shape:", df.shape)

    needed_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df = df[needed_cols].copy()
    df = df[df["type"].isin(["AC", "TOT"])].copy()

    print("\nAfter keeping AC/TOT rows:")
    print(df["type"].value_counts(dropna=False))

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])
    df = convert_f_to_c_if_needed(df)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    modeled_df = build_modeled_dataset(df)
    if modeled_df.empty:
        raise ValueError("No modeled users available after filtering.")

    split_seed = make_random_split_seed()
    train_df, test_df = make_holdout_split(modeled_df, split_seed=split_seed)
    model, result_df, importances = train_and_evaluate_holdout(train_df, test_df)

    elapsed = _time.perf_counter() - total_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\nTotal runtime: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()