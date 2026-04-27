from __future__ import annotations

from pathlib import Path
from collections import Counter
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
CSV_GLOB_PATTERN = "re-curves-PV_PAC_*.csv"
CSV_WINTER_HP_SOURCE = "winter_hp_csv"
CSV_KWH_TO_KW_FACTOR = 4.0

RF_RANDOM_STATE = 42

TEMP_F_THRESHOLD = 55.0

CORR_THRESHOLD = 0.20
TEMP_COLD = 10.0
TEMP_HOT = 28.0
MIN_SAMPLES_CORR = 10

EXPECTED_FREQ = "15min"

# Night-only feature extraction (no fallback)
USE_NIGHT_ONLY_FEATURES = True
NIGHT_RAD_THRESHOLD = 20.0
MIN_NIGHT_ROWS = 100

# HP curve quality
HP_MIN_TOTAL_ROWS = 1000
HP_MIN_ROWS_WINTER = 500
HP_MIN_ROWS_SUMMER = 500
HP_MAX_GAP_DAYS = 21
HP_MIN_COVERAGE_RATIO = 0.50

# TOT quality
TOT_MIN_TOTAL_ROWS = 500
TOT_MIN_ROWS_WINTER = 0
TOT_MIN_ROWS_SUMMER = 0
TOT_MAX_GAP_DAYS = 90
TOT_MIN_COVERAGE_RATIO = 0.10

# Hold-out test sizes by class
N_TEST_WINTER = 6
N_TEST_SUMMER = 4
N_TEST_NOHP = 4

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RF_RANDOM_STATE,
    n_jobs=-1,
)

LABEL_TO_INT = {
    "no_hp": 0,
    "winter_hp": 1,
    "summer_hp": 2,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

FEATURE_COLS = [
    "corr_temp_all",
    "corr_temp_cold",
    "corr_temp_hot",
    "season_balance",
    "thermal_balance",
    "coeff_var",
    "acf_1h",
    "acf_24h",
]


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

            def fmt(seconds: float) -> str:
                if not np.isfinite(seconds):
                    return "?"
                m, s = divmod(int(seconds), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            print(
                f"[{self.label}] {current}/{self.total} "
                f"({pct:5.1f}%) | elapsed={fmt(elapsed)} | eta={fmt(eta_sec)}"
            )


# ============================================================
# BASIC UTILS
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
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def safe_std(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std())


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


def make_random_split_seed() -> int:
    return int(time.time_ns() % (2**32 - 1))


# ============================================================
# TEMPERATURE CONVERSION
# ============================================================

def convert_f_to_c_if_needed(df: pd.DataFrame, threshold_f: float = TEMP_F_THRESHOLD) -> pd.DataFrame:
    print("\n[1] Temperature conversion F -> C")

    max_temp_by_user = df.groupby("id_customer", sort=False)["temp"].max()
    users_to_convert = max_temp_by_user[max_temp_by_user > threshold_f].index.tolist()

    print(f"Total users in current dataframe: {df['id_customer'].nunique()}")
    print(f"Users converted from F to C: {len(users_to_convert)}")
    if users_to_convert:
        print("First converted users:", users_to_convert[:20])

    mask = df["id_customer"].isin(users_to_convert) & df["temp"].notna()
    df.loc[mask, "temp"] = (df.loc[mask, "temp"] - 32.0) * 5.0 / 9.0

    print("Temperature conversion done.")
    return df


# ============================================================
# TIME QUALITY
# ============================================================

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


# ============================================================
# HP LABEL INFERENCE
# ============================================================

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

    if corr_all is not None and corr_all <= -CORR_THRESHOLD:
        return "winter_hp", "all_year_corr", quality
    if corr_all is not None and corr_all >= CORR_THRESHOLD:
        return "summer_hp", "all_year_corr", quality

    cold_df = hp_df[hp_df["temp"] < TEMP_COLD]
    hot_df = hp_df[hp_df["temp"] > TEMP_HOT]

    corr_cold, _ = safe_corr(cold_df["value_kw_mean"], cold_df["temp"])
    corr_hot, _ = safe_corr(hot_df["value_kw_mean"], hot_df["temp"])

    winter_ok = corr_cold is not None and corr_cold <= -CORR_THRESHOLD
    summer_ok = corr_hot is not None and corr_hot >= CORR_THRESHOLD

    if winter_ok and summer_ok:
        if abs(corr_cold) >= abs(corr_hot):
            return "winter_hp", "cold_check_stronger_than_hot", quality
        return "summer_hp", "hot_check_stronger_than_cold", quality

    if winter_ok:
        return "winter_hp", "cold_check", quality
    if summer_ok:
        return "summer_hp", "hot_check", quality

    return None, "weak_after_temperature_checks", quality


def tot_curve_is_usable(tot_df: pd.DataFrame):
    tot_df = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")

    usable, reason, quality = curve_is_time_usable(
        tot_df,
        curve_name="tot",
        min_total_rows=TOT_MIN_TOTAL_ROWS,
        min_rows_winter=TOT_MIN_ROWS_WINTER,
        min_rows_summer=TOT_MIN_ROWS_SUMMER,
        max_gap_days=TOT_MAX_GAP_DAYS,
        min_coverage_ratio=TOT_MIN_COVERAGE_RATIO,
    )
    if not usable:
        return False, reason, quality

    valid_temp_rows = int(tot_df["temp"].notna().sum())
    if valid_temp_rows < MIN_SAMPLES_CORR:
        return False, "tot_too_few_temp_rows", quality

    return True, "ok", quality


# ============================================================
# FEATURE EXTRACTION FROM TOT
# ============================================================

def extract_tot_features(tot_df: pd.DataFrame):
    d = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    d = d.drop_duplicates(subset=["dt_utc"], keep="first")

    if d.empty:
        return None

    d = d.copy()
    d["month"] = d["dt_utc"].dt.month
    d["load"] = pd.to_numeric(d["value_kw_mean"], errors="coerce")
    d["temp_num"] = pd.to_numeric(d["temp"], errors="coerce")
    d["rad_num"] = pd.to_numeric(d["glob_rad"], errors="coerce")

    used_night_only = False
    n_night_rows = 0

    if USE_NIGHT_ONLY_FEATURES:
        night_mask = d["rad_num"].notna() & (d["rad_num"] <= NIGHT_RAD_THRESHOLD)
        d_night = d.loc[night_mask].copy()
        n_night_rows = len(d_night)

        # SOLO NOTTE OBBLIGATORIO: nessun fallback alla curva completa
        if len(d_night) < MIN_NIGHT_ROWS:
            return None

        d_feat = d_night
        used_night_only = True
    else:
        d_feat = d.copy()

    if d_feat.empty:
        return None

    winter_mask = d_feat["month"].isin([12, 1, 2])
    summer_mask = d_feat["month"].isin([6, 7, 8])
    cold_mask = d_feat["temp_num"] < TEMP_COLD
    hot_mask = d_feat["temp_num"] > TEMP_HOT

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
        "used_night_only": int(used_night_only),
        "n_night_rows": int(n_night_rows),
        "n_feature_rows": int(len(d_feat)),
    }


# ============================================================
# WEATHER / CSV ENRICHMENT
# ============================================================

def load_weather_df_from_envdata() -> pd.DataFrame:
    from envdata import env_data
    print("\n[weather] Downloading weather through envdata.py ...")
    combined_df, _ = env_data()

    if combined_df is None or combined_df.empty:
        raise ValueError("env_data() did not return valid weather data.")

    weather = (
        combined_df
        .groupby("timestamp", as_index=False)
        .mean(numeric_only=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    needed = ["timestamp", "t_2m_C", "global_rad_W"]
    missing = [c for c in needed if c not in weather.columns]
    if missing:
        raise ValueError(f"Required columns are missing in the weather data: {missing}")

    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
    weather = weather.dropna(subset=["timestamp"]).copy()
    weather = weather.rename(columns={
        "timestamp": "dt_utc",
        "t_2m_C": "temp",
        "global_rad_W": "glob_rad",
    })

    weather = weather[["dt_utc", "temp", "glob_rad"]].copy()
    weather["temp"] = pd.to_numeric(weather["temp"], errors="coerce")
    weather["glob_rad"] = pd.to_numeric(weather["glob_rad"], errors="coerce")
    weather = weather.sort_values("dt_utc").drop_duplicates("dt_utc", keep="first")

    weather_15 = (
        weather.set_index("dt_utc")[["temp", "glob_rad"]]
        .resample("15min")
        .interpolate(method="time")
        .reset_index()
    )

    print(f"[weather] 15-minute weather rows: {len(weather_15)}")
    print(f"[weather] Range: {weather_15['dt_utc'].min()} -> {weather_15['dt_utc'].max()}")
    return weather_15


def extract_tot_features_unfiltered_night(tot_df: pd.DataFrame):
    d = tot_df.dropna(subset=["dt_utc"]).sort_values("dt_utc")
    d = d.drop_duplicates(subset=["dt_utc"], keep="first")

    if d.empty:
        return None

    d = d.copy()
    d["month"] = d["dt_utc"].dt.month
    d["load"] = pd.to_numeric(d["value_kw_mean"], errors="coerce")
    d["temp_num"] = pd.to_numeric(d["temp"], errors="coerce")
    d["rad_num"] = pd.to_numeric(d["glob_rad"], errors="coerce")

    night_mask = d["rad_num"].notna() & (d["rad_num"] <= NIGHT_RAD_THRESHOLD)
    d_feat = d.loc[night_mask].copy()
    if d_feat.empty:
        return None

    winter_mask = d_feat["month"].isin([12, 1, 2])
    summer_mask = d_feat["month"].isin([6, 7, 8])
    cold_mask = d_feat["temp_num"] < TEMP_COLD
    hot_mask = d_feat["temp_num"] > TEMP_HOT

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
        "n_night_rows": int(len(d_feat)),
        "n_feature_rows": int(len(d_feat)),
    }


def load_and_model_csv_winter_hp(base_dir: Path, weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    csv_paths = sorted(base_dir.glob(CSV_GLOB_PATTERN))
    if not csv_paths:
        print("\n[CSV] No additional winter HP CSV files found.")
        return pd.DataFrame()

    if weather_df is None:
        weather_df = load_weather_df_from_envdata()
    else:
        weather_df = weather_df.copy()

    weather_df = weather_df.sort_values("dt_utc").reset_index(drop=True)

    records = []
    for csv_path in csv_paths:
        print(f"\n[CSV] Reading additional training file: {csv_path.name}")
        df_csv = pd.read_csv(csv_path, sep=";")

        required_cols = ["Date", "Consommation"]
        missing = [c for c in required_cols if c not in df_csv.columns]
        if missing:
            raise ValueError(f"Required columns are missing in {csv_path.name}: {missing}")

        df_csv = df_csv.copy()
        df_csv["dt_utc"] = pd.to_datetime(df_csv["Date"], utc=True, errors="coerce")
        df_csv["value_kw_mean"] = pd.to_numeric(df_csv["Consommation"], errors="coerce") * CSV_KWH_TO_KW_FACTOR
        df_csv = df_csv.dropna(subset=["dt_utc", "value_kw_mean"]).copy()
        df_csv["id_customer"] = csv_path.stem
        df_csv["type"] = "TOT"
        df_csv["source"] = CSV_WINTER_HP_SOURCE

        df_csv = df_csv[["type", "source", "dt_utc", "value_kw_mean", "id_customer"]]
        df_csv = df_csv.sort_values("dt_utc").reset_index(drop=True)
        df_csv = df_csv.merge(weather_df, on="dt_utc", how="left")

        feats = extract_tot_features_unfiltered_night(df_csv)
        if feats is None:
            print(f"[CSV] Skipped {csv_path.name}: no rows available after no-PV-hour selection.")
            continue

        feats["id_customer"] = csv_path.stem
        feats["source"] = CSV_WINTER_HP_SOURCE
        feats["target_name"] = "winter_hp"
        feats["target"] = LABEL_TO_INT["winter_hp"]
        records.append(feats)

        print(f"[CSV] Added {csv_path.name} with {feats['n_feature_rows']} no-PV rows.")

    modeled_csv_df = pd.DataFrame(records)
    print(f"\n[CSV] Additional winter HP modeled users: {len(modeled_csv_df)}")
    return modeled_csv_df


# ============================================================
# BUILD MODELED DATASET
# ============================================================

def build_modeled_dataset(df: pd.DataFrame):
    print("\n[2] Building modeled dataset")

    records = []

    hp_reason_counts = Counter()
    tot_reason_counts = Counter()
    skipped_unlabeled_counts = Counter()

    bad_hp_examples = []
    bad_tot_examples = []
    hp_only_examples = []
    hp_only_counts = Counter()

    grouped = df.groupby("id_customer", sort=False)
    total_users = df["id_customer"].nunique()
    progress = ProgressPrinter(total=total_users, every=25, label="user scan")

    for i, (user_id, user_df) in enumerate(grouped, start=1):
        progress.update(i)

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
                if len(bad_hp_examples) < 20:
                    bad_hp_examples.append((user_id, hp_reason))
                continue

            if not has_tot:
                hp_only_counts[season_label] += 1
                if len(hp_only_examples) < 20:
                    hp_only_examples.append((user_id, season_label, source_name))
                continue

            tot_ok, tot_reason, _ = tot_curve_is_usable(tot_df)
            if not tot_ok:
                tot_reason_counts[tot_reason] += 1
                if len(bad_tot_examples) < 20:
                    bad_tot_examples.append((user_id, tot_reason))
                continue

            feats = extract_tot_features(tot_df)
            if feats is None:
                tot_reason_counts["tot_too_few_night_rows"] += 1
                if len(bad_tot_examples) < 20:
                    bad_tot_examples.append((user_id, "tot_too_few_night_rows"))
                continue

            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = season_label
            feats["target"] = LABEL_TO_INT[season_label]
            records.append(feats)
            continue

        if (not has_hp) and has_tot:
            # Accetta tutte le source (ECO, REFIT, ecc.) come no_hp
            tot_ok, tot_reason, _ = tot_curve_is_usable(tot_df)
            if not tot_ok:
                tot_reason_counts[tot_reason] += 1
                if len(bad_tot_examples) < 20:
                    bad_tot_examples.append((user_id, tot_reason))
                continue

            feats = extract_tot_features(tot_df)
            if feats is None:
                tot_reason_counts["tot_too_few_night_rows"] += 1
                if len(bad_tot_examples) < 20:
                    bad_tot_examples.append((user_id, "tot_too_few_night_rows"))
                continue

            feats["id_customer"] = user_id
            feats["source"] = source_name
            feats["target_name"] = "no_hp"
            feats["target"] = LABEL_TO_INT["no_hp"]
            records.append(feats)
            continue

        skipped_unlabeled_counts["users_without_hp_and_without_tot"] += 1

    modeled_df = pd.DataFrame(records)

    print(f"\nEligible modeled users for RF: {len(modeled_df)}")
    if not modeled_df.empty:
        print("\nClass distribution:")
        print(modeled_df["target_name"].value_counts())

        if "used_night_only" in modeled_df.columns:
            print("\nFeature mode summary:")
            print(
                modeled_df["used_night_only"]
                .value_counts(dropna=False)
                .rename(index={1: "night_only", 0: "full_curve"})
                .to_string()
            )

    print(f"\nValid HP-only users kept out of RF: {sum(hp_only_counts.values())}")
    if hp_only_counts:
        print(pd.Series(hp_only_counts).sort_index())

    print("\nHP discard reasons:")
    if hp_reason_counts:
        for k, v in hp_reason_counts.most_common():
            print(f"- {k}: {v}")
    else:
        print("- none")

    print("\nTOT discard reasons:")
    if tot_reason_counts:
        for k, v in tot_reason_counts.most_common():
            print(f"- {k}: {v}")
    else:
        print("- none")

    print("\nSkipped / unlabeled reasons:")
    if skipped_unlabeled_counts:
        for k, v in skipped_unlabeled_counts.most_common():
            print(f"- {k}: {v}")
    else:
        print("- none")

    if bad_hp_examples:
        print("\nExample discarded HP users:")
        for x in bad_hp_examples:
            print(x)

    if bad_tot_examples:
        print("\nExample discarded users due to TOT:")
        for x in bad_tot_examples:
            print(x)

    if hp_only_examples:
        print("\nExample valid HP-only users kept out of RF:")
        for x in hp_only_examples:
            print(x)

    return modeled_df


# ============================================================
# HOLD-OUT SPLIT
# ============================================================

def make_holdout_split(modeled_df: pd.DataFrame, split_seed: int):
    print("\n[3] Creating hold-out train/test split")

    counts = modeled_df["target_name"].value_counts()
    print("\nAvailable class counts:")
    print(counts)

    n_winter = int(counts.get("winter_hp", 0))
    n_summer = int(counts.get("summer_hp", 0))
    n_nohp = int(counts.get("no_hp", 0))

    if n_winter <= N_TEST_WINTER:
        raise ValueError(f"Too few winter_hp users: {n_winter}")
    if n_summer <= N_TEST_SUMMER:
        raise ValueError(f"Too few summer_hp users: {n_summer}")
    if n_nohp <= N_TEST_NOHP:
        raise ValueError(f"Too few no_hp users: {n_nohp}")

    print(f"\nRandom split seed used in this run: {split_seed}")
    rng = np.random.RandomState(split_seed)

    winter_ids = modeled_df.loc[modeled_df["target_name"] == "winter_hp", "id_customer"].tolist()
    summer_ids = modeled_df.loc[modeled_df["target_name"] == "summer_hp", "id_customer"].tolist()
    nohp_ids = modeled_df.loc[modeled_df["target_name"] == "no_hp", "id_customer"].tolist()

    test_winter_ids = rng.choice(winter_ids, size=N_TEST_WINTER, replace=False).tolist()
    test_summer_ids = rng.choice(summer_ids, size=N_TEST_SUMMER, replace=False).tolist()
    test_nohp_ids = rng.choice(nohp_ids, size=N_TEST_NOHP, replace=False).tolist()

    test_ids = set(test_winter_ids + test_summer_ids + test_nohp_ids)

    train_df = modeled_df[~modeled_df["id_customer"].isin(test_ids)].copy()
    test_df = modeled_df[modeled_df["id_customer"].isin(test_ids)].copy()

    print(f"\nTrain users: {len(train_df)}")
    print(f"Test users: {len(test_df)}")

    print("\nTrain class distribution:")
    print(train_df["target_name"].value_counts())

    print("\nTest class distribution:")
    print(test_df["target_name"].value_counts())

    print("\nTest users:")
    print(
        test_df[["id_customer", "target_name", "source"]]
        .sort_values(["target_name", "id_customer"])
        .to_string(index=False)
    )

    return train_df, test_df


# ============================================================
# MODEL
# ============================================================

def train_and_evaluate_holdout(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n[4] Training Random Forest and evaluating on hold-out set")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    train_classes = sorted(y_train.unique().tolist())
    if len(train_classes) < 3:
        raise ValueError(
            f"Training set does not contain all 3 classes. Present classes: {train_classes}"
        )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_PARAMS)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc_3class = balanced_accuracy_score(y_test, y_pred)

    y_test_bin = (y_test != LABEL_TO_INT["no_hp"]).astype(int)
    y_pred_bin = (pd.Series(y_pred, index=y_test.index) != LABEL_TO_INT["no_hp"]).astype(int)
    bal_acc_binary = balanced_accuracy_score(y_test_bin, y_pred_bin)

    labels_order = [
        LABEL_TO_INT["no_hp"],
        LABEL_TO_INT["winter_hp"],
        LABEL_TO_INT["summer_hp"],
    ]

    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels_order,
        target_names=[INT_TO_LABEL[x] for x in labels_order],
        digits=3,
        zero_division=0,
    )

    result_df = test_df[["id_customer", "source", "target_name", "target"]].copy()
    result_df["pred_target"] = y_pred
    result_df["pred_target_name"] = result_df["pred_target"].map(INT_TO_LABEL)
    result_df["correct"] = result_df["target"] == result_df["pred_target"]

    classes = model.named_steps["rf"].classes_
    proba = model.predict_proba(X_test)

    for class_id in labels_order:
        col_name = f"prob_{INT_TO_LABEL[class_id]}"
        if class_id in classes:
            idx = int(np.where(classes == class_id)[0][0])
            result_df[col_name] = proba[:, idx]
        else:
            result_df[col_name] = 0.0

    print("\n[5] HOLD-OUT RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced accuracy 3-class: {bal_acc_3class:.4f}")
    print(f"Balanced accuracy binary (HP vs no_HP): {bal_acc_binary:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("Order:", [INT_TO_LABEL[x] for x in labels_order])
    print(cm)

    print("\nClassification report:")
    print(report)

    print("\nTest-set predictions:")
    cols_show = [
        "id_customer",
        "source",
        "target_name",
        "pred_target_name",
        "prob_no_hp",
        "prob_winter_hp",
        "prob_summer_hp",
        "correct",
    ]
    extra_cols = [c for c in ["used_night_only", "n_night_rows", "n_feature_rows"] if c in result_df.columns]
    print(
        result_df[["id_customer", "source"] + extra_cols + cols_show[2:]]
        .sort_values(["target_name", "id_customer"])
        .to_string(index=False)
    )

    rf_model = model.named_steps["rf"]
    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nFeature importances:")
    print(importances.to_string(index=False))

    return model, result_df, importances


# ============================================================
# SOURCE CHECK
# ============================================================

def print_source_check(df: pd.DataFrame, modeled_df: pd.DataFrame):
    print("\n[Source check] Sources present in modeled dataset")

    if modeled_df.empty:
        print("No modeled users available.")
        return

    if "source" not in df.columns:
        print("Column 'source' not available in current dataframe.")
        return

    modeled_users = set(modeled_df["id_customer"])
    df_modeled_users = df[df["id_customer"].isin(modeled_users)].copy()

    sources = sorted(df_modeled_users["source"].dropna().unique().tolist())

    print("\nSources found:")
    for s in sources:
        print("-", s)

    print("\nRows by source and type:")
    source_type_counts = (
        df_modeled_users.groupby(["source", "type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(source_type_counts.to_string())

    print("\nUsers by source and type presence:")
    source_user_type = (
        df_modeled_users.groupby(["source", "id_customer", "type"])
        .size()
        .reset_index(name="n_rows")
    )

    source_user_presence = (
        source_user_type.assign(present=1)
        .pivot_table(
            index=["source", "id_customer"],
            columns="type",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    if "HP" not in source_user_presence.columns:
        source_user_presence["HP"] = 0
    if "TOT" not in source_user_presence.columns:
        source_user_presence["TOT"] = 0

    tmp = source_user_presence.copy()
    tmp["has_hp_and_tot"] = ((tmp["HP"] == 1) & (tmp["TOT"] == 1)).astype(int)
    tmp["has_tot_only"] = ((tmp["HP"] == 0) & (tmp["TOT"] == 1)).astype(int)

    source_summary = (
        tmp.groupby("source")
        .agg(
            users_total=("id_customer", "nunique"),
            users_with_tot=("TOT", "sum"),
            users_with_hp=("HP", "sum"),
            users_with_hp_and_tot=("has_hp_and_tot", "sum"),
            users_with_tot_only=("has_tot_only", "sum"),
        )
        .sort_index()
    )

    print(source_summary.to_string())

    print("\nImportant note:")
    print(
        "HP rows can legitimately be present in the analyzed sources because HP is used "
        "only to assign the ground-truth label. The classifier itself uses only TOT-based features."
    )




# ============================================================
# MAIN
# ============================================================

def main():
    total_start = time.perf_counter()

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / INPUT_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\nReading file: {file_path}")
    df = pd.read_parquet(file_path)

    print("\nColumns found:")
    print(df.columns.tolist())

    print("\nAppliance types found:")
    types = sorted(df["type"].dropna().unique().tolist())
    print(types)

    print("\nCounts by type:")
    print(df["type"].value_counts(dropna=False))

    print("\nDataset shape:")
    print(df.shape)

    print("\nReached point after initial inspection.")

    needed_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df = df[needed_cols].copy()

    df = df[df["type"].isin(["HP", "TOT"])].copy()

    print("\nAfter keeping only HP/TOT rows:")
    print(df["type"].value_counts(dropna=False))
    print("Reduced shape:", df.shape)

    print("\nPreprocessing dt_utc once...")
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])

    df = convert_f_to_c_if_needed(df, threshold_f=TEMP_F_THRESHOLD)

    print("\nSorting by id_customer, type, dt_utc...")
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    modeled_df = build_modeled_dataset(df)

    weather_df = load_weather_df_from_envdata()
    modeled_csv_df = load_and_model_csv_winter_hp(base_dir, weather_df=weather_df)
    if not modeled_csv_df.empty:
        modeled_df = pd.concat([modeled_df, modeled_csv_df], ignore_index=True)
        print("\n[CSV] Updated class distribution after CSV append:")
        print(modeled_df["target_name"].value_counts(dropna=False).to_string())

    if modeled_df.empty:
        raise ValueError("No modeled users available after filtering.")

    print_source_check(df, modeled_df)

    split_seed = make_random_split_seed()
    train_df, test_df = make_holdout_split(modeled_df, split_seed=split_seed)
    model, result_df, importances = train_and_evaluate_holdout(train_df, test_df)
    elapsed = time.perf_counter() - total_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\nTotal runtime: {h:02d}:{m:02d}:{s:02d}")




if __name__ == "__main__":
    main()


