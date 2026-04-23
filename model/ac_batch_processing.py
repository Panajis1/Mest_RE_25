"""
AC Detection: Batch Processing (캐시 + Threshold 스캔 버전)

변경사항:
  - modeled_df, inference features, model을 캐시로 저장/로드
  - threshold 스캔 루프 추가 (매번 전체 파이프라인 재실행 불필요)
  - FORCE_REBUILD 플래그로 캐시 강제 재생성 가능
"""
print("Starting AC batch processing...")

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ac_actrainingfunctions import (
    RF_PARAMS,
    TEMP_F_THRESHOLD,
    INT_TO_LABEL,
    FEATURE_COLS,
    AC_PROB_THRESHOLD,
    convert_f_to_c_if_needed,
    build_modeled_dataset,
    make_holdout_split,
    train_and_evaluate_holdout,
    tot_curve_is_usable,
    extract_tot_features,
)

import sys
sys.path.append(r"C:\Users\jiniy\Desktop\CS\Mest_RE_25\data")
from envdata import env_data


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\jiniy\Desktop\CS")
TRAIN_FILE = BASE_DIR / "all_sources_load_with_weather.parquet"

PARQUET_DIR = Path("C:/Users/jiniy/Desktop/CS/ETHZ_ALL")
OUTPUT_DIR = PARQUET_DIR / "ac_detection_outputs"

# ── Cache File Paths ──────────────────────────────────────────
CACHE_DIR = BASE_DIR / "ac_cache"
MODELED_DF_CACHE    = CACHE_DIR / "modeled_df_cache.parquet"
MODEL_CACHE         = CACHE_DIR / "ac_model.pkl"
INFERENCE_FEAT_CACHE = CACHE_DIR / "inference_feats_cache.parquet"

# ── True로 바꾸면 캐시 무시하고 처음부터 재실행 ───────────── 
FORCE_REBUILD_MODELED   = False   # build_modeled_dataset 재실행
FORCE_REBUILD_MODEL     = False   # 모델 재학습
FORCE_REBUILD_INFERENCE = False   # inference feature 재추출

# ── Threshold 스캔 범위 ─────────────────────────────────────
THRESHOLD_SCAN = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70]  # 범위 좁혀서 재스캔
#chosen_threshold = 0.65  # 일단 아무 값, 스캔 후 결정

KWH_TO_KW_FACTOR = 4.0
MAX_ANNUAL_CONSUMPTION_KWH = 10_000.0
NEW_SOURCE_NAME = "new_parquet"


# ============================================================
# CACHE UTILS
# ============================================================

def load_or_build_modeled_df(df: pd.DataFrame) -> pd.DataFrame:
    """modeled_df 캐시 로드 or 빌드 후 저장"""
    if not FORCE_REBUILD_MODELED and MODELED_DF_CACHE.exists():
        print(f"\n[CACHE] Loading modeled_df from cache: {MODELED_DF_CACHE}")
        return pd.read_parquet(MODELED_DF_CACHE)

    print("\n[CACHE] Building modeled_df from scratch...")
    modeled_df = build_modeled_dataset(df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    modeled_df.to_parquet(MODELED_DF_CACHE, index=False)
    print(f"[CACHE] Saved modeled_df → {MODELED_DF_CACHE}")
    return modeled_df


def load_or_fit_model(modeled_df: pd.DataFrame):
    """모델 캐시 로드 or 학습 후 저장"""
    if not FORCE_REBUILD_MODEL and MODEL_CACHE.exists():
        print(f"\n[CACHE] Loading model from cache: {MODEL_CACHE}")
        return joblib.load(MODEL_CACHE)

    print("\n[CACHE] Training model from scratch...")
    model = fit_model(modeled_df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_CACHE)
    print(f"[CACHE] Saved model → {MODEL_CACHE}")
    return model


def load_or_build_inference_feats(weather_df: pd.DataFrame) -> pd.DataFrame:
    """inference feature 캐시 로드 or 추출 후 저장"""
    if not FORCE_REBUILD_INFERENCE and INFERENCE_FEAT_CACHE.exists():
        print(f"\n[CACHE] Loading inference features from cache: {INFERENCE_FEAT_CACHE}")
        return pd.read_parquet(INFERENCE_FEAT_CACHE)

    print("\n[CACHE] Extracting inference features from scratch...")
    parquet_files = list_input_parquet_files()
    all_feats = []
    for parquet_file in parquet_files:
        df_new_tot = load_new_parquet_with_weather(parquet_file, weather_df)
        feats_df = build_inference_features(df_new_tot)
        feats_df["source_file"] = parquet_file.stem
        all_feats.append(feats_df)
        del df_new_tot
        gc.collect()

    combined = pd.concat(all_feats, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(INFERENCE_FEAT_CACHE, index=False)
    print(f"[CACHE] Saved inference features → {INFERENCE_FEAT_CACHE}")
    return combined


# ============================================================
# THRESHOLD SCAN
# ============================================================

def scan_thresholds(model, feats_df: pd.DataFrame):
    """
    prob_has_ac를 먼저 계산해두고,
    threshold만 바꿔가며 AC 비율를 빠르게 확인
    """
    print("\n========== THRESHOLD SCAN ==========")

    X = feats_df[FEATURE_COLS].copy()
    proba = model.predict_proba(X)
    classes = model.named_steps["rf"].classes_

    if 1 in classes:
        idx = int(np.where(classes == 1)[0][0])
        prob_has_ac = proba[:, idx]
    else:
        prob_has_ac = np.zeros(len(feats_df))

    feats_df = feats_df.copy()
    feats_df["prob_has_ac"] = prob_has_ac

    print(f"\n{'Threshold':>10} | {'AC users':>10} | {'Total':>10} | {'AC ratio':>10}")
    print("-" * 48)
    for threshold in THRESHOLD_SCAN:
        n_ac = int((prob_has_ac >= threshold).sum())
        total = len(prob_has_ac)
        ratio = n_ac / total * 100
        marker = " ←" if abs(ratio - 10.0) < 3.0 else ""   # 10% 근처 표시
        print(f"{threshold:>10.2f} | {n_ac:>10,} | {total:>10,} | {ratio:>9.1f}%{marker}")

    return feats_df  # prob_has_ac 컬럼 포함


# ============================================================
# TRAINING ON THE OLD DATASET
# ============================================================

def load_old_training_df() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    print(f"\n[TRAIN] Reading: {TRAIN_FILE}")
    df = pd.read_parquet(TRAIN_FILE)

    needed_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df = df[needed_cols].copy()
    df = df[df["type"].isin(["AC", "TOT"])].copy()

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])
    df = convert_f_to_c_if_needed(df, threshold_f=TEMP_F_THRESHOLD)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    return df   # raw df 반환 (modeled_df 빌드는 load_or_build_modeled_df에서)


def fit_model(modeled_df: pd.DataFrame):
    X = modeled_df[FEATURE_COLS].copy()
    y = modeled_df["target"].copy()

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_PARAMS)),
    ])

    print("\n[TRAIN] Fitting AC model on full labeled dataset...")
    model.fit(X, y)
    print("[TRAIN] Fit completed.")

    rf_model = model.named_steps["rf"]
    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n[TRAIN] Feature importances:")
    print(importances.to_string(index=False))

    return model


# ============================================================
# WEATHER
# ============================================================

def load_weather_df() -> pd.DataFrame:
    print("\n[WEATHER] Downloading weather through envdata.py ...")
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
        raise ValueError(f"Required columns missing in weather data: {missing}")

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"], utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")
    weather = weather.dropna(subset=["timestamp"]).copy()
    weather = weather.rename(columns={"timestamp": "dt_utc", "t_2m_C": "temp", "global_rad_W": "glob_rad"})
    weather = weather[["dt_utc", "temp", "glob_rad"]].copy()
    weather["temp"] = pd.to_numeric(weather["temp"], errors="coerce")
    weather["glob_rad"] = pd.to_numeric(weather["glob_rad"], errors="coerce")
    weather = (
        weather.sort_values("dt_utc")
        .drop_duplicates("dt_utc", keep="first")
        .reset_index(drop=True)
    )

    weather_15 = (
        weather.set_index("dt_utc")[["temp", "glob_rad"]]
        .resample("15min")
        .interpolate(method="time")
        .reset_index()
    )

    print(f"[WEATHER] Resampled 15-min rows: {len(weather_15)}")
    return weather_15


# ============================================================
# PARQUET FILES
# ============================================================

def list_input_parquet_files() -> list[Path]:
    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f"Input parquet directory not found: {PARQUET_DIR}")
    parquet_files = sorted(
        p for p in PARQUET_DIR.iterdir()
        if p.is_file() and p.suffix == ".parquet"
    )
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {PARQUET_DIR}")
    return parquet_files


def load_new_parquet_with_weather(parquet_file: Path, weather_df: pd.DataFrame) -> pd.DataFrame:
    if not parquet_file.exists():
        raise FileNotFoundError(f"New parquet file not found: {parquet_file}")

    print(f"\n[NEW] Reading: {parquet_file}")
    df_new = pd.read_parquet(parquet_file, columns=["ID", "DT_UTC", "CONSO_KWH"])

    df_new = df_new.copy()
    df_new["ID"] = df_new["ID"].astype(str)
    df_new["DT_UTC"] = pd.to_datetime(df_new["DT_UTC"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    df_new["CONSO_KWH"] = pd.to_numeric(df_new["CONSO_KWH"], errors="coerce")
    df_new = df_new.dropna(subset=["ID", "DT_UTC", "CONSO_KWH"]).copy()

    annual_kwh = (
        df_new.groupby("ID", dropna=False)["CONSO_KWH"]
        .sum(min_count=1)
        .rename("annual_consumption_kwh")
        .reset_index()
    )
    valid_ids = set(
        annual_kwh.loc[annual_kwh["annual_consumption_kwh"] <= MAX_ANNUAL_CONSUMPTION_KWH, "ID"].astype(str)
    )
    df_new = df_new[df_new["ID"].isin(valid_ids)].copy()
    if df_new.empty:
        raise ValueError("No users remain after annual consumption filter.")

    df_new["value_kw_mean"] = df_new["CONSO_KWH"] * KWH_TO_KW_FACTOR
    df_new = df_new.rename(columns={"ID": "id_customer", "DT_UTC": "dt_utc"})
    df_new["type"] = "TOT"
    df_new["source"] = NEW_SOURCE_NAME

    df_new = df_new.sort_values("dt_utc").reset_index(drop=True)
    weather_df = weather_df.sort_values("dt_utc").reset_index(drop=True)
    df_new = df_new.merge(weather_df, on="dt_utc", how="left")

    final_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df_new = df_new[final_cols].copy()
    for col in ["glob_rad", "temp", "value_kw_mean"]:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    return df_new


# ============================================================
# FEATURE EXTRACTION FOR INFERENCE
# ============================================================

def build_inference_features(df_new_tot: pd.DataFrame) -> pd.DataFrame:
    records = []
    discard_reasons: dict[str, int] = {}

    for user_id, user_df in df_new_tot.groupby("id_customer", sort=False):
        tot_df = user_df[user_df["type"] == "TOT"].copy()

        tot_ok, tot_reason, _ = tot_curve_is_usable(tot_df)
        if not tot_ok:
            discard_reasons[tot_reason] = discard_reasons.get(tot_reason, 0) + 1
            continue

        feats = extract_tot_features(tot_df)
        if feats is None:
            discard_reasons["tot_feature_extraction_failed"] = (
                discard_reasons.get("tot_feature_extraction_failed", 0) + 1
            )
            continue

        feats["id_customer"] = user_id
        feats["source"] = NEW_SOURCE_NAME
        records.append(feats)

    feats_df = pd.DataFrame(records)
    print(f"\n[NEW] Valid users for inference: {len(feats_df)}")

    if discard_reasons:
        print("[NEW] Discard reasons:")
        for k, v in sorted(discard_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {v}")

    if feats_df.empty:
        raise ValueError("No valid users remain for inference after TOT filters.")

    return feats_df


# ============================================================
# PREDICTION (선택한 threshold로 최종 예측)
# ============================================================

def predict_users(model, feats_df: pd.DataFrame, threshold: float = AC_PROB_THRESHOLD) -> pd.DataFrame:
    X_new = feats_df[FEATURE_COLS].copy()

    proba = model.predict_proba(X_new)
    classes = model.named_steps["rf"].classes_

    extra_cols = [c for c in ["used_daytime_only", "n_day_rows", "n_feature_rows", "source_file"] if c in feats_df.columns]
    out = feats_df[["id_customer", "source"] + extra_cols + FEATURE_COLS].copy()

    for class_id, class_name in INT_TO_LABEL.items():
        col_name = f"prob_{class_name}"
        if class_id in classes:
            idx = int(np.where(classes == class_id)[0][0])
            out[col_name] = proba[:, idx]
        else:
            out[col_name] = 0.0

    out["pred_has_ac"] = (out["prob_has_ac"] >= threshold).astype(int)
    out["pred_target_name"] = out["pred_has_ac"].map({1: "has_ac", 0: "no_ac"})

    return out.sort_values(
        ["pred_has_ac", "prob_has_ac", "id_customer"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_prediction_csv(pred_df: pd.DataFrame, threshold: float) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"ac_labels_all_thresh{int(threshold*100)}.csv"
    cols = ["id_customer", "pred_target_name", "prob_has_ac"]
    if "source_file" in pred_df.columns:
        cols = ["id_customer", "source_file", "pred_target_name", "prob_has_ac"]
    export_df = pred_df[cols].rename(columns={"pred_target_name": "label"})
    export_df.to_csv(output_file, index=False)
    print(f"\n[OUTPUT] CSV saved to: {output_file}")
    return output_file


# ============================================================
# VISUALISATION
# ============================================================

def plot_ac_results(merged_df: pd.DataFrame, output_dir: Path, model=None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir.mkdir(parents=True, exist_ok=True)

    COLORS = {"has_ac": "#E8634A", "no_ac": "#4A90D9"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("AC Detection Results — RE Dataset", fontsize=15, fontweight="bold", y=1.02)

    ax1 = axes[0]
    counts = merged_df["label"].value_counts()
    labels = counts.index.tolist()
    colors = [COLORS.get(l, "#aaa") for l in labels]
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax1.set_title(f"Overall AC ratio\n(n={len(merged_df):,} users)", fontsize=12)

    ax2 = axes[1]
    if "prob_has_ac" in merged_df.columns:
        for label, grp in merged_df.groupby("label"):
            ax2.hist(
                grp["prob_has_ac"].dropna(), bins=30, alpha=0.65,
                color=COLORS.get(label, "#aaa"), label=label,
                edgecolor="white", linewidth=0.5,
            )
        ax2.set_xlabel("prob_has_ac")
        ax2.set_ylabel("Number of users")
        ax2.set_title("Distribution of AC probability score")
        ax2.legend()
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    else:
        ax2.text(0.5, 0.5, "prob_has_ac column\nnot found", ha="center", va="center")

    ax3 = axes[2]
    if model is not None:
        rf_model = model.named_steps["rf"]
        importances = pd.Series(
            rf_model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=True)
        bars = ax3.barh(importances.index, importances.values, color="#4A90D9", edgecolor="white", linewidth=0.8)
        ax3.set_xlabel("Importance")
        ax3.set_title("Feature importances (RandomForest)")
        for bar, val in zip(bars, importances.values):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / "ac_detection_results.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved to: {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    import time as _time
    total_start = _time.perf_counter()

    # ── STEP 1: modeled_df (캐시 우선) ──────────────────────
    print("\n========== STEP 1: MODELED DATASET ==========")
    raw_df = load_old_training_df()
    modeled_df = load_or_build_modeled_df(raw_df)
    del raw_df
    gc.collect()

    print("\n[TRAIN] Class distribution:")
    print(modeled_df["target_name"].value_counts(dropna=False).to_string())

    # ── STEP 1a: Hold-out 검증 ──────────────────────────────
    print("\n---------- [1a] Hold-out validation ----------")
    split_seed = int(_time.time_ns() % (2**32 - 1))
    train_df, test_df = make_holdout_split(modeled_df, split_seed=split_seed)
    _, result_df, _ = train_and_evaluate_holdout(train_df, test_df)
    del train_df, test_df, result_df
    gc.collect()

    # ── STEP 1b: 전체 데이터로 모델 학습 (캐시 우선) ────────
    print("\n---------- [1b] Full model (cache) ----------")
    model = load_or_fit_model(modeled_df)
    del modeled_df
    gc.collect()

    # ── STEP 2: Weather ─────────────────────────────────────
    print("\n========== STEP 2: WEATHER ==========")
    weather_df = load_weather_df()

    # ── STEP 3: Inference features (캐시 우선) ───────────────
    print("\n========== STEP 3: INFERENCE FEATURES ==========")
    feats_df = load_or_build_inference_feats(weather_df)
    del weather_df
    gc.collect()

    # ── STEP 4: Threshold 스캔 ───────────────────────────────
    feats_df = scan_thresholds(model, feats_df)

    # ── STEP 5: 최종 threshold로 예측 & 저장 ─────────────────
    print("\n========== STEP 5: FINAL PREDICTION ==========")
    chosen_threshold =  0.66 #AC_PROB_THRESHOLD   # ← change this based on scan results
    print(f"Using threshold: {chosen_threshold}")

    pred_df = predict_users(model, feats_df, threshold=chosen_threshold)

    print("\nPrediction distribution:")
    print(pred_df["pred_target_name"].value_counts(dropna=False).to_string())

    output_file = save_prediction_csv(pred_df, threshold=chosen_threshold)
    plot_ac_results(pred_df.rename(columns={"pred_target_name": "label"}), OUTPUT_DIR, model=model)

    elapsed = _time.perf_counter() - total_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\nTotal runtime: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()