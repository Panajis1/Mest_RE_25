"""
AC Detection: Batch Processing
Mirror of batch_processing.py but for AC (air conditioner) detection.

Steps:
  1. Train AC classifier on all_sources_load_with_weather.parquet (Dataport AC ground truth)
  2. Download weather for new parquet files
  3. Discover parquet files in PARQUET_DIR
  4. For each parquet: merge weather → extract features → predict → save CSV
"""
print("Starting AC batch processing...")

from pathlib import Path
import gc
import numpy as np
import pandas as pd

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

KWH_TO_KW_FACTOR = 4.0
MAX_ANNUAL_CONSUMPTION_KWH = 10_000.0
NEW_SOURCE_NAME = "new_parquet"


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

    # Keep AC rows (for labels) and TOT rows (for features)
    df = df[df["type"].isin(["AC", "TOT"])].copy()

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])
    df = convert_f_to_c_if_needed(df, threshold_f=TEMP_F_THRESHOLD)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    modeled_df = build_modeled_dataset(df)

    if modeled_df.empty:
        raise ValueError("The modeled training dataset is empty.")

    print("\n[TRAIN] Class distribution:")
    print(modeled_df["target_name"].value_counts(dropna=False).to_string())

    return modeled_df


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
# WEATHER  (identical to HP batch)
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

    print(f"[WEATHER] Original rows: {len(weather)}")

    # Resample to 15 minutes via time interpolation
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

    # Filter by annual consumption
    annual_kwh = (
        df_new.groupby("ID", dropna=False)["CONSO_KWH"]
        .sum(min_count=1)
        .rename("annual_consumption_kwh")
        .reset_index()
    )
    valid_ids = set(
        annual_kwh.loc[annual_kwh["annual_consumption_kwh"] <= MAX_ANNUAL_CONSUMPTION_KWH, "ID"].astype(str)
    )
    total_users = df_new["ID"].nunique()
    print(f"\n[NEW] Total users: {total_users}")
    print(f"[NEW] Kept (≤10 MWh/year): {len(valid_ids)}")
    print(f"[NEW] Discarded (>10 MWh/year): {total_users - len(valid_ids)}")

    df_new = df_new[df_new["ID"].isin(valid_ids)].copy()
    if df_new.empty:
        raise ValueError("No users remain after annual consumption filter.")

    df_new["value_kw_mean"] = df_new["CONSO_KWH"] * KWH_TO_KW_FACTOR
    df_new = df_new.rename(columns={"ID": "id_customer", "DT_UTC": "dt_utc"})
    df_new["type"] = "TOT"
    df_new["source"] = NEW_SOURCE_NAME

    # Exact merge with 15-min weather
    df_new = df_new.sort_values("dt_utc").reset_index(drop=True)
    weather_df = weather_df.sort_values("dt_utc").reset_index(drop=True)
    df_new = df_new.merge(weather_df, on="dt_utc", how="left")

    final_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df_new = df_new[final_cols].copy()
    for col in ["glob_rad", "temp", "value_kw_mean"]:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    print(f"\n[NEW] Shape after weather merge: {df_new.shape}")
    print(f"[NEW] Valid temp values: {df_new['temp'].notna().sum()}")
    print(f"[NEW] Valid glob_rad values: {df_new['glob_rad'].notna().sum()}")
    print(f"[NEW] Rows without matched weather: {df_new['temp'].isna().sum()}")

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

    if not feats_df.empty and "used_daytime_only" in feats_df.columns:
        print("[NEW] Feature mode summary:")
        print(
            feats_df["used_daytime_only"]
            .value_counts(dropna=False)
            .rename(index={1: "daytime_only", 0: "full_curve_fallback"})
            .to_string()
        )

    if discard_reasons:
        print("[NEW] Discard reasons:")
        for k, v in sorted(discard_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {v}")

    if feats_df.empty:
        raise ValueError("No valid users remain for inference after TOT filters.")

    return feats_df


# ============================================================
# PREDICTION
# ============================================================

def predict_users(model, feats_df: pd.DataFrame) -> pd.DataFrame:
    X_new = feats_df[FEATURE_COLS].copy()

    pred = model.predict(X_new)
    proba = model.predict_proba(X_new)
    classes = model.named_steps["rf"].classes_

    extra_cols = [c for c in ["used_daytime_only", "n_day_rows", "n_feature_rows"] if c in feats_df.columns]
    out = feats_df[["id_customer", "source"] + extra_cols + FEATURE_COLS].copy()
    out["pred_target"] = pred
    out["pred_target_name"] = out["pred_target"].map(INT_TO_LABEL)

    # 임계값 적용: prob_has_ac > AC_PROB_THRESHOLD 일 때만 has_ac
    # (기본 0.5보다 높게 설정해 false positive 줄임)
    for class_id, class_name in INT_TO_LABEL.items():
        col_name = f"prob_{class_name}"
        if class_id in classes:
            idx = int(np.where(classes == class_id)[0][0])
            out[col_name] = proba[:, idx]
        else:
            out[col_name] = 0.0

    out["pred_has_ac"] = (out["prob_has_ac"] >= AC_PROB_THRESHOLD).astype(int)
    out["pred_target_name"] = out["pred_has_ac"].map({1: "has_ac", 0: "no_ac"})

    return out.sort_values(
        ["pred_has_ac", "prob_has_ac", "id_customer"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_prediction_csv(pred_df: pd.DataFrame, parquet_file: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{parquet_file.stem}_ac_labels.csv"
    # prob_has_ac도 같이 저장 (시각화용)
    cols = ["id_customer", "pred_target_name"]
    if "prob_has_ac" in pred_df.columns:
        cols.append("prob_has_ac")
    export_df = pred_df[cols].rename(columns={"pred_target_name": "label"})
    export_df.to_csv(output_file, index=False)
    print(f"\n[OUTPUT] CSV saved to: {output_file}")
    return output_file


# ============================================================
# PROCESS ONE PARQUET
# ============================================================

def process_single_parquet(model, weather_df: pd.DataFrame, parquet_file: Path) -> Path:
    print("\n============================================================")
    print(f"Processing: {parquet_file.name}")
    print("============================================================")

    df_new_tot = load_new_parquet_with_weather(parquet_file, weather_df)

    print("\n========== STEP 4: FEATURE EXTRACTION ==========")
    feats_df = build_inference_features(df_new_tot)

    print("\n========== STEP 5: PREDICTION ==========")
    pred_df = predict_users(model, feats_df)

    print("\nPrediction distribution:")
    print(pred_df["pred_target_name"].value_counts(dropna=False).to_string())

    ac_pred = pred_df[pred_df["pred_has_ac"] == 1].copy()
    print(f"\nUsers predicted as AC: {len(ac_pred)}")
    if not ac_pred.empty:
        extra_cols = [c for c in ["used_daytime_only", "n_day_rows", "n_feature_rows"] if c in ac_pred.columns]
        print(ac_pred[["id_customer"] + extra_cols + ["pred_target_name", "prob_has_ac"]].head(20).to_string(index=False))

    output_file = save_prediction_csv(pred_df, parquet_file)

    del df_new_tot, feats_df, pred_df, ac_pred
    gc.collect()

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== STEP 1: TRAIN AC MODEL ==========")
    modeled_df = load_old_training_df()

    # ── Hold-out 검증 (성능 확인) ──────────────────────────
    print("\n---------- [1a] Hold-out validation ----------")
    import time as _time
    split_seed = int(_time.time_ns() % (2**32 - 1))
    train_df, test_df = make_holdout_split(modeled_df, split_seed=split_seed)
    _, result_df, _ = train_and_evaluate_holdout(train_df, test_df)
    print("\n[Hold-out] Validation complete. Now training on full dataset...")

    # ── 전체 데이터로 재학습 (RE 추론용) ───────────────────
    print("\n---------- [1b] Full training ----------")
    model = fit_model(modeled_df)
    del modeled_df, train_df, test_df, result_df
    gc.collect()

    print("\n========== STEP 2: WEATHER ==========")
    weather_df = load_weather_df()

    print("\n========== STEP 3: PARQUET FILE DISCOVERY ==========")
    parquet_files = list_input_parquet_files()
    print(f"Found {len(parquet_files)} parquet files.")
    print(f"Output directory: {OUTPUT_DIR}")

    saved_files = []
    failed_files = []

    for i, parquet_file in enumerate(parquet_files, start=1):
        print(f"\n[{i}/{len(parquet_files)}] Starting {parquet_file.name}")
        try:
            output_file = process_single_parquet(model, weather_df, parquet_file)
            saved_files.append(output_file)
        except Exception as exc:
            failed_files.append((parquet_file.name, str(exc)))
            print(f"\n[ERROR] Failed on {parquet_file.name}: {exc}")
        finally:
            gc.collect()

    print("\n============================================================")
    print("FINAL SUMMARY")
    print("============================================================")
    print(f"Successfully processed: {len(saved_files)}")
    print(f"Failed: {len(failed_files)}")

    if saved_files:
        print("\nSaved CSV files:")
        for p in saved_files:
            print(f"  - {p}")

        # 모든 CSV를 하나로 합치기
        print("\n========== MERGING ALL CSVs ==========")
        merged_parts = []
        for p in saved_files:
            part = pd.read_csv(p)
            part["source_file"] = p.stem
            merged_parts.append(part)

        merged_df = pd.concat(merged_parts, ignore_index=True)
        merged_path = OUTPUT_DIR / "ac_labels_all.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Merged CSV saved to: {merged_path}")
        print(f"Total users in merged file: {len(merged_df)}")
        print("\nLabel distribution:")
        print(merged_df["label"].value_counts().to_string())

        print("\n========== VISUALISATION ==========")
        plot_ac_results(merged_df, OUTPUT_DIR, model=model)

    if failed_files:
        print("\nFiles with errors:")
        for name, err in failed_files:
            print(f"  - {name}: {err}")




# ============================================================
# VISUALISATION
# ============================================================

def plot_ac_results(merged_df: pd.DataFrame, output_dir: Path, model=None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 공통 스타일 ──────────────────────────────────────────
    COLORS = {"has_ac": "#E8634A", "no_ac": "#4A90D9"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("AC Detection Results — RE Dataset", fontsize=15, fontweight="bold", y=1.02)

    # ── 1. 파이 차트: has_ac / no_ac 전체 비율 ───────────────
    ax1 = axes[0]
    counts = merged_df["label"].value_counts()
    labels = counts.index.tolist()
    colors = [COLORS.get(l, "#aaa") for l in labels]
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax1.set_title(f"Overall AC ratio\n(n={len(merged_df):,} users)", fontsize=12)

    # ── 2. 히스토그램: prob_has_ac 분포 ─────────────────────
    ax2 = axes[1]
    if "prob_has_ac" in merged_df.columns:
        for label, grp in merged_df.groupby("label"):
            ax2.hist(
                grp["prob_has_ac"].dropna(),
                bins=30,
                alpha=0.65,
                color=COLORS.get(label, "#aaa"),
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )
        ax2.set_xlabel("prob_has_ac")
        ax2.set_ylabel("Number of users")
        ax2.set_title("Distribution of AC probability score")
        ax2.legend()
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    else:
        ax2.text(0.5, 0.5, "prob_has_ac column\nnot found", ha="center", va="center")
        ax2.set_title("AC probability distribution")

    # ── 3. 피처 중요도 ──────────────────────────────────────
    ax3 = axes[2]
    if model is not None:
        rf_model = model.named_steps["rf"]
        importances = pd.Series(
            rf_model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=True)
        bars = ax3.barh(
            importances.index,
            importances.values,
            color="#4A90D9",
            edgecolor="white",
            linewidth=0.8,
        )
        ax3.set_xlabel("Importance")
        ax3.set_title("Feature importances (RandomForest)")
        for bar, val in zip(bars, importances.values):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "model not provided", ha="center", va="center")
        ax3.set_title("Feature importances")

    plt.tight_layout()
    plot_path = output_dir / "ac_detection_results.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved to: {plot_path}")

if __name__ == "__main__":
    main()