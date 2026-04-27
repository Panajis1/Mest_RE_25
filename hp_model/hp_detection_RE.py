print("Starting batch processing...")
import sys
from pathlib import Path
import gc
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HP_MODEL_DIR = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"

for _p in [str(_HP_MODEL_DIR), str(_DATA_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hp_detection_functions import (
    RF_PARAMS,
    TEMP_F_THRESHOLD,
    INT_TO_LABEL,
    FEATURE_COLS,
    convert_f_to_c_if_needed,
    build_modeled_dataset,
    load_and_model_csv_winter_hp,
    tot_curve_is_usable,
    extract_tot_features,
)

from envdata import env_data


# ============================================================
# CONFIG  ← update these paths before running (will be replaced by config system)
# ============================================================

TRAIN_FILE = _REPO_ROOT / "data" / "processed" / "training" / "all_sources_load_with_weather.parquet"

PARQUET_DIR = _REPO_ROOT / "data" / "raw" / "re"
OUTPUT_DIR = _HP_MODEL_DIR / "hp_detection_outputs"
METADATA_FILE = PARQUET_DIR / "metadata"

KWH_TO_KW_FACTOR = 4.0
MAX_ANNUAL_CONSUMPTION_KWH = 100_000.0
NEW_SOURCE_NAME = "new_parquet"

def load_particuliers_ids(metadata_file: Path) -> set[str]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"\n[METADATA] Reading: {metadata_file}")
    df_meta = pd.read_parquet(metadata_file, columns=["ID", "TYPE_PARTENAIRE_LIBELLE"])

    required_cols = ["ID", "TYPE_PARTENAIRE_LIBELLE"]
    missing = [c for c in required_cols if c not in df_meta.columns]
    if missing:
        raise ValueError(f"Required columns are missing in the metadata file: {missing}")

    df_meta = df_meta.copy()
    df_meta["ID"] = df_meta["ID"].astype(str)

    particuliers_ids = set(
        df_meta.loc[
            df_meta["TYPE_PARTENAIRE_LIBELLE"] == "Particuliers",
            "ID"
        ]
    )

    print(f"[METADATA] Total IDs in metadata: {df_meta['ID'].nunique()}")
    print(f"[METADATA] Particuliers IDs: {len(particuliers_ids)}")

    if not particuliers_ids:
        raise ValueError("No 'Particuliers' IDs found in metadata.")

    return particuliers_ids
# ============================================================
# TRAINING ON THE OLD DATASET
# ============================================================


def load_old_training_df(weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    print(f"\n[TRAIN] Reading: {TRAIN_FILE}")
    df = pd.read_parquet(TRAIN_FILE)

    needed_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing in the training file: {missing}")

    df = df[needed_cols].copy()
    df = df[df["type"].isin(["HP", "TOT"])].copy()

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])

    df = convert_f_to_c_if_needed(df, threshold_f=TEMP_F_THRESHOLD)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    modeled_df = build_modeled_dataset(df)

    csv_modeled_df = load_and_model_csv_winter_hp(BASE_DIR, weather_df=weather_df)
    if not csv_modeled_df.empty:
        modeled_df = pd.concat([modeled_df, csv_modeled_df], ignore_index=True)

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

    print("\n[TRAIN] Fitting model on the full labeled old dataset...")
    model.fit(X, y)
    print("[TRAIN] Fit completed.")

    rf_model = model.named_steps["rf"]
    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n[TRAIN] Feature importance:")
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

    # Correct reconstruction of the mean by timestamp
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

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"], utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")
    weather = weather.dropna(subset=["timestamp"]).copy()

    weather = weather.rename(columns={
        "timestamp": "dt_utc",
        "t_2m_C": "temp",
        "global_rad_W": "glob_rad",
    })

    weather = weather[["dt_utc", "temp", "glob_rad"]].copy()
    weather["temp"] = pd.to_numeric(weather["temp"], errors="coerce")
    weather["glob_rad"] = pd.to_numeric(weather["glob_rad"], errors="coerce")

    weather = (
        weather
        .sort_values("dt_utc")
        .drop_duplicates("dt_utc", keep="first")
        .reset_index(drop=True)
    )

    print(f"[WEATHER] Original weather rows: {len(weather)}")
    print(f"[WEATHER] Original range: {weather['dt_utc'].min()} -> {weather['dt_utc'].max()}")

    # --------------------------------------------------------
    # RESAMPLING TO 15 MINUTES
    # If the exact value at :15 or :45 is missing, use the mean
    # of the values 5 minutes before and 5 minutes after through
    # time interpolation. With 10-minute data this is equivalent
    # to the mean.
    # --------------------------------------------------------
    weather_15 = weather.set_index("dt_utc").sort_index()

    # Keep only useful numeric columns
    weather_15 = weather_15[["temp", "glob_rad"]]

    # Create a 15-minute grid over the full interval
    weather_15 = weather_15.resample("15min").interpolate(method="time")

    # Restore dt_utc as a column
    weather_15 = weather_15.reset_index()

    print(f"[WEATHER] Resampled 15-minute weather rows: {len(weather_15)}")
    print(f"[WEATHER] 15-minute range: {weather_15['dt_utc'].min()} -> {weather_15['dt_utc'].max()}")

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



def load_new_parquet_with_weather(
    parquet_file: Path,
    weather_df: pd.DataFrame,
    particuliers_ids: set[str]
) -> pd.DataFrame:
    if not parquet_file.exists():
        raise FileNotFoundError(f"New parquet file not found: {parquet_file}")

    print(f"\n[NEW] Reading: {parquet_file}")
    df_new = pd.read_parquet(parquet_file, columns=["ID", "DT_UTC", "CONSO_KWH"])

    required_cols = ["ID", "DT_UTC", "CONSO_KWH"]
    missing = [c for c in required_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"Required columns are missing in the new parquet file: {missing}")

    df_new = df_new.copy()
    df_new["ID"] = df_new["ID"].astype(str)
    df_new["DT_UTC"] = pd.to_datetime(
        df_new["DT_UTC"], utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")
    df_new["CONSO_KWH"] = pd.to_numeric(df_new["CONSO_KWH"], errors="coerce")

    df_new = df_new.dropna(subset=["ID", "DT_UTC", "CONSO_KWH"]).copy()

    total_users_before_metadata = df_new["ID"].nunique()

    df_new = df_new[df_new["ID"].isin(particuliers_ids)].copy()

    total_users_after_metadata = df_new["ID"].nunique()

    print(f"\n[NEW] Total users before metadata filter: {total_users_before_metadata}")
    print(f"[NEW] Kept users (Particuliers only): {total_users_after_metadata}")
    print(f"[NEW] Discarded users (non-Particuliers): {total_users_before_metadata - total_users_after_metadata}")

    annual_kwh = (
        df_new.groupby("ID", dropna=False)["CONSO_KWH"]
        .sum(min_count=1)
        .rename("annual_consumption_kwh")
        .reset_index()
    )

    valid_ids = set(
        annual_kwh.loc[
            annual_kwh["annual_consumption_kwh"] <= MAX_ANNUAL_CONSUMPTION_KWH,
            "ID"
        ].astype(str)
    )

    total_users = df_new["ID"].nunique()
    kept_users = len(valid_ids)

    print(f"\n[NEW] Total users: {total_users}")
    print(f"[NEW] Kept users (annual consumption <= 100 MWh): {kept_users}")
    print(f"[NEW] Discarded users (annual consumption > 100 MWh): {total_users - kept_users}")

    df_new = df_new[df_new["ID"].isin(valid_ids)].copy()

    if df_new.empty:
        raise ValueError("No users remain after the annual consumption filter.")

    df_new["value_kw_mean"] = df_new["CONSO_KWH"] * KWH_TO_KW_FACTOR

    df_new = df_new.rename(columns={
        "ID": "id_customer",
        "DT_UTC": "dt_utc",
    })

    df_new["type"] = "TOT"
    df_new["source"] = NEW_SOURCE_NAME

    df_new = df_new.sort_values("dt_utc").reset_index(drop=True)
    weather_df = weather_df.sort_values("dt_utc").reset_index(drop=True)

    # Exact merge: the weather data has already been resampled to 15 minutes
    df_new = df_new.merge(weather_df, on="dt_utc", how="left")

    final_cols = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df_new = df_new[final_cols].copy()

    df_new["glob_rad"] = pd.to_numeric(df_new["glob_rad"], errors="coerce")
    df_new["temp"] = pd.to_numeric(df_new["temp"], errors="coerce")
    df_new["value_kw_mean"] = pd.to_numeric(df_new["value_kw_mean"], errors="coerce")

    print(f"\n[NEW] Shape after weather merge: {df_new.shape}")
    print(f"[NEW] Valid temp values: {df_new['temp'].notna().sum()}")
    print(f"[NEW] Valid glob_rad values: {df_new['glob_rad'].notna().sum()}")

    missing_weather = df_new["temp"].isna().sum()
    print(f"[NEW] Rows without matched weather data: {missing_weather}")

    return df_new


# ============================================================
# FEATURE EXTRACTION INFERENCE
# ============================================================


def build_inference_features(df_new_tot: pd.DataFrame) -> pd.DataFrame:
    records = []
    discard_reasons = {}

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

    if not feats_df.empty and "used_night_only" in feats_df.columns:
        print("[NEW] Feature mode summary:")
        print(
            feats_df["used_night_only"]
            .value_counts(dropna=False)
            .rename(index={1: "night_only", 0: "full_curve_fallback"})
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

    extra_cols = []
    for c in ["used_night_only", "n_night_rows", "n_feature_rows"]:
        if c in feats_df.columns:
            extra_cols.append(c)

    out = feats_df[["id_customer", "source"] + extra_cols + FEATURE_COLS].copy()
    out["pred_target"] = pred
    out["pred_target_name"] = out["pred_target"].map(INT_TO_LABEL)
    out["pred_is_hp"] = out["pred_target_name"].isin(["winter_hp", "summer_hp"]).astype(int)

    for class_id, class_name in INT_TO_LABEL.items():
        col_name = f"prob_{class_name}"
        if class_id in classes:
            idx = int(np.where(classes == class_id)[0][0])
            out[col_name] = proba[:, idx]
        else:
            out[col_name] = 0.0

    out["prob_hp_total"] = out["prob_winter_hp"] + out["prob_summer_hp"]

    return out.sort_values(
        ["pred_is_hp", "prob_hp_total", "id_customer"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


# ============================================================
# SAVE OUTPUT
# ============================================================


def save_prediction_csv(pred_df: pd.DataFrame, parquet_file: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{parquet_file.stem}_hp_labels.csv"
    export_df = pred_df[["id_customer", "pred_target_name"]].rename(
        columns={"pred_target_name": "label"}
    )
    export_df.to_csv(output_file, index=False)

    print(f"\n[OUTPUT] CSV saved to: {output_file}")
    return output_file


# ============================================================
# PROCESS ONE PARQUET
# ============================================================


def process_single_parquet(
    model,
    weather_df: pd.DataFrame,
    parquet_file: Path,
    particuliers_ids: set[str]
) -> Path:
    print("\n============================================================")
    print(f"Processing parquet file: {parquet_file.name}")
    print("============================================================")

    df_new_tot = load_new_parquet_with_weather(parquet_file, weather_df, particuliers_ids)

    print("\n========== STEP 4: FEATURE EXTRACTION ==========")
    feats_df = build_inference_features(df_new_tot)

    print("\n========== STEP 5: PREDICTION ==========")
    pred_df = predict_users(model, feats_df)

    print("\nPrediction distribution:")
    print(pred_df["pred_target_name"].value_counts(dropna=False).to_string())

    print("\nFirst 20 predictions:")
    cols_to_show = [
        "id_customer",
        "pred_target_name",
        "pred_is_hp",
        "prob_no_hp",
        "prob_winter_hp",
        "prob_summer_hp",
        "prob_hp_total",
    ]
    extra_cols = [c for c in ["used_night_only", "n_night_rows", "n_feature_rows"] if c in pred_df.columns]
    print(pred_df[["id_customer"] + extra_cols + cols_to_show[1:]].head(20).to_string(index=False))

    hp_pred = pred_df[pred_df["pred_is_hp"] == 1].copy()

    print("\nNumber of users predicted as HP:", len(hp_pred))
    if not hp_pred.empty:
        cols = ["id_customer", "pred_target_name", "prob_winter_hp", "prob_summer_hp", "prob_hp_total"]
        extra_cols = [c for c in ["used_night_only", "n_night_rows", "n_feature_rows"] if c in hp_pred.columns]
        print("\nFirst 20 users predicted as HP:")
        print(hp_pred[["id_customer"] + extra_cols + cols[1:]].head(20).to_string(index=False))

    output_file = save_prediction_csv(pred_df, parquet_file)

    del df_new_tot, feats_df, pred_df, hp_pred
    gc.collect()

    return output_file


# ============================================================
# MAIN
# ============================================================


def main():
    print("\n========== STEP 1: WEATHER FOR TRAINING CSV AND NEW PARQUETS ==========")
    weather_df = load_weather_df()

    print("\n========== STEP 2: TRAIN ON all_sources_load_with_weather + CSV winter HP ==========")
    modeled_df = load_old_training_df(weather_df=weather_df)
    model = fit_model(modeled_df)

    del modeled_df
    gc.collect()

    print("\n========== STEP 3: LOAD METADATA ==========")
    particuliers_ids = load_particuliers_ids(METADATA_FILE)

    print("\n========== STEP 4: PARQUET FILE DISCOVERY ==========")
    parquet_files = list_input_parquet_files()
    print(f"Found {len(parquet_files)} parquet files to process.")
    print(f"Output directory: {OUTPUT_DIR}")

    saved_files = []
    failed_files = []

    for i, parquet_file in enumerate(parquet_files, start=1):
        print(f"\n[{i}/{len(parquet_files)}] Starting {parquet_file.name}")
        try:
            output_file = process_single_parquet(model, weather_df, parquet_file, particuliers_ids)
            saved_files.append(output_file)
        except Exception as exc:
            failed_files.append((parquet_file.name, str(exc)))
            print(f"\n[ERROR] Failed on {parquet_file.name}: {exc}")
        finally:
            gc.collect()

    print("\n============================================================")
    print("FINAL SUMMARY")
    print("============================================================")
    print(f"Successfully processed files: {len(saved_files)}")
    print(f"Failed files: {len(failed_files)}")

    if saved_files:
        print("\nSaved CSV files:")
        for p in saved_files:
            print(f"- {p}")

    if failed_files:
        print("\nFiles with errors:")
        for file_name, error_msg in failed_files:
            print(f"- {file_name}: {error_msg}")


if __name__ == "__main__":
    main()
