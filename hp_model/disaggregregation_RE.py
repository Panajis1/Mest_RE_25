
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from disaggregation_functions import (
    ScientificTwoStageHPModel,
    add_sample_weights,
    build_aligned_dataset,
    get_feature_columns,
    load_external_weather_15min,
    load_raw_df,
    load_winter_hp_ids,
    log,
    log_memory,
    save_external_outputs_from_chunks,
    stream_external_predictions_to_parquet,
    summarize_external_prediction_chunks,
)

BASE_DIR = Path(r"C:\Users\Master\Documents\Mest_RE_25\hp_label")
PV_FORECAST_DIR = BASE_DIR / "pv_forecast"
HP_LABEL_DIR = BASE_DIR / "hp_detection_outputs"
PREFIX = "internaldisaggregation_scidata_to_external"

def infer_external_time_range(pv_forecast_dir: Path):
    parquet_paths = sorted(pv_forecast_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {pv_forecast_dir}")

    dt_min = None
    dt_max = None
    log(f"[MAIN] Found {len(parquet_paths)} parquet files for time-range scan")
    log_memory("before external time-range scan")
    for i, path in enumerate(parquet_paths, start=1):
        log(f"[SCAN] Time range file {i}/{len(parquet_paths)}: {path.name}")
        df = pd.read_parquet(path, columns=["DT_UTC"])
        x = pd.to_datetime(df["DT_UTC"], utc=True, errors="coerce").dropna()
        del df
        if x.empty:
            log(f"[SCAN] Empty/invalid timestamps: {path.name}")
            continue
        cur_min = x.min()
        cur_max = x.max()
        log(f"[SCAN] {path.name} -> {cur_min} | {cur_max}")
        dt_min = cur_min if dt_min is None else min(dt_min, cur_min)
        dt_max = cur_max if dt_max is None else max(dt_max, cur_max)
        if i == 1 or i % 10 == 0 or i == len(parquet_paths):
            log_memory(f"after time-range file {i}/{len(parquet_paths)}")
    if dt_min is None or dt_max is None:
        raise ValueError("Could not infer time range from external parquet files.")
    return dt_min, dt_max


def main():
    log("[MAIN] Starting run...")
    log(f"[MAIN] BASE_DIR = {BASE_DIR}")
    log(f"[MAIN] PV_FORECAST_DIR = {PV_FORECAST_DIR}")
    log(f"[MAIN] HP_LABEL_DIR = {HP_LABEL_DIR}")
    log_memory("startup")
    # Train only on SCIENTIFIC DATA from internal labeled dataset
    log("[MAIN] Loading internal training data...")
    df = load_raw_df()
    log_memory("after internal raw load")
    log("[MAIN] Building aligned dataset...")
    aligned_df = build_aligned_dataset(df)
    log_memory("after aligned dataset build")
    train_df = aligned_df[aligned_df["source_name"] == "SCIENTIFIC DATA"].copy()

    feature_cols = get_feature_columns(aligned_df)
    train_df = add_sample_weights(train_df)
    log("[MAIN] Training model...")
    model = ScientificTwoStageHPModel().fit(train_df, feature_cols)
    log_memory("after model training")

    # External inference
    winter_ids = load_winter_hp_ids(HP_LABEL_DIR)
    log(f"[MAIN] winter_hp IDs loaded: {len(winter_ids):,}")
    log_memory("after winter ID load")
    log("[MAIN] Inferring external time range...")
    dt_min, dt_max = infer_external_time_range(PV_FORECAST_DIR)
    log(f"[MAIN] External time range: {dt_min} -> {dt_max}")
    log("[MAIN] Loading external weather...")
    weather_15 = load_external_weather_15min(dt_min=dt_min, dt_max=dt_max)
    log_memory("after external weather load")
    chunk_dir = BASE_DIR / f"{PREFIX}_prediction_chunks"
    log(f"[MAIN] Streaming external predictions to Parquet chunks in: {chunk_dir}")
    log_memory("before streaming external predictions")
    stream_external_predictions_to_parquet(
        model=model,
        pv_forecast_dir=PV_FORECAST_DIR,
        winter_ids=winter_ids,
        weather_15=weather_15,
        output_dir=chunk_dir,
    )
    log_memory("after streaming external predictions")

    log("[MAIN] Computing sanity summaries from Parquet chunks...")
    global_summary, by_profile, monthly = summarize_external_prediction_chunks(chunk_dir)
    log_memory("after chunk summarization")

    log("\n[MODEL] Scientific-only model applied to external dataset.")
    log(f"[MODEL] Feature count: {len(feature_cols)}")

    log("\n[EXTERNAL SANITY - GLOBAL]")
    log(global_summary.to_string(index=False))

    log("\n[EXTERNAL SANITY - PROFILE CLASS]")
    log(by_profile.to_string(index=False))

    log("\n[EXTERNAL SANITY - MONTHLY HEAD]")
    log(monthly.head(12).to_string(index=False))

    save_external_outputs_from_chunks(PREFIX, chunk_dir, global_summary, by_profile, monthly)


if __name__ == "__main__":
    main()
