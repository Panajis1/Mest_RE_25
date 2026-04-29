r"""
AC Disaggregation - RE Inference
==================================
Applies the trained AC two-stage model to Romande Energie smart meter data.

Pipeline
--------
STEP 1  Load & align Dataport training data (TOT + AC sub-meter)
STEP 2  Train AcTwoStageModel (classifier + regressor)
STEP 3  Load has_ac customer IDs from Detection outputs
STEP 4  Infer time range from RE parquet files
STEP 5  Download weather data
STEP 6  Stream RE parquets -> predict ac_kw_pred -> save chunks
STEP 7  Summarise results

Usage
-----
    python ac_disaggregation_RE.py

Mirrors HP's disaggregation_RE.py structure.
"""

import sys
import gc
from pathlib import Path
from datetime import datetime

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _REPO_ROOT / "data"

for _p in [str(_SCRIPT_DIR), str(_DATA_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ac_disaggregation import (
    compute_calibration_factor,
    apply_calibration_to_chunks,
    AcTwoStageModel,
    add_sample_weights,
    build_aligned_dataset,
    get_feature_columns,
    load_ac_customer_ids,
    load_or_build_aligned,
    load_or_fit_model,
    load_raw_df,
    load_weather_15min,
    log,
    log_memory,
    save_feature_cols,
    load_feature_cols,
    stream_predictions_to_parquet,
    summarize_predictions,
    split_by_user,
    evaluate_predictions,
    print_evaluation_report,
    N_STREAM_WORKERS,
)

# ============================================================
# CONFIG  ← update these paths before running
# ============================================================

# ← update these paths before running (will be replaced by config system)
TRAIN_FILE     = _REPO_ROOT / "data" / "processed" / "training" / "all_sources_load_with_weather.parquet"

PARQUET_DIR    = _REPO_ROOT / "data" / "raw" / "re"
AC_LABEL_FILE  = _SCRIPT_DIR / "ac_detection_outputs" / "ac_labels_all_thresh66.csv"
OUTPUT_DIR     = _SCRIPT_DIR / "ac_disaggregation_outputs"

PREFIX         = "ac_disaggregation_re"


# ============================================================
# HELPERS
# ============================================================

def infer_re_time_range(parquet_dir: Path) -> tuple:
    """Scan RE parquets to find overall time range (for weather download)."""
    paths = sorted(parquet_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    dt_min = dt_max = None
    log(f"[SCAN] Scanning {len(paths)} parquet file(s) for time range...")

    for i, path in enumerate(paths, 1):
        log(f"[SCAN] {i}/{len(paths)}: {path.name}")
        df = pd.read_parquet(path, columns=["DT_UTC"])
        x  = pd.to_datetime(df["DT_UTC"], utc=True, errors="coerce").dropna()
        del df
        if x.empty:
            continue
        dt_min = x.min() if dt_min is None else min(dt_min, x.min())
        dt_max = x.max() if dt_max is None else max(dt_max, x.max())

    if dt_min is None:
        raise ValueError("Could not infer time range from RE parquets.")

    log(f"[SCAN] Time range: {dt_min} -> {dt_max}")
    return dt_min, dt_max


def save_summary(summary_df: pd.DataFrame, prefix: str) -> None:
    out_path = OUTPUT_DIR / f"{prefix}_customer_summary.csv"
    summary_df.to_csv(out_path, index=False)
    log(f"[SAVE] Customer summary -> {out_path}")

    # Portfolio-level stats
    log("\n[PORTFOLIO]")
    log(f"  Total customers with AC prediction : {len(summary_df):,}")
    log(f"  Total AC energy (kWh)              : {summary_df['ac_energy_kwh'].sum():,.0f}")
    log(f"  Mean AC energy per customer (kWh)  : {summary_df['ac_energy_kwh'].mean():,.1f}")
    log(f"  Mean p95 AC kW per customer        : {summary_df['p95_ac_kw_pred'].mean():.2f}")
    log(f"  Mean fraction timesteps AC ON      : {summary_df['frac_ac_on'].mean():.3f}")


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 60)
    log("AC Disaggregation — RE Inference")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
    log_memory("startup")

    # ── STEP 1: Load training data ───────────────────────────
    log("\n========== STEP 1: LOAD TRAINING DATA ==========")
    raw = load_raw_df(str(TRAIN_FILE))
    log_memory("after raw load")

    # ── STEP 2: Build aligned dataset (cache) ────────────────
    log("\n========== STEP 2: BUILD ALIGNED DATASET ==========")
    aligned = load_or_build_aligned(raw)
    del raw
    gc.collect()
    log_memory("after aligned dataset")

    log(f"\nAligned: {len(aligned):,} rows | {aligned['id_customer'].nunique()} users")
    log(f"AC ON fraction : {aligned['ac_on'].mean():.3f}")

    # ── STEP 3: Train model (cache) ──────────────────────────
    log("\n========== STEP 3: TRAIN MODEL ==========")

    # Hold-out validation (only if model cache doesn't exist)
    from ac_disaggregation import FORCE_REBUILD_MODEL, MODEL_CACHE
    if FORCE_REBUILD_MODEL or not MODEL_CACHE.exists():
        train_df, test_df = split_by_user(aligned)
        feature_cols = get_feature_columns(train_df)
        save_feature_cols(feature_cols)
        train_df = add_sample_weights(train_df)

        val_model = AcTwoStageModel()
        log(f"Fitting validation model on {train_df['id_customer'].nunique()} users...")
        val_model.fit(train_df, feature_cols)

        log("Evaluating on hold-out test set...")
        test_pred = val_model.predict(test_df, apply_month_gating=False)
        kw_m, on_m, by_user_df = evaluate_predictions(test_pred)
        print_evaluation_report(kw_m, on_m, by_user_df)
        del train_df, test_df, test_pred, val_model
        gc.collect()
    else:
        log("[CACHE] Skipping hold-out validation (model cache exists)")
        feature_cols = load_feature_cols()
        if feature_cols is None:
            train_df, _ = split_by_user(aligned)
            feature_cols = get_feature_columns(train_df)
            save_feature_cols(feature_cols)
            del train_df

    # Load or train full model
    model = load_or_fit_model(aligned, feature_cols)
    del aligned
    gc.collect()
    log_memory("after model training")

    # ── STEP 3: Load has_ac IDs ───────────────────────────────
    log("\n========== STEP 4: LOAD has_ac CUSTOMER IDs ==========")
    ac_ids = load_ac_customer_ids(AC_LABEL_FILE) #ac_ids = load_ac_customer_ids(AC_LABEL_DIR)
    log(f"has_ac customers: {len(ac_ids):,}")
    log_memory("after AC ID load")

    # ── STEP 4: Infer time range ──────────────────────────────
    log("\n========== STEP 5: INFER TIME RANGE ==========")
    dt_min, dt_max = infer_re_time_range(PARQUET_DIR)

    # ── STEP 5: Weather ───────────────────────────────────────
    log("\n========== STEP 6: LOAD WEATHER ==========")
    weather_15 = load_weather_15min(dt_min=dt_min, dt_max=dt_max)
    log_memory("after weather load")

    # ── STEP 6: Stream predictions ────────────────────────────
    log("\n========== STEP 7: STREAM PREDICTIONS ==========")
    chunk_dir = OUTPUT_DIR / f"{PREFIX}_chunks"
    stream_predictions_to_parquet(
        model=model,
        parquet_dir=PARQUET_DIR,
        ac_ids=ac_ids,
        weather_15=weather_15,
        output_dir=chunk_dir,
    )
    log_memory("after streaming")

    # ── STEP 7b: Calibration ─────────────────────────────────
    log("\n========== STEP 7b: CALIBRATION ==========")
    # Estimate mean ac_kw_pred from first chunk for calibration
    first_chunks = sorted(chunk_dir.glob("*.parquet"))
    if first_chunks:
        sample_df = pd.read_parquet(first_chunks[0])
        # Only use summer ON predictions for mean
        summer_on = sample_df[
            (sample_df["dt_utc"].dt.month.isin([6, 7, 8])) &
            (sample_df["ac_on_prob"] >= 0.5)
        ]
        model_mean_pred_kw = float(summer_on["ac_kw_pred"].mean()) if not summer_on.empty else 0.0
        del sample_df, summer_on
    else:
        model_mean_pred_kw = 0.0

    calib_factor = compute_calibration_factor(
        parquet_dir=PARQUET_DIR,
        ac_ids=ac_ids,
        weather_15=weather_15,
        model_mean_pred_kw=model_mean_pred_kw,
    )
    apply_calibration_to_chunks(chunk_dir, calib_factor)

    # ── STEP 8: Summarise ─────────────────────────────────────
    log("\n========== STEP 8: SUMMARISE ==========")
    summary_df = summarize_predictions(chunk_dir)
    save_summary(summary_df, PREFIX)

    log("\n[DONE] AC disaggregation complete.")


if __name__ == "__main__":
    main()