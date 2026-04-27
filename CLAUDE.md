# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package in editable mode (required before running anything)
pip install -e .

# Run all unit tests
python -m pytest tests/unit/ -v

# Run integration tests (generates synthetic data — no real parquets needed)
python -m pytest tests/integration/ -v -m integration

# Run a single test file
python -m pytest tests/unit/test_features.py -v

# Run the full detection pipeline
python scripts/run_pipeline.py --config config/re_production.yaml

# Run PV-only (faster, for iteration)
python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv

# Train detector models (requires processed training data)
python scripts/train_models.py --config config/re_production.yaml

# Export result figures
python scripts/export_figures.py --config config/re_production.yaml

# Verify package imports cleanly
python -c "import re_nilm; print(re_nilm.__version__)"
```

## Architecture

This repo implements a NILM (Non-Intrusive Load Monitoring) pipeline for Swiss smart meter data from Romande Énergie. It detects PV, AC, heat pumps, and batteries for thousands of customers from 15-min parquet files.

### Data flow

```
data/raw/re/*.parquet  +  weather (MeteoSwiss/Open-Meteo)
         ↓
  CustomerFileIndex (pipeline/customer_index.py)
         ↓
  StreamingEngine (pipeline/streaming.py)  — batched, checkpointed, resumable
         ↓
  PipelineOrchestrator (pipeline/orchestrator.py)
    1. PVDetector → pv_indicators.parquet
    2. PVCapacityEstimator → pv_capacity.parquet
    3. BatteryDetector + BatteryCapacityEstimator → battery_results.parquet
    4. ACDetector → ac_detection.parquet
    5. HeatPumpDetector → hp_detection.parquet
    6. EVDetector → ev_detection.parquet
    7. ACDisaggregationEstimator → ac_disagg_15min.parquet
    8. HPDisaggregationEstimator → hp_disagg_15min.parquet
    9. EVSessionEstimator → ev_sessions_15min.parquet
   10. Join → results_all_customers.parquet
```

### Key design decisions

- **Every step has its own checkpoint file** (`{step_name}_ckpt.parquet`). Steps do not share checkpoints so resuming one step doesn't skip another.
- **Battery detection requires PV** (configurable via `enforce_pv_required`). The battery detector receives `pv_result` via `**context`.
- **ML detectors (AC, HP) require trained models** at `models/ac_detector_v1.joblib` and `models/hp_detector_v1.joblib`. If not present, those steps are silently skipped and logged as warnings.
- **AC disaggregation model** is pre-trained and lives at `models/ac_disaggregator_v1.pkl` (note: `.pkl` extension, not `.joblib`). It was trained on sklearn 1.8.0 — if the environment has a different sklearn version, the model may need to be retrained.
- **PV detection is unsupervised** (no model artifact needed) — runs immediately.
- **HP detection** uses night-only features (`global_rad < 20 W/m²`). AC detection uses daytime features (`global_rad > 50 W/m²`). Both use the same underlying math in `re_nilm/features/load.py`.
- **EV detection is fully heuristic** (no model artifact). `EVDetector` scores sustained flat-power load blocks (≥2.8 kW, ≥90 min, std/mean ≤ 0.15) using a soft probability over 5 signals. Source: `ev_scripts/newlogic_ev.ipynb`. `EVSessionEstimator` produces per-session detail for EV-positive customers. Tune `prob_threshold` (default 0.3) if detection rate is too high/low for your data.

### Package structure

```
re_nilm/
  data/
    loaders/          — SmartMeterLoader, WeatherLoader (MeteoSwiss + Open-Meteo), DataportLoader
    preprocessing.py  — clean_and_resample, normalize_zscore
    alignment.py      — align_weather_to_meter, to_utc_naive
    schemas.py        — TypedDicts for SmartMeterRow, WeatherRow, etc.
  features/
    load.py           — shared feature functions used by all detectors
    weather.py        — daily radiation, temperature bands, CDD/HDD
    temporal.py       — calendar features, lags, rolling windows (for disaggregation models)
  detectors/          — PVDetector, ACDetector, HeatPumpDetector, BatteryDetector, EVDetector
  estimators/         — PVCapacityEstimator, BatteryCapacityEstimator, AC/HP disaggregators, EVSessionEstimator
  pipeline/
    customer_index.py — {customer_id → [parquet paths]} mapping
    streaming.py      — StreamingEngine: batched, checkpointed, parallel
    orchestrator.py   — PipelineOrchestrator: composes all detectors + estimators
  training/
    base.py           — AbstractTrainer
    trainers/         — ACDetectorTrainer, HPDetectorTrainer
  portfolio/          — aggregation, evaluation, forecasting (delegates to model/pv_detection.py)
  visualization/      — customer and portfolio plots (delegates to model/pv_detection.py)
```

### Input data schemas

- **Smart meter**: Parquet files with columns `[ID, DT_UTC, CONSO_KWH, PROD_KWH]` at 15-min resolution. DT_UTC is UTC naive.
- **Weather**: `[dt_utc, t_2m_C, global_rad_W]` at 15-min resolution, UTC naive.
- **Training data**: `all_sources_load_with_weather.parquet` — unified table with per-customer features and appliance labels from Dataport.

### Legacy code (model/, hp_model/, data/)

The original detection scripts still live in `model/`, `hp_model/`, and `data/`. The `re_nilm` package delegates to them rather than duplicating. Full migration is a future task. Do not modify these files without checking what `re_nilm` imports from them.

### Python compatibility

The repo targets Python 3.9+. All files in `re_nilm/` must include `from __future__ import annotations` at the top to support union type syntax (`X | Y`, `dict[str, ...]`) on Python 3.9.

### Config

All parameters live in `config/default.yaml` with production overrides in `config/re_production.yaml`. The orchestrator merges them with `_deep_merge`. Paths in config are relative to the repo root.
