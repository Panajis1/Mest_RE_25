# re_nilm — Romande Énergie Appliance Detection Pipeline

Non-intrusive load monitoring for Swiss smart meter data. Detects PV systems, batteries, EV chargers, air-conditioning, and heat pumps from 15-minute parquet files without sub-metering.

## Detectors

| Detector | Signal | Output columns | Model |
|---|---|---|---|
| **PV** | Production–radiation correlation | `has_pv`, `prob_pv`, `yearly_prod_kwh` | Heuristic |
| **PV Capacity** | Bootstrap regression on PV slope | `pv_capacity_kwp`, `pv_ci_lower/upper`, `sc_share` | Heuristic |
| **Battery** | Load shift on dark vs sunny days | `has_battery`, `prob_battery`, `battery_capacity_kwh` | Heuristic |
| **AC** | Summer daytime load–temperature correlation | `has_ac`, `prob_ac` | Random Forest (requires training) |
| **Heat Pump** | Winter night load–temperature correlation | `has_hp`, `prob_hp`, `hp_type` | Random Forest (requires training) |
| **EV** | Sustained flat-power load blocks (≥ 2.8 kW, ≥ 90 min) | `has_ev`, `prob_ev`, `ev_total_sessions` | Heuristic |

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Point the config at your data
#    Edit config/re_production.yaml → data.re_data_dir

# 3. Run PV + battery + EV (no trained models needed)
python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv,battery,ev

# 4. Full pipeline (requires trained AC/HP models — see Training below)
python scripts/run_pipeline.py --config config/re_production.yaml
```

Results land in `data/processed/out/results_all_customers.parquet`.

## Data Requirements

**Smart meter files**: One or more parquet files in a directory. Each file may contain multiple customers.

| Column | Type | Description |
|---|---|---|
| `ID` | str | Customer identifier |
| `DT_UTC` | datetime | Timestamp, UTC naive |
| `CONSO_KWH` | float | Grid import energy per 15-min interval (kWh) |
| `PROD_KWH` | float | Grid export energy per 15-min interval (kWh) |

**Weather**: Fetched automatically from MeteoSwiss public API (no key required) on first run and cached to `data/processed/out/weather_cache.parquet`. Alternatively supply your own `[dt_utc, t_2m_C, global_rad_W]` parquet.

## Configuration

All parameters live in `config/default.yaml`. Override per-deployment in `config/re_production.yaml` — only set what differs.

```yaml
data:
  re_data_dir: "data/re_data/ETHZ_ALL"   # directory of customer parquets

pipeline:
  n_workers: 8           # parallel workers (set to 1 for debugging)
  batch_size: 500        # customers per batch
  resume_from_checkpoint: true

detectors:
  enabled: [pv, battery, ac, heat_pump, ev]

models:
  pv:
    corr_threshold: 0.3
    min_yearly_prod_kwh: 1.0
    capacity_bootstrap_n: 200
  ev:
    prob_threshold: 0.3    # lower = more sensitive; raise to 0.5 if too many false positives
    grid_threshold_kw: 2.8
  # ac / heat_pump: see default.yaml for full parameter list
```

## Running the Pipeline

```bash
# PV only (fastest, no ML models needed)
python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv

# PV + EV (both heuristic, no models needed)
python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv,battery,ev

# Full pipeline
python scripts/run_pipeline.py --config config/re_production.yaml

# Re-run from scratch (ignore existing checkpoints)
python scripts/run_pipeline.py --config config/re_production.yaml --no-resume
```

Each detector step saves its own checkpoint. If the run is interrupted, re-running resumes from where it stopped.

## Training AC and HP Models

AC and HP detection require a Random Forest trained on labelled Dataport households. The training dataset (`all_sources_load_with_weather.parquet`) must contain per-customer feature columns and labels (`has_ac`, `hp_type`).

```bash
python scripts/train_models.py \
    --config config/re_production.yaml \
    --training-data data/processed_data/all_sources_load_with_weather.parquet \
    --models ac_detector,hp_detector
```

Trained models are saved to `models/ac_detector_v1.joblib` and `models/hp_detector_v1.joblib`. Once present, the pipeline picks them up automatically.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/re_nilm_demo.ipynb` | Full walkthrough: single customer → small sample → full portfolio |
| `notebooks/example_single_customer.ipynb` | Minimal single-customer example |

## Project Structure

```
re_nilm/                  # installable package
  detectors/              # PVDetector, BatteryDetector, ACDetector, HeatPumpDetector, EVDetector
  estimators/             # PVCapacityEstimator, BatteryCapacityEstimator, EVSessionEstimator, ...
  pipeline/
    orchestrator.py       # composes all detectors, drives the full run
    streaming.py          # batched, checkpointed, resumable engine
    customer_index.py     # {customer_id → parquet paths} mapping
  features/               # shared feature functions (load, weather, temporal)
  data/loaders/           # SmartMeterLoader, WeatherLoader, DataportLoader
  training/               # ACDetectorTrainer, HPDetectorTrainer
  portfolio/              # aggregation, evaluation, forecasting
  visualization/          # customer and portfolio plots

config/
  default.yaml            # all default parameters
  re_production.yaml      # Romande Énergie overrides

scripts/
  run_pipeline.py         # CLI entry point
  train_models.py         # train AC / HP detectors
  export_figures.py       # export result plots

models/                   # serialized model artifacts (gitignored)
data/                     # raw and processed data (gitignored)
tests/                    # unit + integration tests
```

## Running Tests

```bash
pip install -e .
python -m pytest tests/unit/ -v           # 58 tests, synthetic data only
python -m pytest tests/integration/ -v -m integration   # end-to-end, no real data needed
```
