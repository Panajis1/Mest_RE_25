# re_nilm - Romande Energie Appliance Detection Pipeline

`re_nilm` is an installable Python package for non-intrusive load monitoring
on Swiss 15-minute smart-meter data. It detects PV systems, batteries, EV
charging, air-conditioning, and heat pumps without sub-metering, then joins the
portfolio-level scalar outputs into one customer table.

The current pipeline is built for Romande Energie style parquet exports, but
most runtime logic is configured through YAML and can be reused for another
portfolio with the same meter schema.

## What It Does

| Component | Signal used | Main outputs | Runtime dependency |
|---|---|---|---|
| PV detector | Production and radiation patterns | `has_pv`, `prob_pv`, `yearly_prod_kwh`, `yearly_cons_kwh` | Heuristic |
| PV capacity estimator | PV production slope vs radiation, bootstrap uncertainty | `pv_capacity_kwp`, `pv_ci_lower`, `pv_ci_upper`, `sc_share` | Heuristic |
| Battery detector + estimator | Load-shift behavior on dark and sunny days | `has_battery`, `prob_battery`, `battery_capacity_kwh` | Heuristic, usually requires PV |
| AC detector | Summer daytime load-temperature response | `has_ac`, `prob_ac` | `models/ac_detector_v1.joblib` |
| Heat pump detector | Winter night load-temperature response | `has_hp`, `prob_hp`, `hp_type` | `models/hp_detector_v1.joblib` |
| EV detector | Sustained flat high-power import blocks | `has_ev`, `prob_ev`, session counts | Heuristic |
| AC / HP disaggregation | 15-minute appliance load estimates | `ac_disagg_15min.parquet`, `hp_disagg_15min.parquet` | Trained disaggregator artifacts |
| EV session estimator | EV-positive customer intervals | `ev_sessions_15min.parquet` | Heuristic |

## Quick Start

Use the project virtual environment if it already exists:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Point `config/re_production.yaml` at your local data directory:

```yaml
data:
  re_data_dir: "data/re_data/ETHZ_ALL"
  metadata_dir: "data/re_data/ETHZ_ALL"
```

Then run a laptop-friendly portfolio pass:

```bash
python scripts/run_pipeline.py \
  --config config/re_production.yaml \
  --resume \
  --skip-ac-disagg \
  --skip-hp-disagg \
  --pv-bootstrap-n 50

python scripts/export_figures.py --config config/re_production.yaml
```

The joined scalar output is written to
`data/processed/out/results_all_customers.parquet`. Portfolio PNGs are written
to `docs/figures/results`.

## Input Data

Smart-meter input is one or more parquet files under `data.re_data_dir`. Each
file may contain many customers. Files with `metadata` in the file name are
ignored by the customer index.

| Column | Type | Meaning |
|---|---|---|
| `ID` | string | Customer identifier |
| `DT_UTC` | datetime | 15-minute timestamp, UTC naive |
| `CONSO_KWH` | float | Grid import energy in the interval |
| `PROD_KWH` | float | Grid export energy in the interval |

Optional metadata lives under `data.metadata_dir` and is used for cohort
filtering. The current default cohort keeps only customers with
`TYPE_PARTENAIRE_LIBELLE == "Particuliers"` and total consumption below
`100_000` kWh.

Weather is loaded automatically by `WeatherLoader` from MeteoSwiss and cached in
`output.results_dir` as `weather_meteoswiss_*.parquet`. The pipeline expects
weather columns `dt_utc`, `t_2m_C`, and `global_rad_W`. If
`pipeline.weather_required` is `true`, weather failures abort the run before
detectors start.

## Configuration

`re_nilm.pipeline.orchestrator.load_config()` loads `config/default.yaml` and
deep-merges the override passed via `--config`. Paths in the config are resolved
relative to the repository root unless they are already absolute.

Important defaults:

```yaml
data:
  customer_type_filter: "Particuliers"
  max_consumption_kwh: 100_000

pipeline:
  n_workers: 4
  batch_size: 500
  resume_from_checkpoint: true
  write_mode: "parts"
  checkpoint_format: "text"
  memory_limit_mb: 6000
  disagg_part_size: 100
  weather_required: true
  skip_ac_disagg: false
  skip_hp_disagg: false
  skip_pv_capacity: false

models:
  pv:
    capacity_bootstrap_n: 200
    capacity_bootstrap_n_fast: 50
  ac:
    detector_path: "models/ac_detector_v1.joblib"
    disaggregator_path: "models/ac_disaggregator_v1.pkl"
  heat_pump:
    detector_path: "models/hp_detector_v1.joblib"
    disaggregator_path: "models/hp_disaggregator_v1.joblib"

output:
  results_dir: "data/processed/out"
  figures_dir: "docs/figures/results"
```

`config/re_production.yaml` currently overrides the data paths, enables all
detectors, uses `n_workers: "auto"`, and sets PV capacity bootstrapping to `10`
for faster iteration.

## Pipeline Workflow

`scripts/run_pipeline.py` creates a `PipelineOrchestrator`, which runs the
following sequence:

1. Load weather and cache it in `output.results_dir`.
2. Build or load `customer_file_index.json`, applying the customer type and
   consumption filters.
3. Run PV detection -> `pv_indicators.parquet`.
4. Run PV capacity estimation -> `pv_capacity.parquet`.
5. Run battery detection/capacity estimation -> `battery_results.parquet`.
6. Run AC detection -> `ac_detection.parquet`.
7. Run heat pump detection -> `hp_detection.parquet`.
8. Run EV detection -> `ev_detection.parquet`.
9. Run AC disaggregation for AC-positive customers -> `ac_disagg_15min.parquet`.
10. Run HP disaggregation for winter heat-pump customers -> `hp_disagg_15min.parquet`.
11. Run EV session extraction for EV-positive customers -> `ev_sessions_15min.parquet`.
12. Join scalar result tables -> `results_all_customers.parquet`.

When AC/HP detector model files are missing, those steps log a warning and
return empty result columns; PV, PV capacity, battery, and EV can still run.
When disaggregator artifacts are missing or skipped, scalar detection results
are still joined.

## Running the Pipeline

```bash
# Full configured pipeline
python scripts/run_pipeline.py --config config/re_production.yaml

# PV only
python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv

# Heuristic-only pass: PV, battery, EV
python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv,battery,ev

# Resume explicitly
python scripts/run_pipeline.py --config config/re_production.yaml --resume

# Start fresh and remove current step outputs/checkpoints
python scripts/run_pipeline.py --config config/re_production.yaml --no-resume

# Skip heavy 15-minute disaggregation outputs
python scripts/run_pipeline.py --config config/re_production.yaml --skip-ac-disagg --skip-hp-disagg

# Use fewer PV capacity bootstrap iterations for local iteration
python scripts/run_pipeline.py --config config/re_production.yaml --pv-bootstrap-n 50

# Skip PV capacity entirely, useful when only classifications are needed
python scripts/run_pipeline.py --config config/re_production.yaml --skip-pv-capacity
```

Resume behavior is step-based. If a final step output already exists and resume
is enabled, the orchestrator skips that whole step. Within an incomplete step,
`StreamingEngine` skips customers already present in that step's checkpoint.

## Checkpoints and Performance

Each streamed scalar step has its own checkpoint derived from the output name,
for example `pv_capacity_ckpt.parquet` plus an append-only
`pv_capacity_ckpt.txt` when `checkpoint_format: "text"`.

With `write_mode: "parts"`, batches are written to immutable parquet parts and
compacted once at the end of the step. This avoids repeatedly reading and
rewriting a growing output file. Set `keep_part_files: true` only when debugging
part-file behavior.

For a normal laptop:

```yaml
pipeline:
  n_workers: 2        # or 1 for the most predictable memory use
  batch_size: 200
  memory_limit_mb: 4000

models:
  pv:
    capacity_bootstrap_n: 10
```

The CLI override `--pv-bootstrap-n` is the fastest way to tune PV capacity
runtime without editing YAML.

## Outputs

All paths below are relative to `output.results_dir` unless configured
otherwise.

| File | Contents |
|---|---|
| `customer_file_index.json` | Filtered customer-to-parquet mapping |
| `weather_meteoswiss_*.parquet` | Cached weather input |
| `pv_indicators.parquet` | PV classification and annual production features |
| `pv_capacity.parquet` | PV capacity, confidence interval, self-consumption |
| `battery_results.parquet` | Battery classification and capacity estimate |
| `ac_detection.parquet` | AC classification probabilities |
| `hp_detection.parquet` | Heat-pump classification and type |
| `ev_detection.parquet` | EV classification and summary session features |
| `ac_disagg_15min.parquet` | 15-minute AC estimates for AC-positive customers |
| `hp_disagg_15min.parquet` | 15-minute HP estimates for winter-HP customers |
| `ev_sessions_15min.parquet` | EV charging session intervals |
| `results_all_customers.parquet` | Joined scalar result table |

## Exported Figures

After the pipeline has produced `results_all_customers.parquet`, run:

```bash
python scripts/export_figures.py --config config/re_production.yaml
```

The script writes PNG figures to `output.figures_dir`:

| Figure | Description |
|---|---|
| `appliance_adoption_shares.png` | Share of customers classified as PV, AC, HP, battery, EV |
| `appliance_probability_distributions.png` | Detector probability distributions |
| `appliance_cooccurrence_heatmap.png` | Technology co-occurrence among customers |
| `technology_portfolio_summaries.png` | Portfolio-level appliance summary panels |
| `pv_installed_capacity_summary.png` | Aggregate PV capacity with uncertainty and PV counts |
| `pv_capacity_distribution.png` | PV capacity distribution, x-axis capped at 100 kWp |
| `pv_population_statistics.png` | PV capacity and population statistics |
| `pv_capacity_vs_production.png` | Capacity vs annual PV production validation |

Static PNG export uses Plotly plus Kaleido. Install the dev extra if PNG export
fails:

```bash
pip install -e ".[dev]"
```

## Training AC and HP Models

AC and HP detectors are Random Forest models trained on labelled Dataport
households. The training dataset should contain per-customer features and labels
such as `has_ac` and `hp_type`.

```bash
python scripts/train_models.py \
  --config config/re_production.yaml \
  --training-data data/processed/training/all_sources_load_with_weather.parquet \
  --models ac_detector,hp_detector
```

The script writes `models/ac_detector_v1.joblib` and
`models/hp_detector_v1.joblib`. The bundled runtime pins scikit-learn to
`>=1.3,<1.4`; regenerate model artifacts in the same environment if loading
fails with pickle, `_loss`, or NumPy `BitGenerator` errors.

The AC disaggregator artifact is separate from detector training:
`models/ac_disaggregator_v1.pkl`. The HP disaggregator is
`models/hp_disaggregator_v1.joblib`.

## Validation and Tests

Run unit tests with synthetic data:

```bash
python -m pytest tests/unit/ -v
```

Run the synthetic integration test suite:

```bash
python -m pytest tests/integration/ -v -m integration
```

Validate detector outputs against legacy scripts when real data and weather are
available:

```bash
python scripts/validate_detectors.py \
  --data-dir data/re_data/ETHZ_ALL \
  --weather data/processed/out/weather_meteoswiss_None_None.parquet
```

`validate_detectors.py` depends on legacy modules under `old_files/`; it is a
data validation utility, not a lightweight CI smoke test.

## Demo Notebook

`notebooks/re_nilm_demo.ipynb` is the detailed walkthrough. It covers:

1. Loading weather and the filtered customer index.
2. Running detectors on a single customer.
3. Training AC/HP models when labelled data is present.
4. Running a small direct portfolio sample.
5. Running or loading the full orchestrator output.
6. Generating portfolio summaries and PV forecast examples.
7. Mapping notebook cells to the equivalent CLI commands.

Keep `RUN_FULL_PIPELINE = False` unless you intentionally want to launch a full
portfolio run from the notebook.

## Project Structure

```text
re_nilm/
  data/
    loaders/              # SmartMeterLoader, WeatherLoader, DataportLoader
    preprocessing.py      # resampling and cleaning helpers
    alignment.py          # meter/weather timestamp alignment
  detectors/              # PV, battery, AC, heat pump, EV detectors
  estimators/             # capacity, disaggregation, EV session estimators
  features/               # shared load, weather, temporal feature functions
  pipeline/
    customer_index.py     # filtered customer_id -> parquet paths mapping
    streaming.py          # batched checkpointed processing engine
    orchestrator.py       # end-to-end pipeline composition
  portfolio/              # aggregation, evaluation, forecasting helpers
  training/               # AC and HP detector trainers
  visualization/          # customer and portfolio plotting utilities

config/
  default.yaml            # default runtime configuration
  re_production.yaml      # local production overrides

scripts/
  run_pipeline.py         # main pipeline CLI
  train_models.py         # detector training CLI
  export_figures.py       # portfolio PNG export CLI
  validate_detectors.py   # comparison against legacy detector logic

models/                   # serialized model artifacts
data/                     # raw, processed, and output data
docs/figures/results/     # exported PNG figures
tests/                    # unit and integration tests
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Pipeline finishes immediately | Resume found existing final outputs | Remove the specific output/checkpoint or run with `--no-resume` |
| Some customers are missing | Cohort filter excluded non-`Particuliers` or >100 MWh customers | Check `data.customer_type_filter`, `data.max_consumption_kwh`, and `customer_file_index.json` |
| PV capacity step takes a long time | Bootstrap count and many PV-positive customers | Use `--pv-bootstrap-n 10` or `--skip-pv-capacity` for iteration |
| AC/HP steps are skipped | Model artifact is missing | Run `scripts/train_models.py` or place the expected model files in `models/` |
| PNG export fails | Kaleido is missing | Install `pip install -e ".[dev]"` |
| Model loading fails with pickle/sklearn errors | Artifact was saved with a different library version | Regenerate the model in this `.venv` with scikit-learn `>=1.3,<1.4` |
