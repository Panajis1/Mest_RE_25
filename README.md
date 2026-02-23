# Mest_CS_Romande_Energie

Case study repository for the Romande Energie CS: MeteoSwiss environmental data and modelling (feature extraction, unsupervised vs supervised comparison).

## Prerequisites

- Python **3.10+** (recommended)

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Mest_RE_25
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create
   python3 -m venv .venv

   # Activate (macOS / Linux)
   source .venv/bin/activate

   # Activate (Windows, cmd)
   .venv\Scripts\activate.bat

   # Activate (Windows, PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project structure

- **`data/`** – Data loading and reference data  
  - `envdata.py` – Fetches and processes MeteoSwiss station data (temperature, radiation, wind, snow).  
  - `load_smart_meter.py` – Load Romande Energie smart meter parquet files (`get_parquet_files`, `load_customer_data`, `load_all_data_generator`).  
  - `load_dataport.py` – Load Pecan Street Dataport 15-min data, EV metadata, weather, and pricing/messages from `data/raw_data/data_port/`.  
  - `preprocessing.py` – Clean and resample smart meter series (`clean_and_resample`, `normalize_zscore`).  
  - `reference_data.py`, `RE.py` – Reference data helpers (to be extended).
- **`scripts/`** – One-off and exploration scripts  
  - `explore_dataport.py` – Summarise and plot Dataport load curves and appliances (PV, EV, AC, heat).  
  - `extract_dataport_appliance_data.py` – Extract load curves and appliance loads (TOT, HP, PV, EV, AC) for appliance households to `data/processed_data/` (long format Parquet + manifest).
- **`model/`** – Modelling pipeline  
  - `1_feature_extraction.py` – Feature extraction.  
  - `2_A_unsupervised.py` – Unsupervised approach (A).  
  - `3_B_supervised.py` – Supervised approach (B).  
  - `4_comparison_AB.py` – Comparison of A vs B.

## Usage

- **Environment data**: Import `env_data` from `data.envdata` to load historical MeteoSwiss data from public CSV URLs (MeteoSwiss Ogd-SMN). Default stations include Biere, St.Prex, Vevey/Corseaux, Villars-Tiercelin, Mathod, Bullet/La Fretaz. Run or import from repo root, or add the repo root to `PYTHONPATH` so `from data.envdata import env_data` works.
- **Smart meter data**: Place parquet files in a directory (e.g. `data/` or `data/ETHZ/`). Use `data.load_smart_meter.get_parquet_files(data_dir)`, `load_customer_data(file_path)`, or `load_all_data_generator(data_dir)` with that path. Optionally pipe through `data.preprocessing.clean_and_resample` for resampled, gap-filled series.
- **Model pipeline**: Run the scripts in `model/` in order (1 → 2 → 3 → 4) once they are implemented.

## Data

This section describes where data lives, in what format, and how it is used. The **`data/`** directory is both the Python package for loading code and the root for raw and processed datasets.

### Directory structure

```
data/
├── raw_data/          # External inputs (as downloaded or received)
│   ├── data_port/     # Pecan Street Dataport (household circuits, EV, weather)
│   └── uk_national_grid/   # EV charging (e.g. GreenFlux CSV)
├── processed_data/    # Script outputs: derived tables, long-format series
├── envdata.py         # MeteoSwiss environmental data (fetched from URLs)
├── load_dataport.py   # Dataport loaders
├── load_smart_meter.py
└── preprocessing.py
```

- **`raw_data/`** – Unprocessed source data. Layout and formats depend on the provider; see subsections below.
- **`processed_data/`** – Results of pipeline scripts (e.g. gap-filled series, appliance-household extracts). Typically **Parquet** for time series and **CSV** for manifests or small lookup tables.
- **Smart meter data** used by `load_smart_meter` may live under `data/` or elsewhere; point the loader at the directory containing **Parquet** files.

### What is where (format overview)

| Location | Format | Description |
|----------|--------|-------------|
| `data/raw_data/data_port/` | SQLite, ZIP (CSV inside), optional CSV/CSV.gz | Pecan Street Dataport: 15‑min circuit-level data per region, EV metadata, weather, optional pricing/events. |
| `data/raw_data/uk_national_grid/` | CSV | EV charging data (e.g. GreenFlux); scripts can write processed output to `processed_data/`. |
| `data/processed_data/` | Parquet, CSV | Derived datasets: e.g. long-format load curves and appliance loads (Source, ID customer, type, Value_KW_mean, dt_utc), plus manifest CSVs. |
| MeteoSwiss | CSV (URLs) | Environmental data (temperature, radiation, wind, etc.) is fetched by `data.envdata` from public Ogd-SMN endpoints; no local raw copy required, no API key. |

### Context by source

- **Environmental (MeteoSwiss)** – Used for feature extraction and modelling. Load via `data.envdata`; data is read from public CSVs (e.g. `data.geo.admin.ch/ch.meteoschweiz.ogd-smn/...`).
- **Dataport (Pecan Street)** – Household-level load and appliance data (grid, solar, EV, AC, heat, etc.) at 15‑min resolution, by region. Stored as SQLite tables per region, plus ZIPs containing EV metadata and weather CSVs, and optional project-specific CSVs. Use `data.load_dataport` (e.g. `get_available_regions`, `load_15min_data`, `load_all_15min_data`). Column names and schema are documented in the loader and in `scripts/explore_dataport.py`.
- **Processed outputs** – Scripts such as `extract_dataport_appliance_data.py` produce long-format Parquet and manifest CSV in `data/processed_data/`. Run from repo root: `python scripts/extract_dataport_appliance_data.py`.


