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
  - `envdata` – Fetches and processes MeteoSwiss station data (temperature, radiation, wind, snow).  
  - `reference_data.py`, `RE.py` – Reference data helpers (to be extended).
- **`model/`** – Modelling pipeline  
  - `1_feature_extraction.py` – Feature extraction.  
  - `2_A_unsupervised.py` – Unsupervised approach (A).  
  - `3_B_supervised.py` – Supervised approach (B).  
  - `4_comparison_AB.py` – Comparison of A vs B.

## Usage

- **Environment data**: The script in `data/envdata` loads historical MeteoSwiss data from public CSV URLs (MeteoSwiss Ogd-SMN). Default stations include Biere, St.Prex, Vevey/Corseaux, Villars-Tiercelin, Mathod, Bullet/La Fretaz. Run or import it once the module path is set (e.g. add `data` to `PYTHONPATH` or run from repo root).
- **Model pipeline**: Run the scripts in `model/` in order (1 → 2 → 3 → 4) once they are implemented.

## Data

Environmental data is sourced from **MeteoSwiss** (Swiss Federal Office of Meteorology and Climatology), via the Ogd-SMN historical CSVs (e.g. `data.geo.admin.ch/ch.meteoschweiz.ogd-smn/...`). No API key is required for the public CSV endpoints used in `data/envdata`.


