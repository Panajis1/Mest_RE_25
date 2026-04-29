# Heat Pump Detection & Disaggregation

## Overview

This module implements a two-stage algorithm for detecting and estimating heat pump (HP) consumption from aggregate electricity load curves.

The approach follows:

1. **Detection**
   A Random Forest classifier identifies whether a household has a heat pump (`no_hp`, `winter_hp`, `summer_hp`).

2. **Disaggregation (Estimation)**
   A two-stage model reconstructs the HP consumption from total load using:

   * a classifier (HP on/off probability)
   * a regressor (HP share of total load)

The methodology is described in detail in the report (Section 2.1) .

---

## Repository Structure

```
hp_model/
│
├── hp_detection_functions.py        # Feature engineering + RF model
├── hp_detection_RE.py               # Detection pipeline (external data)
│
├── disaggregation_functions.py      # HP consumption model
├── disaggregregation_RE.py          # Disaggregation pipeline
│
└── (generated outputs)
    ├── hp_detection_outputs/        # CSV outputs
    ├── *_prediction_chunks/         # Parquet outputs
```

---

## ⚠️ Important: File Paths

Many scripts contain **hardcoded local paths**, e.g.:

```python
BASE_DIR = Path("C:/Users/Master/Documents/Mest_RE_25/hp_label")
```

These paths must be changed before running the code.

### You must update:

* input datasets (`.parquet`)
* output directories
* weather data paths
* metadata paths

Search and replace all occurrences of:

```
Path("C:/Users/...")
```

---

## Methodology

### 1. Detection (Random Forest)

* Input: aggregate load (TOT), temperature, solar radiation

* Features:

  * correlation with temperature
  * seasonal and thermal balance
  * autocorrelation (1h, 24h)
  * load variability

* Output: class label per household

The model is trained on labeled datasets and learns temperature-dependent patterns typical of heat pumps.

---

### 2. Disaggregation (Two-stage model)

Applied only to `winter_hp` users.

The model estimates HP consumption as:

$$
P_{HP}(t) = P_{tot}(t) \cdot \hat{r}(t) \cdot \hat{p}_{on}(t)
$$

Where:

* (\hat{p}_{on}(t)): probability HP is active
* (\hat{r}(t)): fraction of total load

Features include:

* temporal (hour, day, season)
* weather (temperature, irradiation)
* load dynamics (lags, rolling statistics)
* user-level statistics

---

## How to Run

---

### Step 1 — Heat Pump Detection

```bash
python hp_detection_RE.py
```

This will:

* train the Random Forest model
* process external data
* output HP labels for each user

---

### Step 2 — Heat Pump Disaggregation

```bash
python disaggregregation_RE.py
```

This will:

* train the two-stage model
* apply it to detected `winter_hp` users
* generate HP consumption estimates

---



## Outputs

### Detection:

* CSV files with HP labels

### Disaggregation:

* Parquet files with:

  * estimated HP load
  * summary statistics
  * aggregated results

---

## Important Assumptions

* Feature extraction for the detection part is performed **only at night** (low solar irradiation)
* Detection relies on **temperature-load correlation**
* Disaggregation assumes:

  * intermittent HP operation
  * meaningful contribution to total load

---

## Limitations

* Difficult to distinguish HP from other electric heating systems
* Model performance depends on training data quality
* External datasets may differ significantly
* Limited availability of labeled negative examples

---

## Notes

* Scripts are designed for large datasets (chunk-based processing)
* Memory usage is monitored
* External weather data is required for inference

---

## Usage Disclaimer

Users must adapt:

* file paths
* data format
* preprocessing steps

before applying it to new datasets.
