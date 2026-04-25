# BD Heuristic Detailed Report

This document explains how `BD_heuristic.py` works (same implementation as `model/BD_heuristic.py`), including:

- preset/tunable parameters,
- battery detection logic,
- battery capacity estimation logic,
- runtime flow and outputs.

---

## 1) Purpose and Inputs

The script estimates, per residential customer:

- battery probability (`battery_prob`, percent),
- binary battery flag (`has_battery`),
- battery capacity estimate (`estimated_battery_capacity_kwh`) and interval (`capacity_ci_lower_kwh`, `capacity_ci_upper_kwh`),
- PV-related attributes (`is_pv_customer`, `has_pv_prob`, `pv_capacity_kwp`),
- quality/status fields (`status`, `observed_gap_kwh`).

Primary input sources:

- Smart meter + production data from parquet files in `DATA_DIR` (loaded in batches via `data.fast_load_smart_meter.load_all_data_parallel_generator`).
- Weather data from `WEATHER_CSV`.
- PV summary features from `PV_SUMMARY_CSV`.

Primary outputs:

- customer-level CSV: `RESULTS_CSV` (default `battery_residential_results_v7.csv`).
- optional per-customer profile plots for detected batteries in `PLOT_DIR`.
- color convention for generated analysis plots (from `model/BD_h_analysis.py`):
  - black = baseline/no-detection/needs-review category,
  - red = detected/reliable category.

---

## 2) Preset/Tunable Parameters (Top of File)

All key hardcoded values are centralized near the top of the file.

### 2.1 Data / I/O / Runtime

- `DATA_DIR`  
  Base data directory for parquet inputs + metadata.
- `PARTNER_TYPE`  
  Partner segment filter (default residential: `"Particuliers"`).
- `BATCH_SIZE`  
  Batch size for parallel loader.
- `PLOT_DIR`  
  Output directory for detected-customer profile plots.
- `PV_SUMMARY_CSV`  
  Path to PV probability/capacity summary CSV.
- `WEATHER_CSV`  
  Path to weather time series CSV.
- `RESULTS_CSV`  
  Output file for final battery results.
- `PROGRESS_PRINT_EVERY`  
  Log progress every N analyses.
- `GC_EVERY_N_BATCHES`  
  Run `gc.collect()` every N batches.

### 2.2 Time and Merge Handling

- `INPUT_TIMEZONE`  
  Source timezone for meter index if tz-naive (default `"UTC"`).
- `LOCAL_TIMEZONE`  
  Analysis timezone (default `"Europe/Zurich"`).
- `WEATHER_MERGE_DIRECTION`  
  `merge_asof` direction for joining weather to meter data.

### 2.3 Day Selection and Matching

- `DARK_DAY_RAD_MAX_W`  
  Max daily radiation threshold to classify dark days.
- `SUNNY_DAY_RAD_MIN_W`  
  Min daily radiation threshold to classify sunny days.
- `MIN_DARK_REFERENCE_DAYS`  
  Minimum number of dark reference days required.
- `MIN_SUNNY_MATCH_DAYS`  
  Minimum number of sunny matched days required.
- `TEMP_MATCH_BUFFER_C`  
  Temperature matching band around dark-day mean.

### 2.4 Evening Window Detection

- `SUNSET_SCAN_START_HOUR`  
  Earliest hour to search for sunset in average sunny radiation profile.
- `SUNSET_RAD_THRESHOLD_W`  
  Radiation threshold used to identify sunset.
- `DEFAULT_SUNSET_TIME`  
  Fallback sunset time if dynamic search fails.
- `EVENING_END_TIME`  
  End of analysis window for evening discharge.

### 2.5 Feature Conversion and PV Potential

- `RAD_W_TO_KWH_PER_15MIN`  
  Conversion factor from W/m2 to kWh/m2 for 15-min intervals.
- `PV_POTENTIAL_EFFICIENCY`  
  PV conversion/efficiency factor in potential production estimate.
- `DEFAULT_PV_CAPACITY_KWP`  
  Fallback PV size when no customer-specific PV capacity exists.

### 2.6 Classification Thresholds

- `PV_PROBABILITY_THRESHOLD`  
  Threshold to tag a customer as PV from `has_pv_prob`.
- `BATTERY_CLASSIFICATION_THRESHOLD`  
  Final battery threshold (0.5 = 50%).

### 2.7 Sigmoid Score Model

- `SIGMOID_INTERCEPT`
- `SIGMOID_PEAK_SHIFT_WEIGHT`
- `SIGMOID_GAP_RATIO_WEIGHT`
- `INJECTION_RATIO_REFERENCE`
- `INJECTION_BONUS_WEIGHT`
- `INJECTION_RATIO_MIN_POTENTIAL_KWH`
- `INJECTION_RATIO_CLIP_MIN`
- `INJECTION_RATIO_CLIP_MAX`
- `SIGMOID_CLIP_MIN`
- `SIGMOID_CLIP_MAX`

These parameters define and stabilize the battery probability score.

### 2.8 Post-sigmoid Guardrails

- `SECURITY_MIN_GAP_KWH`
- `SECURITY_MAX_GAP_KWH`
- `SECURITY_OUTSIDE_SCALER`
- `PHYSICAL_GAP_FACTOR`
- `PHYSICAL_OUTSIDE_SCALER`

These reduce confidence for implausible energy-gap regimes.

### 2.9 Capacity Estimation

- `SHIFT_BLEND_GAP_WEIGHT`
- `SHIFT_BLEND_SELFCONS_WEIGHT`
- `CAPACITY_ESTIMATE_QUANTILE`
- `CAPACITY_LOW_QUANTILE`
- `CAPACITY_HIGH_QUANTILE`
- `CAPACITY_MIN_PHYSICAL_KWH`
- `CAPACITY_MAX_PER_KWP`

These control inferred shift blending and quantile-based capacity estimation.

### 2.10 Plot/Derived Helpers

- `PEAK_SHIFT_STEP_MINUTES`
- `PLOT_WIDTH`
- `PLOT_HEIGHT`
- `PLOT_DARK_ALPHA`
- `PLOT_FILL_ALPHA`

---

## 3) High-Level Processing Flow

1. Load PV summary (`PV_SUMMARY_CSV`) and weather (`WEATHER_CSV`).
2. Preprocess weather index once (`_prepare_weather_for_merge`).
3. Estimate expected customer count from metadata (`_estimate_partner_customer_count`) for progress logging.
4. Stream customer data in batches from the parallel loader.
5. For each customer group:
   - run `analyze_battery_residential_v7(...)`,
   - optionally save profile plot when battery is detected,
   - append output row.
6. Write final CSV (`RESULTS_CSV`) and summary logs.

---

## 4) Battery Detection Logic (Detailed)

Battery detection is implemented in `analyze_battery_residential_v7`.

### 4.1 Time Alignment and Weather Merge

- If customer index has no timezone, localize with `INPUT_TIMEZONE`.
- Convert to `LOCAL_TIMEZONE`, then drop tz info for naive alignment.
- `merge_asof` weather onto customer timestamps.

### 4.2 Build Daily Features

From merged 15-min data:

- Convert radiation to interval energy:
  - `rad_kwh_per_m2 = global_rad_W * RAD_W_TO_KWH_PER_15MIN`
- Daily aggregation:
  - average temperature (`t_2m_C` mean),
  - max radiation (`global_rad_W` max),
  - daily radiation energy (`rad_kwh_per_m2` sum),
  - daily production (`PROD_KWH` sum).

### 4.3 Select Dark and Sunny Day Sets

- Dark days: `daily global_rad_W < DARK_DAY_RAD_MAX_W`.
- Reject if dark days `< MIN_DARK_REFERENCE_DAYS`.
- Sunny candidates:
  - `daily global_rad_W >= SUNNY_DAY_RAD_MIN_W`
  - and daily temperature within `avg_temp_dark ± TEMP_MATCH_BUFFER_C`.
- Reject if sunny matches `< MIN_SUNNY_MATCH_DAYS`.

### 4.4 Build Evening Profiles and Dynamic Sunset

- Compute average sunny-day radiation profile by time-of-day.
- Daily Solar Peak: For each day, the script identifies the time of maximum global radiation.
- Sunset chosen as first time after `SUNSET_SCAN_START_HOUR` where radiation falls below `SUNSET_RAD_THRESHOLD_W`; fallback `DEFAULT_SUNSET_TIME`.
- Window Alignment in Summer: The window might start as late as 21:00. Winter: The window might start as early as 16:30.
- Build dark and sunny evening consumption profiles from `sunset_time` to `EVENING_END_TIME`.
- Profile Integration: The Evening Profile is integrated from this dynamic start time until `EVENING_END_TIME`. This ensures that any residual daytime PV production is excluded from the battery discharge calculation.

### 4.5 Core Detection Signals

From evening profiles:

- `dark_energy = trapezoid(prof_dark)`
- `sunny_energy = trapezoid(prof_sun)`
- `avg_gap_kwh = dark_energy - sunny_energy`
- `gap_ratio = avg_gap_kwh / dark_energy` (if dark energy > 0)
- `peak_shift = (argmax_sunny - argmax_dark) * PEAK_SHIFT_STEP_MINUTES`

Interpretation:

- Larger positive evening gap suggests stored PV energy displacing evening demand.
- Peak timing shift contributes additional evidence.

### 4.6 Injection/PV Consistency Signal

- Infer PV capacity (`pv_capacity_kwp`) from prioritized columns in PV summary.
- Determine PV-customer flag:
  - If `has_pv_prob` exists: `>= PV_PROBABILITY_THRESHOLD`.
  - Else fallback: capacity > 0.
- Compute PV potential:
  - `pot_high = pv_capacity_for_potential * avg_daily_radiation * PV_POTENTIAL_EFFICIENCY`
- Compute injection ratio:
  - `inj_ratio = clip(avg_injection / pot_high, INJECTION_RATIO_CLIP_MIN, INJECTION_RATIO_CLIP_MAX)`
  - fallback to 1.0 if potential is too small (`INJECTION_RATIO_MIN_POTENTIAL_KWH`).

Low injection ratio (vs reference) increases battery likelihood (`inj_bonus`).

### 4.7 Sigmoid Probability Score

Linear score:

- `inj_bonus = INJECTION_BONUS_WEIGHT * (INJECTION_RATIO_REFERENCE - inj_ratio)`
- `z = SIGMOID_INTERCEPT + SIGMOID_PEAK_SHIFT_WEIGHT * peak_shift + SIGMOID_GAP_RATIO_WEIGHT * gap_ratio + inj_bonus`

Sigmoid with clipping:

- `prob = 1 / (1 + exp(-clip(z, SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX)))`

### 4.8 Guardrails and Final Probability

Two multiplicative scalers:

- Security scaler:
  - 1.0 if `SECURITY_MIN_GAP_KWH <= avg_gap_kwh <= SECURITY_MAX_GAP_KWH`
  - else `SECURITY_OUTSIDE_SCALER`
- Physical scaler:
  - 1.0 if `avg_gap_kwh <= pot_high * PHYSICAL_GAP_FACTOR`
  - else `PHYSICAL_OUTSIDE_SCALER`

Final:

- `final_prob = prob * security_scaler * phys_scaler`
- reported as percent: `battery_prob = round(final_prob * 100, 2)`
- binary decision:
  - `has_battery = "Yes"` if `final_prob >= BATTERY_CLASSIFICATION_THRESHOLD`
  - else `"No"`.

---

## 5) Capacity Estimation Logic (Detailed)

Capacity estimation is based on inferred shifted energy on sunny evenings.

### 5.1 Added Self-Consumption Signal

- Detect candidate columns containing self-consumption hints (`_detect_added_selfcons_columns`).
- Convert detected columns numeric and aggregate to `added_selfcons_kwh_est`.

### 5.2 Daily Evening Aggregation

For each day in evening window:

- sum `CONSO_KWH`, `PROD_KWH`, `rad_kwh_per_m2`, `added_selfcons_kwh_est`.
- Compute dark-day evening baseline consumption.

### 5.3 Inferred Shift per Sunny Day

For each matched sunny day:

- `evening_gap_kwh = max(dark_evening_baseline - sunny_evening_conso, 0)`
- `pv_potential_kwh = pv_capacity * evening_rad * PV_POTENTIAL_EFFICIENCY`
- `available_surplus_kwh = max(pv_potential_kwh - evening_prod, 0)`

Then:

- If added self-consumption signal exists:
  - `inferred_shift_kwh = SHIFT_BLEND_GAP_WEIGHT * evening_gap_kwh + SHIFT_BLEND_SELFCONS_WEIGHT * added_selfcons_kwh_est`
- Else:
  - `inferred_shift_kwh = min(evening_gap_kwh, available_surplus_kwh)`

Finally clip inferred shift at 0.

### 5.4 Quantile-Based Capacity Output

From `valid_shift` (non-null inferred shifts):

- point estimate:
  - `cap_est = quantile(CAPACITY_ESTIMATE_QUANTILE)` (default 90th percentile)
- interval:
  - `cap_low = quantile(CAPACITY_LOW_QUANTILE)` (25th percentile)
  - `cap_high = quantile(CAPACITY_HIGH_QUANTILE)` (75th percentile)

Physical cap (if PV capacity known):

- `max_physical = max(CAPACITY_MIN_PHYSICAL_KWH, CAPACITY_MAX_PER_KWP * pv_capacity_kwp)`
- clip estimate and interval to `[0, max_physical]`.

Returned fields:

- `estimated_battery_capacity_kwh`
- `capacity_ci_lower_kwh`
- `capacity_ci_upper_kwh`

---

## 6) Output Schema (Per Customer)

Key columns written to `RESULTS_CSV`:

- `ID`
- `battery_prob` (0..100)
- `has_battery` (`Yes`/`No`)
- `status` (`Success` or rejection/error reason)
- `observed_gap_kwh`
- `estimated_battery_capacity_kwh`
- `capacity_ci_lower_kwh`
- `capacity_ci_upper_kwh`
- `is_pv_customer`
- `has_pv_prob`
- `pv_capacity_kwp`

---

## 7) Rejection and Error States

Common early exits:

- `REJECTED: No Baseline`  
  Not enough dark reference days.
- `REJECTED: No Sunny Match`  
  Not enough sunny days matched by temperature.
- `ERROR: ...`  
  Unexpected processing exception.

---

## 8) Notes on Runtime Behavior

Current runtime-oriented design choices:

- weather preprocessing done once globally,
- metadata-based customer count hint (no second full data stream),
- `groupby(..., sort=False)` for lower overhead,
- throttled progress prints (`PROGRESS_PRINT_EVERY`),
- periodic GC (`GC_EVERY_N_BATCHES`) instead of every batch.

These were added to reduce runtime while preserving detection concepts.

---

## 9) Plotting Palette and Reliability Assessment

The customer battery plotting utility (`model/BD_h_analysis.py`) now uses a strict
black/red visual language for consistency:

- black = baseline / non-detected / reliable bucket,
- red = detected / highlighted / needs-review bucket.

Main output plots include:

- `battery_detected_share_pie.png`
- `battery_probability_distribution.png`
- `battery_detection_status_breakdown.png` (if status exists)
- `battery_capacity_detected_histogram.png` (if capacity exists)
- `battery_detected_pv_split_pie.png`

Additional reliability-focused plot:

- `battery_detection_reliability_assessment.png`
  - left panel: overall reliable vs needs-review split,
  - right panel: reliable detected vs detected-needs-review counts.

Reliability rule used in plotting:

- status is success/ok **and**
- absolute probability distance from threshold is at least `reliability_margin_pct`
  (default 15 percentage points).
