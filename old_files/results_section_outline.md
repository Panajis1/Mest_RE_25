# Results — section structure (PV portfolio study)

Use this skeleton when writing or revising the paper narrative. Align numbers with `scripts/pv_detection_exploration.py` and helpers in `model/pv_detection.py`.

## Inclusion and framing

- **Cohort**: residential **Particuliers** (`TYPE_PARTENAIRE_LIBELLE`), streaming scan over smart-meter parquets, **total historical consumption** `sum(CONSO_KWH) ≤ 100` MWh (`MAX_CONS_KWH`), PV **indicator** stage, then **anomaly filter** on indicators (`pv_indicators` → `pv_indicators_clean`).
- **No external ground truth** for PV labels or nameplate capacity: do **not** claim precision/recall, register MAE, or “validated accuracy.”
- **Primary emphasis**: portfolio **installed PV capacity** (kWp) and **uncertainty brackets** from `aggregate_portfolio_estimates` (`portfolio_agg`), plus minimal diagnostics so totals stay interpretable.

---

## 1. Study population and throughput (2–4 sentences)

**Reader question:** Who is in the sample, and how did filters reduce throughput?

- Report counts in pipeline order:
  1. Particuliers IDs in metadata (`len(particulier_ids)`).
  2. Particuliers with meter data seen in the streaming scan (`len(customer_summary)`).
  3. After consumption cap (`len(target_ids)`).
  4. After PV indicators (`len(pv_indicators)`).
  5. After anomaly filter (`len(pv_indicators_clean)`).
  6. Customers contributing to portfolio capacity totals: `portfolio_agg["n_customers"]` (hybrid `pv_capacity_kwp > 0` only; see `aggregate_portfolio_estimates`).

---

## 2. PV detection prevalence (short)

**Reader question:** How many sites screen as PV-related?

- Summarize **`has_pv`** prevalence on `pv_indicators_clean` (e.g. `mean(has_pv)`, counts).
- Optionally note **`has_pv_prob`** is a **heuristic** correlation-based score, not a calibrated probability.
- One sentence: internal **screening signal** only; not validated against labels or registers.

---

## 3. Portfolio installed capacity (core block)

**Reader question:** What total kWp does the model imply, and how wide is uncertainty?

- From `portfolio_agg` (after `aggregate_portfolio_estimates(prob_summary, pv_indicators_clean)`):
  - **Point total**: `total_hybrid_kwp` (sum over customers with `pv_capacity_kwp > 0`).
  - **Bounds** (explicitly **brackets**, not “true” CI):  
    - **Conservative**: `ci_conservative` — sum of per-customer interval endpoints (perfect correlation).  
    - **Independence**: `ci_independence` — sqrt-sum-of-variances propagation (zero correlation).  
  - State in text that the **real** portfolio uncertainty lies **between** these extremes (shared meteo vs. idiosyncratic noise), matching the docstring of `aggregate_portfolio_estimates`.

---

## 4. One supporting diagnostic (single figure/table in prose)

**Pick one** to avoid clutter:

- **Option A — Floor vs regression:** `portfolio_floor_to_reg_ratio` plus one sentence: deviation from ~1 suggests systematic disagreement between the floor and regression capacity paths (export-dominated sites, model stress, etc.—keep qualitative).
- **Option B — Self-consumption context:** `aggregate_sc_share` (and optionally `aggregate_sc_ci_*`) — interprets portfolio as more export-dominated vs. self-consumed **relative to the model**, not ground truth.

---

## 5. Internal plausibility screening (very short)

**Reader question:** What sanity checks did we run?

- From `evaluate_portfolio` → `evaluation`:
  - **Yield flags**: `evaluation["yield_stats"]["flag_counts"]` (same categories as `yield_df["yield_flag"]`: plausible / low / high / no_data). Frame as **sanity screening** on implied specific yield vs. capacity, not external validation.
  - Optionally one line on **`capacity_flag_counts`** on `flagged_df` if space allows; otherwise omit or “supplementary.”

---

## 6. Forecast (optional, one short block)

**Reader question:** Did we generate a consistent full-year forecast file?

- State that a **full-year, 15-minute** forecast parquet was written (path/year from script: `FORECAST_OUTPUT_PATH`, `FORECAST_YEAR`).
- One tight paragraph: **portfolio total forecast kWh** vs. **sum of `pv_capacity_kwp`** via `validate_portfolio_forecast` (`total_forecast_kwh`, `total_capacity_kwp`, `portfolio_specific_yield_kwh_kwp`, `ok`) — **internal consistency / plausibility**, not validation against measured annual generation at portfolio scale unless you later add that analysis.
- Keep mechanics in **Methods**; Results only states what was produced and the aggregate check.

---

## What to omit (unless ground truth is added later)

- Confusion-matrix-style PV classification performance.
- Claims validated against an official PV register.
- Panel-by-panel narration of `plot_evaluation_dashboard` and per-customer plots — cite as supplementary if needed.

---

## Placeholder checklist (for `docs/results_section_draft.tex`)

| LaTeX / narrative placeholder | Python source |
|------------------------------|---------------|
| \(N_{\mathrm{meta}}\) | `len(particulier_ids)` |
| \(N_{\mathrm{scan}}\) | `len(customer_summary)` |
| \(N_{\mathrm{cap}}\) | `len(target_ids)` |
| \(N_{\mathrm{ind}}\) | `len(pv_indicators)` |
| \(N_{\mathrm{clean}}\) | `len(pv_indicators_clean)` |
| \(N_{+}\) | `portfolio_agg["n_customers"]` |
| \(p_{\mathrm{pv}}\), \(K_{\mathrm{pv}}\) | `pv_indicators_clean["has_pv"].mean()`, `int(pv_indicators_clean["has_pv"].sum())` |
| \(C_{\mathrm{tot}}\), CI tuples | `portfolio_agg["total_hybrid_kwp"]`, `portfolio_agg["ci_conservative"]`, `portfolio_agg["ci_independence"]` |
| \(R_{\mathrm{f2r}}\) (diagnostic A) | `portfolio_agg["portfolio_floor_to_reg_ratio"]` |
| Aggregate SC share (diagnostic B) | `portfolio_agg["aggregate_sc_share"]` |
| Yield flag counts | `evaluation["yield_stats"]["flag_counts"]` or `evaluation["yield_df"]["yield_flag"].value_counts()` |
| Capacity flags | `evaluation["capacity_flag_counts"]` |
| Forecast year, path | `FORECAST_YEAR`, `FORECAST_OUTPUT_PATH` |
| \(E_{\mathrm{fc}}\), \(C_{\mathrm{sum}}\), implied yield, `ok` | `validate_portfolio_forecast(FORECAST_OUTPUT_PATH, prob_summary)` return dict |
