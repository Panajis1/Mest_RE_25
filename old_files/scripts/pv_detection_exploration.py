#imports
#%%

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from pathlib import Path
from tqdm import tqdm
try:
    _notebook_file = __file__
except NameError:
    _notebook_file = os.getcwd()
_repo_root = Path(_notebook_file).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


from model.pv_detection import *

DATA_DIR = str(_repo_root / "data" / "re_data" / "ETHZ_ALL")
PV_INDICATORS_PARQUET = _repo_root / "data" / "out" / "pv_indicators_clean.parquet"

# Romande Energie: Particuliers + total CONSO over history <= 100 MWh (see stream_customer_summary docstring)
MAX_CONS_KWH = 100_000

#%%
# =========================================================================
# Phase 0: Setup (tiny, stays in memory the whole time)
# =========================================================================

# 0a. Load regional meteo data (~few MB)
print("Loading meteo data...")
combined_meteo, avg_meteo_15min = load_meteo_data()

# 0b. Compute daily weather features (rad_bucket) from meteo alone
print("Computing daily weather features...")
daily_weather = compute_daily_weather(avg_meteo_15min)

# 0c. Metadata: Particuliers only (TYPE_PARTENAIRE_LIBELLE == "Particuliers")
print("Loading customer metadata...")
metadata = load_customer_metadata(DATA_DIR)
if metadata is None or metadata.empty:
    raise FileNotFoundError(
        f"Missing or empty metadata parquet under {DATA_DIR!r} (expected .../metadata)."
    )
particulier_ids = particuliers_customer_ids(metadata)
print(f"  {len(particulier_ids):,} Particuliers IDs (metadata)")


#%%
# =========================================================================
# Phase 1: Streaming customer summary + PV indicators
# =========================================================================

# 1a. Scan all files: only Particuliers, then keep IDs with total CONSO <= 100 MWh
print("Streaming customer summary...")
customer_summary, target_ids = stream_customer_summary(
    DATA_DIR,
    max_cons_kwh=MAX_CONS_KWH,
    allowed_ids=particulier_ids,
)
print(
    f"  {len(customer_summary)} Particuliers with meter data in scan, "
    f"{len(target_ids)} with sum(CONSO_KWH) <= {MAX_CONS_KWH / 1000:.0f} MWh"
)

# 1a-ii. Customer-to-file index restricted to eligible IDs (same full file scan, smaller dict)
print("Building customer-file index (restricted to eligible IDs)...")
cust_file_index = build_customer_file_index(DATA_DIR, restrict_to_ids=target_ids)
print(f"  {len(cust_file_index)} customers across {len(set(f for fs in cust_file_index.values() for f in fs))} files")

#%%
# 1b. Stream through files to compute PV indicators for target customers
print("Streaming PV indicator computation...")
pv_indicators = stream_pv_indicators(
    DATA_DIR, target_ids, avg_meteo_15min, daily_weather, cust_file_index,
)
pv_indicators = classify_pv_customers(pv_indicators)
print(f"  {len(pv_indicators)} customers with indicators")

# 1c. Filter anomalies
valid_mask = (pv_indicators["DeltaProd"] >= 0) & ~(
    (pv_indicators["yearly_prod"] > 5000)
    & (pv_indicators["corr_prod_rad"] < 0.4)
)
pv_indicators_clean = pv_indicators[valid_mask].copy()
dropped = len(pv_indicators) - len(pv_indicators_clean)
if dropped > 0:
    print(f"  Dropped {dropped} anomalous customers")

# Persist indicators for offline figure export / Results artefacts (slim table)
pv_indicators_clean.to_parquet(PV_INDICATORS_PARQUET, index=False)
print(f"  Saved {len(pv_indicators_clean)} rows to {PV_INDICATORS_PARQUET}")

#%%
# =========================================================================
# Phase 2: Streaming capacity estimation (file-by-file bootstrap)
# =========================================================================
print("Streaming capacity estimation...")
AUTOSAVE_ENABLED = True
AUTOSAVE_PATH = str(_repo_root / "data" / "out" / "capacity_autosave.parquet")  # same as before
AUTOSAVE_EVERY_CUSTOMERS = 1000  # keep same as the run you started (or higher/lower; does not affect resume correctness)

RESUME_FROM_AUTOSAVE = True
OVERWRITE_AUTOSAVE = False

RESUME_FROM_FILE = None  # unless you intentionally want to jump to a specific parquet file
STREAM_MAX_WORKERS = 3
STREAM_BATCH_SIZE = 12
STREAM_N_BOOTSTRAP = 50

print(
    "Phase 2 settings: "
    f"workers={STREAM_MAX_WORKERS}, "
    f"batch_size={STREAM_BATCH_SIZE}, "
    f"n_bootstrap={STREAM_N_BOOTSTRAP}, "
    f"autosave_every={AUTOSAVE_EVERY_CUSTOMERS}"
)

prob_summary = process_customers_streaming(
    DATA_DIR,
    avg_meteo_15min,
    pv_indicators_clean,
    n_bootstrap=STREAM_N_BOOTSTRAP,
    max_workers=STREAM_MAX_WORKERS,
    batch_size=STREAM_BATCH_SIZE,
    daily_weather=daily_weather,
    cust_file_index=cust_file_index,
    autosave_enabled=AUTOSAVE_ENABLED,
    autosave_path=AUTOSAVE_PATH,
    autosave_every_customers=AUTOSAVE_EVERY_CUSTOMERS,
    resume_from_autosave=RESUME_FROM_AUTOSAVE,
    resume_from_file=RESUME_FROM_FILE,
    overwrite_autosave=OVERWRITE_AUTOSAVE,
)

#%%
# Optional: save/load prob_summary
#prob_summary.to_csv("prob_summary.csv", index=False)
try:
    prob_summary = pd.read_csv("prob_summary_from_autosave.csv")
except FileNotFoundError:
    print("prob_summary_from_autosave.csv not found; using in-memory prob_summary.")

#open pv indicators clean
pv_indicators_clean = pd.read_parquet(PV_INDICATORS_PARQUET)
print(f"  Loaded {len(pv_indicators_clean)} rows from {PV_INDICATORS_PARQUET}")

# %%
# =========================================================================
# Phase 2b: Streaming PV forecasting (per-customer, full-year, 15-min)
# =========================================================================
FORECAST_ENABLED = True
FORECAST_YEAR = 2024
FORECAST_OUTPUT_PATH = str(_repo_root / "data" / "out" / f"pv_forecast_{FORECAST_YEAR}_15min.parquet")

if FORECAST_ENABLED:
    print(f"Streaming PV forecasts for year={FORECAST_YEAR} ...")
    out_path = forecast_pv_for_customers_streaming(
        data_dir=DATA_DIR,
        cust_file_index=cust_file_index,
        prob_summary=prob_summary,
        avg_meteo_15min=avg_meteo_15min,
        forecast_year=FORECAST_YEAR,
        output_path=FORECAST_OUTPUT_PATH,
        batch_customers=20,
        default_pr=0.85,
        min_capacity_kwp=0.1,
    )
    print(f"PV forecast parquet written to: {out_path}")

# %%
# =========================================================================
# Phase 3: Evaluation and plotting (no large data in memory)
# =========================================================================

# Static figures for the paper / docs (PNG via kaleido); also shown interactively.
RESULTS_FIG_DIR = _repo_root / "docs" / "figures" / "results"
RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

# 5. Portfolio-level aggregation (uses only prob_summary + pv_indicators)
portfolio_agg = aggregate_portfolio_estimates(prob_summary, pv_indicators_clean)
if portfolio_agg:
    print(f"Total hybrid capacity : {portfolio_agg['total_hybrid_kwp']:.1f} kWp")
    print(f"Independence 95% CI   : [{portfolio_agg['ci_independence'][0]:.1f}, {portfolio_agg['ci_independence'][1]:.1f}] kWp")
    print(f"Conservative 95% CI   : [{portfolio_agg['ci_conservative'][0]:.1f}, {portfolio_agg['ci_conservative'][1]:.1f}] kWp")
    print(f"Aggregate SC share    : {portfolio_agg['aggregate_sc_share']:.1%}")
    print(f"Floor/Reg ratio       : {portfolio_agg['portfolio_floor_to_reg_ratio']:.2f}")

# %%
# 6. Evaluation framework
evaluation = evaluate_portfolio(
    prob_summary,
    pv_indicators_clean,
    metadata=metadata,
    segment_col="TYPE_PARTENAIRE_LIBELLE",
)
print_evaluation_report(evaluation)

# %%
# 7. Evaluation dashboard plots
plot_evaluation_dashboard(evaluation, show=True, save_dir=str(RESULTS_FIG_DIR))

# %%
# 8. Portfolio-level plots (streaming -- loads files one at a time)
plot_portfolio_aggregate_load_streaming(
    DATA_DIR, cust_file_index, avg_meteo_15min,
    prob_summary, resolution="H", show=True, save_dir=str(RESULTS_FIG_DIR),
)
if portfolio_agg:
    plot_portfolio_pv_capacity(prob_summary, portfolio_agg, show=True, save_dir=str(RESULTS_FIG_DIR))

# %%
# 9. Per-customer plots (these only use summary data, no raw data needed)
plot_population_statistics(pv_indicators_clean, show=True, save_dir=str(RESULTS_FIG_DIR))
plot_capacity_vs_production_with_ci(
    prob_summary, pv_indicators_clean, show=True, save_dir=str(RESULTS_FIG_DIR),
)
plot_capacity_vs_self_consumption(prob_summary, show=True, save_dir=str(RESULTS_FIG_DIR))

# %%
# 10. Inspect individual customers (loads only the customer's file on demand)
cust_id_to_inspect = "019e6179a1f50a7770bf47670b80a98ed2831362"
cust_data = load_single_customer(
    cust_id_to_inspect, DATA_DIR, cust_file_index, avg_meteo_15min,
)
if not cust_data.empty:
    plot_yearly_customer_capacity(
        customer_id=cust_id_to_inspect,
        re_data_with_meteo=cust_data,
        prob_summary=prob_summary,
        show=True,
    )

# %%
# 11. Inspect top customers with reasonable self-consumption
filtered = prob_summary[prob_summary["sc_share_mean"] >= 0.2]
if not filtered.empty:
    biggest_customers = filtered.nlargest(5, "pv_capacity_kwp")
    for i, row in biggest_customers.iterrows():
        customer_id = row["customer_id"]
        print(f"Plotting customer: {customer_id} with sc_share_mean={row['sc_share_mean']:.2f} and pv_capacity_kwp={row['pv_capacity_kwp']:.2f}")
        cust_data = load_single_customer(
            customer_id, DATA_DIR, cust_file_index, avg_meteo_15min,
        )
        if not cust_data.empty:
            plot_yearly_customer_capacity(
                customer_id=customer_id,
                re_data_with_meteo=cust_data,
                prob_summary=prob_summary,
                show=True,
            )
        del cust_data
else:
    print("No customer found with sc_share_mean >= 0.2")

# %%
# 12. Yield plausibility detail
yield_df = evaluation.get("yield_df")
if yield_df is not None:
    print("\n--- Yield Flag Summary ---")
    print(yield_df["yield_flag"].value_counts())
    print(f"\nSpecific yield stats:")
    print(yield_df["specific_yield_kwh_kwp"].describe())

# %%
# 13. Segment plausibility detail
seg_stats = evaluation.get("segment_stats")
if seg_stats is not None and not seg_stats.empty:
    print("\n--- Segment Capacity Stats ---")
    print(seg_stats.to_string(index=False))

flagged_df = evaluation.get("flagged_df")
if flagged_df is not None:
    print("\n--- Capacity Flag Summary ---")
    print(flagged_df["capacity_flag"].value_counts())

# %%
prob_path = str(_repo_root / "scripts" / "prob_summary.csv")
prob_summary = pd.read_csv(prob_path)
# Filter to customers with a meaningful capacity (>0)
prob_valid = prob_summary[prob_summary["pv_capacity_kwp"] > 0].copy()
top5 = prob_valid.nlargest(5, "pv_capacity_kwp")
print("Top 5 customers by estimated PV capacity:")
print(top5[["customer_id", "pv_capacity_kwp", "sc_share_mean"]].to_string(index=False))
# ----------------------------------------------------------------------
# 2. Meteo + small file index for top-5 only (same DATA_DIR / pipeline cohort)
# ----------------------------------------------------------------------
combined_meteo, avg_meteo_15min = load_meteo_data()
data_dir = DATA_DIR
top5_ids = set(top5["customer_id"].astype(str))
cust_file_index = build_customer_file_index(data_dir, restrict_to_ids=top5_ids)
# ----------------------------------------------------------------------
# 3. Plot full-year profile for each of the top-5
# ----------------------------------------------------------------------
for _, row in top5.iterrows():
    cid = row["customer_id"]
    print(f"\nPlotting yearly profile for customer {cid} "
          f"(capacity ≈ {row['pv_capacity_kwp']:.1f} kWp, "
          f"SC ≈ {row['sc_share_mean']:.2f})")
    # Load only this customer's time series (streaming across its files)
    cust_data = load_single_customer(
        customer_id=cid,
        data_dir=data_dir,
        cust_file_index=cust_file_index,
        avg_meteo_15min=avg_meteo_15min,
    )
    if cust_data.empty:
        print(f"  Warning: no data found for {cid}, skipping.")
        continue
    # Use your existing full-year plot helper
    plot_yearly_customer_capacity(
        customer_id=cid,
        re_data_with_meteo=cust_data,
        prob_summary=prob_summary,
        show=True,
    )
# %%
# %%
# Plot a few random customers: PV forecast (from parquet) vs meter PROD (optional)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from model.pv_detection import load_single_customer, build_customer_file_index

# --- config ---
FORECAST_PATH = str(_repo_root / "data" / "out" / f"pv_forecast_{FORECAST_YEAR}_15min.parquet")
N_SAMPLE = 5
MIN_CAP_KWP = 0.1  # keep consistent with forecast_pv_for_customers_streaming(...)
SEED =55

# pick a readable window (edit freely)
WINDOW_START = pd.Timestamp(f"{int(FORECAST_YEAR)}-06-01")
WINDOW_DAYS = 14

p = Path(FORECAST_PATH)
if not p.exists():
    raise FileNotFoundError(f"Missing forecast parquet: {p}")

# prob_summary should already exist in-memory from your run; if not, load autosave:
if "prob_summary" not in globals() or prob_summary is None or len(prob_summary) == 0:
    autosave = _repo_root / "data" / "out" / "capacity_autosave.parquet"
    if not autosave.exists():
        raise FileNotFoundError(
            "No in-memory prob_summary and no autosave parquet found. "
            f"Expected: {autosave}"
        )
    prob_summary = pd.read_parquet(autosave)

ps = prob_summary.copy()
ps["customer_id"] = ps["customer_id"].astype(str)
ps["pv_capacity_kwp"] = pd.to_numeric(ps["pv_capacity_kwp"], errors="coerce")

candidates = ps.loc[ps["pv_capacity_kwp"] >= float(MIN_CAP_KWP), "customer_id"]
if candidates.empty:
    raise ValueError("No customers with pv_capacity_kwp >= MIN_CAP_KWP; widen threshold or check prob_summary.")

rng = np.random.default_rng(int(SEED))
sample_ids = rng.choice(candidates.unique(), size=min(int(N_SAMPLE), len(candidates.unique())), replace=False)

# Fast filtered read (needs pyarrow)
import pyarrow.dataset as ds

dset = ds.dataset(str(p), format="parquet")
fc = dset.to_table(filter=ds.field("customer_id").isin(list(map(str, sample_ids)))).to_pandas()
fc["customer_id"] = fc["customer_id"].astype(str)
fc["DT_UTC"] = pd.to_datetime(fc["DT_UTC"])

# small index for loading meter traces only for sampled IDs
cust_file_index_small = build_customer_file_index(DATA_DIR, restrict_to_ids=set(sample_ids))

t0 = WINDOW_START
t1 = WINDOW_START + pd.Timedelta(days=int(WINDOW_DAYS))

for cid in sample_ids:
    cid = str(cid)
    g = fc.loc[fc["customer_id"] == cid].sort_values("DT_UTC")
    gg = g[(g["DT_UTC"] >= t0) & (g["DT_UTC"] < t1)].copy()

    cap = float(ps.loc[ps["customer_id"] == cid, "pv_capacity_kwp"].iloc[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gg["DT_UTC"],
            y=gg["pv_forecast_kwh_15min"],
            name="PV forecast (kWh/15m)",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gg["DT_UTC"],
            y=gg["global_rad_W"],
            name="global_rad_W (W/m²)",
            mode="lines",
            yaxis="y2",
            opacity=0.35,
        )
    )

    # optional: observed production channel (same window)
    try:
        hist = load_single_customer(cid, DATA_DIR, cust_file_index_small, avg_meteo_15min)
        if hist is not None and not hist.empty:
            hh = hist[(hist["DT_UTC"] >= t0) & (hist["DT_UTC"] < t1)].copy()
            fig.add_trace(
                go.Scatter(
                    x=hh["DT_UTC"],
                    y=hh["PROD_KWH"],
                    name="meter PROD_KWH (kWh/15m)",
                    mode="lines",
                )
            )
    except Exception as e:
        print(f"[{cid}] could not load meter series: {e}")

    fig.update_layout(
        title=f"PV forecast vs meteo (+ meter PROD) — {cid} | cap={cap:.2f} kWp | {t0.date()}..{t1.date()}",
        xaxis_title="UTC",
        yaxis=dict(title="kWh / 15 min"),
        yaxis2=dict(title="W/m²", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.show()

# %%
