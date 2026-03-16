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
#%%
# 1. Load and pre-filter customers
combined_meteo, avg_meteo_15min = load_meteo_data()
re_data_df, re_data_df_small, customer_summary = load_re_data()

# 2. Align and build features (now includes temperature for demand correction)
meteo_window, re_data_with_meteo = align_meteo_with_re_data(
    avg_meteo_15min, re_data_df_small
)
daily_features, daily_weather = build_daily_features(
    meteo_window, re_data_with_meteo
)

# 3. Indicators + classification
pv_indicators = compute_pv_indicators(daily_features, re_data_with_meteo)
pv_indicators = classify_pv_customers(pv_indicators)

valid_mask = (pv_indicators["DeltaProd"] >= 0) & ~(
    (pv_indicators["yearly_prod"] > 5000)
    & (pv_indicators["corr_prod_rad"] < 0.4)
)
pv_indicators_clean = pv_indicators[valid_mask].copy()
#%%
# 4. Probabilistic capacity estimation (v2: corrected floor + combiner + demand correction)
prob_summary = compute_probabilistic_capacity_parallel(
    re_data_with_meteo,
    daily_features,
    pv_indicators_clean,
    n_bootstrap=50,
    max_workers=2,
    batch_size=50,
)

#%%
# Optional: save/load prob_summary
# prob_summary.to_csv("prob_summary.csv", index=False)
import pandas as pd
try:
    prob_summary = pd.read_csv("prob_summary.csv")
except FileNotFoundError:
    print("prob_summary.csv not found; using in-memory prob_summary from previous step.")

# %%
# 5. Portfolio-level aggregation
portfolio_agg = aggregate_portfolio_estimates(prob_summary, pv_indicators_clean)
if portfolio_agg:
    print(f"Total hybrid capacity : {portfolio_agg['total_hybrid_kwp']:.1f} kWp")
    print(f"Independence 95% CI   : [{portfolio_agg['ci_independence'][0]:.1f}, {portfolio_agg['ci_independence'][1]:.1f}] kWp")
    print(f"Conservative 95% CI   : [{portfolio_agg['ci_conservative'][0]:.1f}, {portfolio_agg['ci_conservative'][1]:.1f}] kWp")
    print(f"Aggregate SC share    : {portfolio_agg['aggregate_sc_share']:.1%}")
    print(f"Floor/Reg ratio       : {portfolio_agg['portfolio_floor_to_reg_ratio']:.2f}")

# %%
# 6. Evaluation framework (v2)
metadata = load_customer_metadata()
evaluation = evaluate_portfolio(
    prob_summary, pv_indicators_clean, metadata=metadata,
)
print_evaluation_report(evaluation)

# %%
# 7. Evaluation dashboard plots
plot_evaluation_dashboard(evaluation, show=True)

# %%
# 8. Portfolio-level plots
plot_portfolio_aggregate_load(re_data_with_meteo, prob_summary, resolution="H", show=True)
if portfolio_agg:
    plot_portfolio_pv_capacity(prob_summary, portfolio_agg, show=True)

# %%
# 9. Per-customer plots
plot_population_statistics(pv_indicators_clean, show=True)
plot_capacity_vs_production_with_ci(prob_summary, pv_indicators_clean, show=True)
plot_capacity_vs_self_consumption(prob_summary, show=True)

# %%
# 10. Inspect individual customers
plot_yearly_customer_capacity(
    customer_id="019e6179a1f50a7770bf47670b80a98ed2831362",
    re_data_with_meteo=re_data_with_meteo,
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
        plot_yearly_customer_capacity(
            customer_id=customer_id,
            re_data_with_meteo=re_data_with_meteo,
            prob_summary=prob_summary,
            show=True,
        )
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
