import pandas as pd
import numpy as np
import os
import gc
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

data_dir = "C:\\Users\\Aline\\Documents\\Studium\\Case Study\\processed_data\\ETHZ_ALL"


# --- 1. Path & Setup ---
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import data.envdata as meteo
import data.fast_load_smart_meter as re_data

# --- 1. Core Detection Logic ---
def analyze_battery_residential_v7(df_customer, weather_df, pv_row, temp_buffer=2.0):
    """
    V7: Residential-only detection with strict capacity guardrails (5-30 kWh).
    """
    res = {
        "battery_prob": 0, "has_battery": "No", 
        "observed_gap_kwh": 0, "max_potential_kwh": 0,
        "peak_shift_mins": 0, "status": "Success", "profiles": None
    }
    
    try:
        # Step A: Data Alignment
        df_customer.index = df_customer.index.as_unit('ms')
        weather_df.index = weather_df.index.as_unit('ms')
        merged = pd.merge_asof(df_customer.sort_index(), weather_df, 
                              left_index=True, right_index=True, direction='backward')

        # Step B: Solar Potential (Corrected Units: W -> kW -> kWh)
        merged['rad_kwh_per_m2'] = (merged['global_rad_W'] / 1000) * 0.25
        daily = merged.resample('D').agg({
            't_2m_C': 'mean', 'global_rad_W': 'max',
            'rad_kwh_per_m2': 'sum', 'CONSO_KWH': 'sum'
        }).dropna()

        # Step C: Match 'Dark' vs 'Sunny' Days (Temperature controlled)
        dark_days = daily[daily['global_rad_W'] < 100].index
        avg_temp_dark = daily.loc[dark_days, 't_2m_C'].mean()
        matched_sunny = daily[
            (daily['global_rad_W'] >= 100) & 
            (daily['t_2m_C'].between(avg_temp_dark - temp_buffer, avg_temp_dark + temp_buffer))
        ].index

        if len(dark_days) < 2 or len(matched_sunny) < 2:
            res["status"] = "Insufficient data/weather matching"
            return res

        # Step D: PV Capacity Setup (Assume 5kWp if missing)
        pv_cap_kw = pv_row.get('pv_capacity_ci_upper', 5.0)
        if pd.isna(pv_cap_kw) or pv_cap_kw <= 0: pv_cap_kw = 5.0 

        # Step E: Profile & Gap Calculation (16:00 - 23:59)
        is_dark = pd.Series(merged.index.date).isin(dark_days.date).values
        is_sunny = pd.Series(merged.index.date).isin(matched_sunny.date).values
        
        prof_dark = merged[is_dark].between_time('16:00', '23:59').groupby(lambda x: x.time())['CONSO_KWH'].mean()
        prof_sun = merged[is_sunny].between_time('16:00', '23:59').groupby(lambda x: x.time())['CONSO_KWH'].mean()

        # Gap is the area between the two curves (the missing grid energy)
        avg_gap_kwh = np.trapz(prof_dark.values) - np.trapz(prof_sun.values)
        gap_ratio = avg_gap_kwh / np.trapz(prof_dark.values) if np.trapz(prof_dark.values) > 0 else 0
        peak_shift = (np.argmax(prof_sun.values) - np.argmax(prof_dark.values)) * 15

        # Step F: RESIDENTIAL SECURITY CHECK (5 - 30 kWh)
        # "Speicherkapazitäten typischerweise von 5 bis 30 kWh im Wohnbereich"
        if 5.0 <= avg_gap_kwh <= 30.0:
            security_scaler = 1.0
        else:
            # If gap is 1kWh (too small) or 50kWh (too big), it's likely not a home battery
            security_scaler = 0.15 
            res["status"] = f"Filtered: Gap {avg_gap_kwh:.2f}kWh outside 5-30kWh range"

        # Step G: Physical Feasibility (Solar must be able to charge the battery)
        pot_high = pv_cap_kw * daily.loc[matched_sunny, 'rad_kwh_per_m2'].mean() * 0.8
        phys_scaler = 1.0 if avg_gap_kwh <= (pot_high * 1.3) else 0.3

        # Step H: Final Logistic Score
        z = -5.0 + (0.06 * peak_shift) + (7.5 * gap_ratio)
        prob = (1 / (1 + np.exp(-z))) * security_scaler * phys_scaler

        res.update({
            "battery_prob": round(prob * 100, 2),
            "has_battery": "Yes" if (prob * 100) > 75 else "No",
            "observed_gap_kwh": round(avg_gap_kwh, 3),
            "max_potential_kwh": round(pot_high, 3),
            "peak_shift_mins": peak_shift,
            "profiles": (prof_dark, prof_sun)
        })
        return res
    except Exception as e:
        res["status"] = f"Error: {e}"
        return res

# --- 2. Execution Block ---
if __name__ == "__main__":
    freeze_support()
    PLOT_DIR = "residential_profiles_v7"
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

    # Load External Data (Update paths as needed)
    pv_summary = pd.read_csv("scripts/prob_summary.csv").set_index('customer_id')
    import data.envdata as meteo
    import data.fast_load_smart_meter as re_data
    _, weather_df = meteo.env_data()

    data_gen = re_data.load_all_data_parallel_generator(data_dir, partner_type="Particuliers", batch_size=2)

    results = []
    MAX_TEST = 10
    processed = 0

    for batch_df in data_gen:
        for customer_id, customer_data in batch_df.groupby("ID"):
            if processed >= MAX_TEST: break
            
            pv_info = pv_summary.loc[customer_id] if customer_id in pv_summary.index else pd.Series(dtype=float)
            analysis = analyze_battery_residential_v7(customer_data.set_index("DT_UTC"), weather_df, pv_info)
            
            # Save strictly Residential High-Prob Plots
            if analysis["has_battery"] == "Yes" and analysis["profiles"] is not None:
                dark, sun = analysis["profiles"]
                plt.figure(figsize=(8, 4))
                x = [t.hour + t.minute/60 for t in dark.index]
                plt.plot(x, dark.values, 'k--', alpha=0.5, label='Grid (Dark Day)')
                plt.plot(x, sun.values, 'orange', label='Battery (Sunny Day)')
                plt.fill_between(x, sun.values, dark.values, color='orange', alpha=0.2)
                plt.title(f"RESIDENTIAL ID: {customer_id} | Prob: {analysis['battery_prob']}%")
                plt.savefig(f"{PLOT_DIR}/{customer_id}.png")
                plt.close()

            results.append({"ID": customer_id, **{k: v for k, v in analysis.items() if k != 'profiles'}})
            processed += 1
            if processed % 10 == 0: print(f"Progress: {processed}/{MAX_TEST}")

        if processed >= MAX_TEST: break
        del batch_df
        gc.collect()

    # Final Output
    final_df = pd.DataFrame(results)
    print("\n--- FINAL RESIDENTIAL SUMMARY ---")
    print(f"Total: {len(final_df)} | Found: {(final_df['has_battery']=='Yes').sum()}")
    print(f"Avg Prob: {final_df['battery_prob'].mean():.2f}%")
    final_df.to_csv("battery_residential_results_v7.csv", index=False)