import pandas as pd
import numpy as np
import os
import gc
import sys
from pathlib import Path
from scipy.stats import ranksums
from tqdm import tqdm
from multiprocessing import freeze_support

# --- 1. Path & Setup ---
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import data.envdata as meteo
import data.fast_load_smart_meter as re_data

def analyze_battery_robust_logit_v3(df_customer, weather_df, pv_row, temp_buffer=2.0):
    """
    Scientific battery detection using Dynamic Solar Potential 
    calculated from actual radiation data.
    """
    try:
        # 1. Standardize and Merge
        # Ensure we have the right column name for radiation
        if 'global_rad_W' not in weather_df.columns:
            # Fallback if your weather_df uses a different naming convention
            possible_cols = [c for c in weather_df.columns if 'rad' in c.lower() or 'ghi' in c.lower()]
            if possible_cols:
                weather_df = weather_df.rename(columns={possible_cols[0]: 'global_rad_W'})
        
        df_customer.index = df_customer.index.as_unit('us')
        weather_df.index = weather_df.index.as_unit('us')
        
        merged = pd.merge_asof(df_customer.sort_index(), weather_df, 
                              left_index=True, right_index=True, direction='backward')

        # 2. Identify Matched Days
        # Create a temporary column for energy (Wh per 15-min interval)
        # Power (W) * Time (0.25h) / 1000 = kWh
        merged['rad_kwh_per_interval'] = (merged['global_rad_W'] / 1000) * 0.25

        daily = merged.resample('D').agg({
            't_2m_C': 'mean', 
            'global_rad_W': 'max',
            'rad_kwh_per_interval': 'sum' # This is the daily kWh/m2
        }).dropna()
        
        daily = daily.rename(columns={'rad_kwh_per_interval': 'daily_solar_kwh_m2'})

        dark_days = daily[daily['global_rad_W'] < 100].index
        
        if len(dark_days) < 2:
            return {"battery_prob": 0, "has_battery": "No", "status": "Insufficient dark days"}
            
        avg_temp_dark = daily.loc[dark_days, 't_2m_C'].mean()
        
        matched_norm_idx = daily[
            (daily['global_rad_W'] >= 100) & 
            (daily['t_2m_C'].between(avg_temp_dark - temp_buffer, avg_temp_dark + temp_buffer))
        ].index

        if len(matched_norm_idx) < 2:
            return {"battery_prob": 0, "has_battery": "No", "status": "No matched sunny days"}

        # 3. CALCULATE ACTUAL DAILY PV POTENTIAL
        # Use the specific radiation observed on the matched sunny days
        avg_radiation_kwh_m2 = daily.loc[matched_norm_idx, 'daily_solar_kwh_m2'].mean()
        
        PR = 0.8 # Performance Ratio
        pot_high = pv_row.get('pv_capacity_ci_upper', 0) * avg_radiation_kwh_m2 * PR
        pot_low = pv_row.get('pv_capacity_ci_lower', 0) * avg_radiation_kwh_m2 * PR

        # 4. Extract Profiles and Features
        is_dark = pd.Series(merged.index.date).isin(dark_days.date).values
        is_norm = pd.Series(merged.index.date).isin(matched_norm_idx.date).values
        
        profile_dark = merged[is_dark].between_time('16:00', '23:59').groupby(lambda x: x.time())['CONSO_KWH'].mean()
        profile_norm = merged[is_norm].between_time('16:00', '23:59').groupby(lambda x: x.time())['CONSO_KWH'].mean()

        avg_gap_kwh = np.trapz(profile_dark.values) - np.trapz(profile_norm.values)
        peak_shift = (np.argmax(profile_norm.values) - np.argmax(profile_dark.values)) * 15
        gap_ratio = avg_gap_kwh / np.trapz(profile_dark.values) if np.trapz(profile_dark.values) > 0 else 0

        # 5. DYNAMIC PHYSICAL SCALER
        # If the customer has no estimated PV (pot_high is 0), 
        # a solar-charged battery is impossible.
        if pot_high <= 0:
            physical_scaler = 0.0  # Force probability to 0
        elif avg_gap_kwh > pot_high:
            physical_scaler = 0.2  # Gap is larger than solar energy available
        elif avg_gap_kwh < pot_low and avg_gap_kwh > 0.3:
            physical_scaler = 1.2  # High confidence
        else:
            physical_scaler = 1.0

        # 6. LOGISTIC PROBABILITY
        z = -4.5 + (0.07 * peak_shift) + (5.5 * gap_ratio)
        
        # Apply the sigmoid function
        prob = (1 / (1 + np.exp(-z))) * physical_scaler
        
        # Apply Significance Penalty
        e_dark = merged[is_dark].between_time('16:00', '20:00').groupby(level=0)['CONSO_KWH'].sum()
        e_norm = merged[is_norm].between_time('16:00', '20:00').groupby(level=0)['CONSO_KWH'].sum()
        _, p_val = ranksums(e_dark, e_norm)
        if p_val > 0.05: 
            prob *= 0.5

        # FINAL CAP: Ensure result is between 0 and 100
        final_prob = max(0, min(prob, 1.0)) * 100

        return {
            "battery_prob": round(final_prob, 2),
            "has_battery": "Yes" if final_prob > 75 else "No",
            "observed_gap_kwh": round(avg_gap_kwh, 3),
            "max_potential_kwh": round(pot_high, 3),
            "peak_shift_mins": peak_shift,
            "p_val": round(p_val, 4)
        }
    except Exception as e:
        return {"battery_prob": 0, "has_battery": "No", "status": f"Error: {e}"}


# --- 3. Execution Block (30-Customer Test) ---
if __name__ == "__main__":
    freeze_support()

    data_dir = "C:\\Users\\Aline\\Documents\\Studium\\Case Study\\processed_data\\ETHZ"
    PV_SUMMARY_PATH = "scripts/prob_summary.csv" 
    
    # Load constraints
    pv_summary = pd.read_csv(PV_SUMMARY_PATH).set_index('customer_id')
    _, weather_df = meteo.env_data()

    # Generator setup
    data_gen = re_data.load_all_data_parallel_generator(data_dir, partner_type="Particuliers", batch_size=2)

    results_collector = []
    MAX_TEST_CUSTOMERS = 100
    total_processed = 0

    print(f"Starting test run for {MAX_TEST_CUSTOMERS} customers...")

    for batch_df in data_gen:
        # Group by ID to process individual customers
        for customer_id, customer_data in batch_df.groupby("ID"):
            if total_processed >= MAX_TEST_CUSTOMERS:
                break # Exit the inner loop

            # Prepare customer data
            customer_data = customer_data.set_index("DT_UTC").sort_index()

            # Identify PV Info
            pv_info = pv_summary.loc[customer_id] if customer_id in pv_summary.index else pd.Series()

            # Analyze
            analysis = analyze_battery_robust_logit_v3(customer_data, weather_df, pv_info)

            results_collector.append({
                "ID": customer_id,
                "avg_daily_conso": customer_data["CONSO_KWH"].resample('D').sum().mean(),
                **analysis
            })
            
            total_processed += 1
            if total_processed % 5 == 0:
                print(f"Processed {total_processed}/{MAX_TEST_CUSTOMERS}...")

        # Memory management
        del batch_df
        gc.collect()
        
        # Exit the outer generator loop if target reached
        if total_processed >= MAX_TEST_CUSTOMERS:
            print("Target reached. Ending test run.")
            break

    # Save output
    test_df = pd.DataFrame(results_collector)
    test_df.to_csv("battery_test_30_results.csv", index=False)
    print("\n--- Test Results ---")
    print(test_df[['ID', 'battery_prob', 'has_battery', 'observed_gap_kwh', 'max_potential_kwh']].head(10))

"""
# --- 3. Execution Block ---
if __name__ == "__main__":
    # Required for Windows multiprocessing
    freeze_support()

    data_dir = "C:\\Users\\Aline\\Documents\\Studium\\Case Study\\processed_data\\ETHZ"
    PV_SUMMARY_PATH = "scripts/prob_summary.csv" 
    
    # Load constraints
    pv_summary = pd.read_csv(PV_SUMMARY_PATH).set_index('customer_id')
    _, weather_df = meteo.env_data()

    # Generator setup
    data_gen = re_data.load_all_data_parallel_generator(data_dir, partner_type="Particuliers", batch_size=2)

    results_collector = []

    for batch_df in data_gen:
        for customer_id, customer_data in batch_df.groupby("ID"):
            # Prepare customer data
            customer_data = customer_data.set_index("DT_UTC").sort_index()

            # Identify PV Info
            pv_info = pv_summary.loc[customer_id] if customer_id in pv_summary.index else pd.Series()

            # Analyze using V3 (Robust Dynamic Potential)
            analysis = analyze_battery_robust_logit_v3(customer_data, weather_df, pv_info)

            results_collector.append({
                "ID": customer_id,
                "avg_daily_conso": customer_data["CONSO_KWH"].resample('D').sum().mean(),
                **analysis
            })

        # Batch memory management
        del batch_df
        gc.collect()

    # Save output
    final_df = pd.DataFrame(results_collector)
    final_df.to_csv("scripts/battery_detection_final.csv", index=False)
    print(f"Analysis complete. Results saved for {len(final_df)} customers.")"""