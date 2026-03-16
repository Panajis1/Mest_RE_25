import pandas as pd
import envdata
import load_smart_meter
import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# 1. Setup & Data Loading
DIR = "C:\\Users\\Aline\\Documents\\Studium\\Case Study\\processed_data\\ETHZ"
weather_df = envdata.env_data().sort_index()

# Fix precision for merge (Unit mismatch fix)
weather_df.index = weather_df.index.as_unit('us')

# Load and process one customer
listparquet = load_smart_meter.get_parquet_files(DIR)
df_raw = load_smart_meter.load_customer_data(listparquet[23])
df_customer = preprocessing.clean_and_resample(df_raw)

# Ensure index is DT_UTC and sorted
if 'DT_UTC' in df_customer.columns:
    df_customer = df_customer.set_index('DT_UTC')
df_customer.index = df_customer.index.as_unit('us')
df_customer = df_customer.sort_index()

# 2. Merge Weather into Meter Data
merged = pd.merge_asof(df_customer, weather_df, left_index=True, right_index=True, direction='backward')

# 3. Identify "Dark Days"
daily_max_rad = merged['global_rad_W'].resample('D').max()
dark_days = daily_max_rad[daily_max_rad < 100].index.date

# 4. Filter for Evening (16:00 - 24:00) on those specific days
evening_mask = merged.index.indexer_between_time('16:00', '23:59')
evening_data = merged.iloc[evening_mask].copy()

# Correct way to filter multiple dates
is_dark_day = pd.Series(evening_data.index.date).isin(dark_days).values
dark_evening_profiles = evening_data[is_dark_day].copy()
#%%
# 5. Plotting Dark Evening Profiles
if not dark_evening_profiles.empty:
    plt.figure(figsize=(12, 6))
    for date, group in dark_evening_profiles.groupby(dark_evening_profiles.index.date):
        x_axis = group.index.hour + group.index.minute/60
        plt.plot(x_axis, group['CONSO_KWH'], alpha=0.5, label=str(date))
    
    plt.title(f'Evening Consumption (16:00-24:00) on Gloomy Days (Found {len(dark_days)} days)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Consumption (kWh)')
    plt.xticks(range(17, 25))
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No days found with Max Radiation < 100 W/m²")

"""# 6. Calculate Correlation (using the 'merged' variable)
# Only use columns that exist in the dataframe
cols_to_corr = [c for c in ['CONSO_KWH', 't_2m_C', 'global_rad_W', 'PROD_KWH'] if c in merged.columns]
correlation_matrix = merged[cols_to_corr].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 7. Visualization: Radiation vs Production
if 'PROD_KWH' in merged.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x='global_rad_W', y='PROD_KWH', alpha=0.3)
    plt.title('Correlation: Radiation vs. Energy Production')
    plt.xlabel('Global Radiation (W/m²)')
    plt.ylabel('Production (kWh)')
    plt.show()"""
# %%
print("dark days identified:", dark_days)
# %%
# 1. Split the merged data into Dark and Normal days
# We use the 'dark_days' list we generated earlier
is_dark_day_mask = pd.Series(merged.index.date).isin(dark_days).values

# Separate the dataframe
dark_df = merged[is_dark_day_mask].copy()
normal_df = merged[~is_dark_day_mask].copy()

# 2. Extract the evening window (17:00 - 24:00)
dark_evening = dark_df.between_time('16:00', '23:59')
normal_evening = normal_df.between_time('16:00', '23:59')

# 3. Create the Aggregated "Typical" Profiles
# We group by the time of day to get the average 15-min consumption
dark_profile = dark_evening.groupby(dark_evening.index.time)['CONSO_KWH'].mean()
normal_profile = normal_evening.groupby(normal_evening.index.time)['CONSO_KWH'].mean()

# 4. Plotting the Comparison
plt.figure(figsize=(12, 6))

# Plot the lines
plt.plot(dark_profile.index.astype(str), dark_profile.values, 
         label=f'Average Dark Day ({len(dark_days)} days)', 
         color='#2c3e50', linewidth=2.5)

plt.plot(normal_profile.index.astype(str), normal_profile.values, 
         label='Average Normal Day', 
         color='#e67e22', linewidth=2.5, linestyle='--')

# Styling the plot
plt.title(f'Individual Customer Profile: Dark vs. Normal Day Evening\n(Customer ID: {df_raw["ID"].iloc[0] if "ID" in df_raw.columns else "Unknown"})', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Consumption (kWh)', fontsize=12)

# Show every 4th tick (every hour) for readability
plt.xticks(dark_profile.index.astype(str)[::4], rotation=45)
plt.grid(True, alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Quick Insight Calculation
avg_diff = ((dark_profile.mean() - normal_profile.mean()) / normal_profile.mean()) * 100
print(f"On dark days, this customer uses {avg_diff:.1f}% more energy in the evening on average.")
# %%
import pandas as pd
import matplotlib.pyplot as plt

# 1. Prepare Profiles (from our previous aggregated data)
# dark_profile and normal_profile are indexed by time (15-min intervals)

# 2. Find the Peak Time for both scenarios
peak_time_dark = dark_profile.idxmax()
peak_time_normal = normal_profile.idxmax()

# 3. Calculate the "Evening Ramp" (Slope from 17:00 to 19:00)
# We look at the first 2 hours of our evening window
ramp_dark = dark_profile.iloc[8] - dark_profile.iloc[0]   # 2 hours = 8 * 15min
ramp_normal = normal_profile.iloc[8] - normal_profile.iloc[0]

# 4. Visualization with Peak Markers
plt.figure(figsize=(12, 6))

plt.plot(dark_profile.index.astype(str), dark_profile.values, 
         label='Dark Days (Grid Dependent)', color='#2c3e50', lw=2)
plt.plot(normal_profile.index.astype(str), normal_profile.values, 
         label='Normal Days (Potential Battery)', color='#e67e22', lw=2, ls='--')

# Mark the peaks
plt.scatter(str(peak_time_dark), dark_profile.max(), color='red', s=100, zorder=5, label=f'Peak Dark: {peak_time_dark}')
plt.scatter(str(peak_time_normal), normal_profile.max(), color='green', s=100, zorder=5, label=f'Peak Normal: {peak_time_normal}')

plt.title('Battery Detection: Peak Shift & Ramp Analysis')
plt.ylabel('kWh')
plt.xticks(dark_profile.index.astype(str)[::4], rotation=45)
plt.legend()
plt.show()

# 5. Output Indicators
print(f"Peak Shift: {peak_time_normal} (Normal) -> {peak_time_dark} (Dark)")
print(f"Ramp Intensity Increase: {((ramp_dark - ramp_normal) / ramp_normal) * 100:.1f}%")
# %%
import numpy as np

def calculate_battery_score(dark_profile, normal_profile):
    """
    Returns a score from 0-100.
    High score = High probability of a battery.
    """
    # 1. PEAK SHIFT CALCULATION
    # Find index of peak (in 15-min intervals)
    peak_idx_dark = np.argmax(dark_profile.values)
    peak_idx_normal = np.argmax(normal_profile.values)
    
    # Positive shift means peak happened LATER on normal days
    # (Typical of battery discharging until empty)
    shift_minutes = (peak_idx_normal - peak_idx_dark) * 15
    
    # 2. RAMP INTENSITY
    # Measure the jump in consumption in the first 90 minutes (16:00 to 17:30)
    ramp_dark = dark_profile.iloc[6] - dark_profile.iloc[0]
    ramp_normal = normal_profile.iloc[6] - normal_profile.iloc[0]
    
    # 3. SCORE LOGIC
    score = 0
    # A peak shift of 1 hour or more is a strong battery indicator
    if shift_minutes >= 60: score += 50
    elif shift_minutes > 0: score += 25
    
    # If the ramp is much "flatter" on normal days, it suggests battery smoothing
    if ramp_dark > (ramp_normal * 1.5): score += 50
    elif ramp_dark > ramp_normal: score += 20
    
    return score, shift_minutes

# --- Implementation ---
prob_score, minutes_shifted = calculate_battery_score(dark_profile, normal_profile)

print(f"Battery Probability Score: {prob_score}/100")
print(f"The evening peak is delayed by {minutes_shifted} minutes on sunny/normal days.")


#%%

# 1. Calculate Daily Metrics
# We need daily average temperature to find matching days
daily_stats = merged.resample('D').agg({
    't_2m_C': 'mean',
    'global_rad_W': 'max'
})

# 2. Identify the "Dark Days" and their Average Temp
dark_days_idx = daily_stats[daily_stats['global_rad_W'] < 100].index
avg_temp_dark = daily_stats.loc[dark_days_idx, 't_2m_C'].mean()

print(f"Average temp on Dark Days: {avg_temp_dark:.2f}°C")

# 3. Find "Normal Days" with similar temperature
# Criteria: Not a dark day AND temp is within +/- 2 degrees of dark days
temp_buffer = 2.0
matched_normal_days_idx = daily_stats[
    (daily_stats['global_rad_W'] >= 100) & 
    (daily_stats['t_2m_C'] >= avg_temp_dark - temp_buffer) & 
    (daily_stats['t_2m_C'] <= avg_temp_dark + temp_buffer)
].index

print(f"Found {len(matched_normal_days_idx)} Normal Days with matching temperature.")

# 4. Filter the main Merged DF using these specific dates
is_dark = pd.Series(merged.index.date).isin(dark_days_idx.date).values
is_matched_normal = pd.Series(merged.index.date).isin(matched_normal_days_idx.date).values

dark_evening = merged[is_dark].between_time('16:00', '23:59')
normal_matched_evening = merged[is_matched_normal].between_time('16:00', '23:59')

# 5. Aggregate
dark_profile = dark_evening.groupby(dark_evening.index.time)['CONSO_KWH'].mean()
normal_profile = normal_matched_evening.groupby(normal_matched_evening.index.time)['CONSO_KWH'].mean()

# 6. Plotting the "Fair" Comparison
plt.figure(figsize=(12, 6))
plt.plot(dark_profile.index.astype(str), dark_profile.values, label='Dark Days', color='#2c3e50', lw=2)
plt.plot(normal_profile.index.astype(str), normal_profile.values, label='Normal Days (Temp Matched)', color='#e67e22', lw=2)

plt.title(f'Temperature-Matched Profile Comparison ({avg_temp_dark:.1f}°C)')
plt.ylabel('Consumption (kWh)')
plt.xticks(dark_profile.index.astype(str)[::4], rotation=45)
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
# %%
prob_score, minutes_shifted = calculate_battery_score(dark_profile, normal_profile)
print(f"Battery Probability Score: {prob_score}/100")
print(f"The evening peak is delayed by {minutes_shifted} minutes on sunny/normal days.")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# Custom module imports
import envdata
import load_smart_meter
import preprocessing

def calculate_battery_logistic_prob(profile_dark, profile_norm, p_val):
    """
    Applies a Logistic Regression framework to physical evening profiles.
    Returns a probability percentage.
    """
    # 1. Feature Extraction: Peak Shift (X1)
    # How many 15-min intervals did the peak move?
    peak_idx_dark = np.argmax(profile_dark.values)
    peak_idx_norm = np.argmax(profile_norm.values)
    x1_shift_mins = (peak_idx_norm - peak_idx_dark) * 15
    
    # 2. Feature Extraction: Energy Substitution Ratio (X2)
    # Area Under Curve (AUC) difference between 16:00 and 22:00
    auc_dark = np.trapz(profile_dark.values)
    auc_norm = np.trapz(profile_norm.values)
    x2_gap_ratio = (auc_dark - auc_norm) / auc_dark if auc_dark > 0 else 0

    # 3. Logistic Regression Coefficients (Beta)
    # These are calibrated for residential battery behavior
    # intercept (b0), shift_weight (b1), gap_weight (b2)
    b0 = -4.5  # Bias (assumes batteries are relatively rare)
    b1 = 0.08  # ~1.2 hour shift makes battery very likely
    b2 = 5.0   # ~30% energy gap makes battery very likely
    
    # Linear Combination (z)
    z = b0 + (b1 * x1_shift_mins) + (b2 * x2_gap_ratio)
    
    # Logistic Function (Sigmoid)
    # P(y=1) = 1 / (1 + e^-z)
    probability = 1 / (1 + np.exp(-z))
    
    # 4. Statistical Adjustment
    # If the difference isn't significant (p > 0.05), we penalize the probability
    if p_val > 0.05:
        probability *= 0.5 

    return {
        "probability": round(probability * 100, 2),
        "features": {
            "shift_mins": x1_shift_mins,
            "gap_ratio": round(x2_gap_ratio, 3),
            "significance_p": round(p_val, 4)
        }
    }


def analyze_customer_battery(file_path, weather_df, temp_buffer=2.0, rad_threshold=100):
    """
    Analyzes a single PV-customer to detect battery storage via 
    temperature-matched evening profiles (16:00 - 24:00).
    """
    # --- 1. Load and Standardize Data ---
    df_raw = load_smart_meter.load_customer_data(file_path)
    if df_raw.empty:
        return None
    
    # Preprocessing (15-min resampling)
    df_customer = preprocessing.clean_and_resample(df_raw)
    
    # Set Index and fix Precision Unit Mismatch (ms vs us)
    if 'DT_UTC' in df_customer.columns:
        df_customer = df_customer.set_index('DT_UTC')
        df_customer.index = df_customer.index.as_unit('ms')
    

    df_customer.index = df_customer.index.as_unit('ms')
    df_customer = df_customer.sort_index()

    # --- 2. Merge Weather Data ---
    merged = pd.merge_asof(
        df_customer, 
        weather_df, 
        left_index=True, 
        right_index=True, 
        direction='backward'
    )

    # --- 3. Temperature-Matched Filtering ---
    # Create daily metrics to find comparable thermal loads
    daily = merged.resample('D').agg({
        't_2m_C': 'mean',
        'global_rad_W': 'max'
    }).dropna()

    # Define Dark Days (Gloomy/Winter-like)
    dark_days_idx = daily[daily['global_rad_W'] < rad_threshold].index
    if len(dark_days_idx) < 2:
        return {"error": "Insufficient dark days found"}

    avg_temp_dark = daily.loc[dark_days_idx, 't_2m_C'].mean()

    # Find Normal Days (Sunny) with the SAME average temperature
    matched_normal_idx = daily[
        (daily['global_rad_W'] >= rad_threshold) & 
        (daily['t_2m_C'].between(avg_temp_dark - temp_buffer, avg_temp_dark + temp_buffer))
    ].index

    if len(matched_normal_idx) < 2:
        return {"error": f"No normal days found matching {avg_temp_dark:.1f}C"}

    # --- 4. Extract Aggregated Evening Profiles (16:00 - 24:00) ---
    is_dark = pd.Series(merged.index.date).isin(dark_days_idx.date).values
    is_norm = pd.Series(merged.index.date).isin(matched_normal_idx.date).values

    dark_ev = merged[is_dark].between_time('16:00', '23:59')
    norm_ev = merged[is_norm].between_time('16:00', '23:59')

    profile_dark = dark_ev.groupby(dark_ev.index.time)['CONSO_KWH'].mean()
    profile_norm = norm_ev.groupby(norm_ev.index.time)['CONSO_KWH'].mean()

    # --- 5. Battery Metrics ---
    # A: Peak Shift (Batteries delay grid peak)
    peak_idx_dark = np.argmax(profile_dark.values)
    peak_idx_norm = np.argmax(profile_norm.values)
    shift_mins = (peak_idx_norm - peak_idx_dark) * 15

    # B: Ramp Difference (16:00 to 18:00)
    # If the surge at sunset is 'suppressed' on sunny days, it indicates a battery
    ramp_dark = profile_dark.iloc[8] - profile_dark.iloc[0]
    ramp_norm = profile_norm.iloc[8] - profile_norm.iloc[0]
    ramp_ratio = ramp_dark / max(ramp_norm, 0.01)

    # C: Statistical Test (Energy Gap 16:00 - 20:00)
    energy_dark = dark_ev.between_time('16:00', '20:00').groupby(level=0).sum()['CONSO_KWH']
    energy_norm = norm_ev.between_time('16:00', '20:00').groupby(level=0).sum()['CONSO_KWH']
    _, p_val = ranksums(energy_dark, energy_norm)

    #-----5.5. Battery Probability Score (Logistic Regression)-----
    # --- Implementation Example ---
    # Assuming 'profile_dark', 'profile_norm', and 'p_val' are ready from previous steps:
    result = calculate_battery_logistic_prob(profile_dark, profile_norm, p_val)

    print(f"Logistic Probability of Battery: {result['probability']}%")
    print(f"Shift Detected: {result['features']['shift_mins']} minutes")
    print(f"Energy Substitution: {result['features']['gap_ratio']*100}%")


    # --- 6. Scoring Logic ---
    score = 0
    if shift_mins >= 45: score += 40 
    if ramp_ratio > 1.5: score += 30 
    if p_val < 0.05: score += 30    

    # --- 7. Visualization ---
    plt.figure(figsize=(12, 6))
    x_times = [t.strftime('%H:%M') for t in profile_dark.index]
    
    plt.plot(x_times, profile_dark.values, label='Dark Day (Grid Only)', color='#2c3e50', lw=2)
    plt.plot(x_times, profile_norm.values, label='Normal Day (Solar+Battery)', color='#e67e22', lw=2, ls='--')
    
    plt.fill_between(x_times, profile_dark.values, profile_norm.values, color='orange', alpha=0.1)
    
    plt.title(f"Battery Detection (Temp Match: {avg_temp_dark:.1f}°C)\nScore: {score}/100 | Peak Shift: {shift_mins}m")
    plt.ylabel('Consumption (kWh)')
    plt.xticks(x_times[::4], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    return {
        "score": score,
        "p_value": p_val,
        "shift": shift_mins,
        "temp_matched": avg_temp_dark
    }


