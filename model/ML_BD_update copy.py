import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import joblib
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
FORCE_RETRAIN = False  
THRESHOLD_PERCENTILE = 60  # Lower = more sensitive to potential batteries
MODEL_PATH = "power_autoencoder.pth"
SCALER_PATH = "data_scaler.pkl"
META_PATH = "model_metadata.joblib"
DIR = r"C:\Users\Aline\Documents\Studium\Case Study\processed_data"
WEATHER_CACHE_PATH = os.path.join(DIR, 'prepared_weather.parquet')

# 1. Environment Pathing
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import data.envdata as meteo
import data.fast_load_smart_meter as re_data

# --- 1. MODEL ARCHITECTURE ---
class PowerAutoencoder(nn.Module):
    def __init__(self, feature_dim=3, hidden_dim=64):
        super(PowerAutoencoder, self).__init__()
        self.gcn = nn.Linear(feature_dim, hidden_dim) 
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        x = torch.relu(self.gcn(x))
        _, (h_n, _) = self.encoder_lstm(x)
        encoded = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        decoded_input = encoded.unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder_lstm(decoded_input)
        return self.output_layer(reconstructed)

# --- 2. DATASET CLASS (For Training) ---
class SmartMeterDataset(Dataset):
    def __init__(self, data, seq_length=96):
        self.sequences = []
        data = data.copy()
        data['date_only'] = pd.to_datetime(data['dt_local']).dt.date
        id_col = 'id_customer' if 'id_customer' in data.columns else 'ID'
        
        grouped = data.groupby([id_col, 'date_only'])
        for _, group in grouped:
            if len(group) == seq_length:
                group = group.sort_values('dt_local')
                feats = group[['value_kw_mean', 'glob_rad', 'temp']].values
                self.sequences.append(torch.FloatTensor(feats))
                
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]

# --- 3. HELPERS ---
def save_assets(model, scaler, threshold):
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({'threshold': threshold, 'percentile': THRESHOLD_PERCENTILE}, META_PATH)

def load_assets():
    if not os.path.exists(MODEL_PATH) or FORCE_RETRAIN:
        return None, None, None
    model = PowerAutoencoder(3, 64)
    model.load_state_dict(torch.load(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    return model, scaler, meta['threshold']

def prepare_weather(df):
    mapping = {'t_2m_C': 'temp', 'global_rad_W': 'glob_rad'}
    df = df.rename(columns=mapping)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['temp', 'glob_rad']]

# --- 4. TRAINING & CALIBRATION ---
def train_and_calibrate(df_train, percentile=75):
    df_train['dt_local'] = pd.to_datetime(df_train['dt_local']).dt.tz_localize(None)
    
    # Filter for Baseline (TOT)
    type_col = 'type' if 'type' in df_train.columns else 'type_measurement'
    df_baseline = df_train[df_train[type_col].str.upper() == 'TOT'].copy() if type_col in df_train.columns else df_train.copy()
    if 'CONSO_KWH' in df_baseline.columns:
        df_baseline = df_baseline.rename(columns={'CONSO_KWH': 'value_kw_mean'})

    scaler = MinMaxScaler()
    cols = ['value_kw_mean', 'glob_rad', 'temp']
    df_baseline[cols] = scaler.fit_transform(df_baseline[cols])
    
    loader = DataLoader(SmartMeterDataset(df_baseline), batch_size=32, shuffle=True)
    model = PowerAutoencoder(3, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(loader.dataset)} days...")
    model.train()
    for epoch in range(10):
        for batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            recon = model(batch)
            mse = torch.mean((recon[:, :, 0] - batch[:, :, 0]) ** 2, dim=1)
            errors.extend(mse.tolist())
    
    return model, scaler, np.nanpercentile(errors, percentile)

# --- 5. PLOTTING ---
def plot_battery_results(results_list, threshold):
    if not results_list: return
    df = pd.DataFrame(results_list)
    df['anomaly_score'] = df['MSE'] / threshold
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    suspect_count = df['ID'].nunique()
    total_users = 12034 
    
    axes[0].pie([suspect_count, total_users - suspect_count], labels=['Suspect', 'Normal'], 
                autopct='%1.1f%%', colors=['#E15759', '#4A90E2'], startangle=90)
    axes[0].set_title("Detection Ratio")

    sns.histplot(data=df, x='anomaly_score', bins=30, ax=axes[1], color='#E15759', kde=True)
    axes[1].axvline(1.0, color='black', linestyle='--')
    axes[1].set_title("Anomaly Score Distribution")
    plt.show()

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Weather Setup
    if os.path.exists(WEATHER_CACHE_PATH):
        weather_df = pd.read_parquet(WEATHER_CACHE_PATH)
    else:
        _, raw_weather = meteo.env_data()
        weather_df = prepare_weather(raw_weather)
        weather_df.to_parquet(WEATHER_CACHE_PATH)

    # B. Model Load/Train
    model, fitted_scaler, anomaly_threshold = load_assets()
    if model is None:
        train_file = os.path.join(DIR, 'all_sources_load_with_weather.parquet')
        df_train_raw = pd.read_parquet(train_file)
        model, fitted_scaler, anomaly_threshold = train_and_calibrate(df_train_raw, THRESHOLD_PERCENTILE)
        save_assets(model, fitted_scaler, anomaly_threshold)

    # C. DETECTION PHASE (Optimized for 20k IDs)
    results_collector = []
    data_gen = re_data.load_all_data_parallel_generator(os.path.join(DIR, 'ETHZ_ALL'), batch_size=2)
    model.eval()
    feature_cols = ['value_kw_mean', 'glob_rad', 'temp']

    print(f"Detection started (Threshold: {anomaly_threshold:.6f})...")

    for batch_idx, batch_df in enumerate(data_gen):
        if batch_df.empty: continue
        
        batch_df['DT_UTC'] = pd.to_datetime(batch_df['DT_UTC']).dt.tz_localize(None)
        if 'CONSO_KWH' in batch_df.columns:
            batch_df = batch_df.rename(columns={'CONSO_KWH': 'value_kw_mean'})
        print(batch_df['DT_UTC'])
        batch_df = batch_df.merge(weather_df, left_on="DT_UTC", right_index=True, how="left")
        batch_df[feature_cols[1:]] = batch_df[feature_cols[1:]].ffill().bfill()
        print(weather_df.head())
        # VECTORIZED SCALING (Huge speedup)
        batch_df[feature_cols] = fitted_scaler.transform(batch_df[feature_cols])
        
        id_col = 'ID' if 'ID' in batch_df.columns else 'id_customer'
        batch_df['date_only'] = batch_df['DT_UTC'].dt.date
        
        # GROUP BY DAY
        day_tensors = []
        day_metadata = []

        for (c_id, date), day_data in batch_df.groupby([id_col, 'date_only']):
            if len(day_data) == 96:
                day_values = day_data.sort_values("DT_UTC")[feature_cols].values
                day_tensors.append(day_values)
                day_metadata.append((c_id, date))

        # BATCHED INFERENCE
        if day_tensors:
            input_batch = torch.FloatTensor(np.array(day_tensors))
            with torch.no_grad():
                recon = model(input_batch)
                err_per_step = (recon[:, :, 0] - input_batch[:, :, 0]) ** 2
                mean_errors = torch.mean(err_per_step, dim=1)
                spike_counts = (err_per_step > anomaly_threshold).sum(dim=1)

            # Record Anomalies
            for i in range(len(mean_errors)):
                m_err = mean_errors[i].item()
                spikes = spike_counts[i].item()
                if m_err > anomaly_threshold or spikes >= 4:
                    c_id, date = day_metadata[i]
                    results_collector.append({
                        "ID": c_id, "Date": date, "MSE": round(m_err, 6),
                        "Confidence": round(m_err / anomaly_threshold, 2), "Spikes": spikes
                    })

        if (batch_idx + 1) % 5 == 0: print(f"Processed {batch_idx + 1} batches...")
        del batch_df; gc.collect()

    # D. Final Summary
    if results_collector:
        final_df = pd.DataFrame(results_collector)
        summary = final_df.groupby('ID').agg({'Date': 'count', 'Confidence': 'mean', 'Spikes': 'mean'})
        summary['Score'] = (summary['Date'] * summary['Confidence']).round(2)
        summary = summary.sort_values('Score', ascending=False)
        summary.to_csv("TOP_BATTERY_SUSPECTS.csv")
        print(f"\nFound {len(summary)} potential battery owners.")
        print(summary.head(10))
        plot_battery_results(results_collector, anomaly_threshold)
    else:
        print("\nNo anomalies detected.")