#%%
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# 1. Force the DLL directory for Windows/Conda stability
env_bin = r"C:\Users\Aline\anaconda3\envs\MEST_RE_25\Library\bin"
if os.path.exists(env_bin):
    os.add_dll_directory(env_bin)

# 2. Setup paths for repo imports
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

    def forward(self, x, adj):
        x = torch.relu(self.gcn(x))
        _, (h_n, _) = self.encoder_lstm(x)
        encoded = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        decoded_input = encoded.unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder_lstm(decoded_input)
        return self.output_layer(reconstructed)

# --- 2. DATASET CLASS ---
class SmartMeterDataset(Dataset):
    def __init__(self, data, seq_length=96):
        self.sequences = []
        self.customer_ids = []
        data['date'] = pd.to_datetime(data['dt_local']).dt.date
        grouped = data.groupby(['id_customer', 'date'])
        
        for (c_id, date), group in grouped:
            if len(group) == seq_length:
                group = group.sort_values('dt_local')
                feats = group[['value_kw_mean', 'glob_rad', 'temp']].values
                self.sequences.append(torch.FloatTensor(feats))
                self.customer_ids.append(c_id)
                
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.customer_ids[idx], self.sequences[idx]

# --- 3. TRAINING FUNCTION ---
def train_and_calibrate(df_train):
    print("Pre-processing training data...")
    # Filter for baseline households
    #df_train = df_train[df_train['source'].isin(['ECO', 'REFIT'])].copy()
    df_train = df_train[df_train['type'].isin(['TOT'])].copy()
    scaler = MinMaxScaler()
    df_train[['value_kw_mean', 'glob_rad', 'temp']] = scaler.fit_transform(
        df_train[['value_kw_mean', 'glob_rad', 'temp']]
    )
    
    loader = DataLoader(SmartMeterDataset(df_train), batch_size=32, shuffle=True)
    model = PowerAutoencoder(feature_dim=3, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Starting training on {len(loader.dataset)} daily sequences...")
    model.train()
    for epoch in range(15):
        total_loss = 0
        for _, batch_data in loader:
            optimizer.zero_grad()
            output = model(batch_data, torch.eye(1))
            loss = criterion(output, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/15 - Loss: {total_loss/len(loader):.6f}")
    
    model.eval()
    errors = []
    with torch.no_grad():
        for _, batch_data in loader:
            recon = model(batch_data, torch.eye(1))
            mse = torch.mean((recon[:, :, 0] - batch_data[:, :, 0]) ** 2, dim=1)
            errors.extend(mse.tolist())
    
    threshold = np.percentile(errors, 80)
    return model, scaler, threshold



#%%
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gc
import joblib
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
FORCE_RETRAIN = False  # Set to True if you want to ignore saved models
MODEL_PATH = "power_autoencoder.pth"
SCALER_PATH = "data_scaler.pkl"
META_PATH = "model_metadata.joblib"
DIR = r"C:\Users\Aline\Documents\Studium\Case Study\processed_data"

# 1. Windows DLL Fix
env_bin = r"C:\Users\Aline\anaconda3\envs\MEST_RE_25\Library\bin"
if os.path.exists(env_bin):
    os.add_dll_directory(env_bin)

# 2. Repo Pathing
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

    def forward(self, x, adj):
        x = torch.relu(self.gcn(x))
        _, (h_n, _) = self.encoder_lstm(x)
        encoded = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        decoded_input = encoded.unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder_lstm(decoded_input)
        return self.output_layer(reconstructed)

# --- 2. HELPERS ---
def save_assets(model, scaler, threshold):
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({'threshold': threshold}, META_PATH)

def load_assets():
    if not os.path.exists(MODEL_PATH) or FORCE_RETRAIN:
        return None, None, None
    model = PowerAutoencoder(3, 64)
    model.load_state_dict(torch.load(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    return model, scaler, meta['threshold']

# --- 3. UPDATED TRAINING ---
def train_and_calibrate(df_train, weather_df):
   # 1. Identify the time column (we know it's 'dt_local' now)
    time_col = 'dt_local'
    """
    # 2. DROP existing weather columns to avoid temp_x / temp_y confusion
    cols_to_drop = [c for c in ['temp', 'glob_rad'] if c in df_train.columns]
    if cols_to_drop:
        print(f"Dropping existing weather columns: {cols_to_drop}")
        df_train = df_train.drop(columns=cols_to_drop)
    """
    # 3. Standardize time (Naive)
    df_train[time_col] = pd.to_datetime(df_train[time_col]).dt.tz_localize(None)
    
    """# 4. Merge fresh Swiss weather
    df_train = df_train.merge(
        weather_df, 
        left_on=time_col, 
        right_index=True, 
        how="left"
    )
    """
    # 5. Rename consumption if needed
    if 'CONSO_KWH' in df_train.columns:
        df_train = df_train.rename(columns={'CONSO_KWH': 'value_kw_mean'})
    # 2. Standardize Training Time (Match the weather index format)
    df_train[time_col] = pd.to_datetime(df_train[time_col]).dt.tz_localize(None)
 
    """src_col = 'source' 
    df_baseline = df_train[df_train[src_col].isin(['ECO', 'REFIT'])].copy()
    """
    src_col = 'type' 
    df_baseline = df_train[df_train[src_col].isin(['TOT'])].copy()
    
    if df_baseline.empty:
        raise ValueError(f"No TOT data found in column '{src_col}'!")

    # 5. Scaling
    scaler = MinMaxScaler()
    # We use the names already present in weather_df
    cols = ['value_kw_mean', 'glob_rad', 'temp']
    df_baseline[cols] = scaler.fit_transform(df_baseline[cols])
    
    
    # 7. Model Training Loop
    loader = DataLoader(SmartMeterDataset(df_baseline), batch_size=32, shuffle=True)
    model = PowerAutoencoder(3, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {len(df_baseline)} baseline rows...")
    model.train()
    for epoch in range(15):
        for _, batch_data in loader:
            optimizer.zero_grad()
            output = model(batch_data, torch.eye(1))
            loss = nn.MSELoss()(output, batch_data)
            loss.backward()
            optimizer.step()
    
    # 8. Threshold Calculation (80th Percentile)
    model.eval()
    errors = []
    with torch.no_grad():
        for _, batch_data in loader:
            recon = model(batch_data, torch.eye(1))
            mse = torch.mean((recon[:, :, 0] - batch_data[:, :, 0]) ** 2, dim=1)
            errors.extend(mse.tolist())
    
    threshold = np.nanpercentile(errors, 80)
    return model, scaler, threshold


# --- Safe Weather Preparation ---
def prepare_weather(df):
    # Print columns to your console so you can see exactly what's there if it fails
    # print(f"DEBUG: Weather columns found: {df.columns.tolist()}")
    
    # Map whatever the names are to our internal model names
    mapping = {
        't_2m_C': 'temp', 
        'global_rad_W': 'glob_rad',
        'temp': 'temp',        # case-safety
        'glob_rad': 'glob_rad' # case-safety
    }
    
    # Rename only the columns that actually exist in the mapping
    df = df.rename(columns=mapping)
    
    # Standardize the index (the timestamp)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    return df[['temp', 'glob_rad']] # Only return what the model needs


# --- 4. MAIN ---
if __name__ == "__main__":
    # 1. Load Model, Scaler, and Threshold
    model, fitted_scaler, anomaly_threshold = load_assets()

    # 2. Prepare Weather Data ONCE (to be used for training OR detection)
    print("Downloading and preparing MeteoSwiss weather data...")
    _, raw_weather = meteo.env_data()
    weather_df = prepare_weather(raw_weather)
    weather_df.to_parquet(os.path.join(DIR, 'prepared_weather.parquet'))  # Save for future use
    print(f"Weather Ready. Columns: {weather_df.columns.tolist()}") 

    if model is None:
        print("Training assets not found. Training new model...")
        df_normal = pd.read_parquet(os.path.join(DIR, 'all_sources_load_with_weather.parquet')).copy()
        # Pass the prepared weather_df into training to ensure alignment
        model, fitted_scaler, anomaly_threshold = train_and_calibrate(df_normal, weather_df)
        save_assets(model, fitted_scaler, anomaly_threshold)

    # 3. DETECTION PHASE
    results_collector = []
    # Adjust batch_size based on your RAM; 2-4 is usually safe for parallel loading
    data_gen = re_data.load_all_data_parallel_generator(os.path.join(DIR, 'ETHZ_ALL'), batch_size=2)

    model.eval()
    print("Starting Detection Phase...")

    
    MAX_TEST = 100
    processed = 0

    for batch_idx, batch_df in enumerate(data_gen):
        if batch_df.empty:
            continue
        if processed >= MAX_TEST: break
        # A. Standardize Time (Naive UTC to match Weather Index)
        batch_df['DT_UTC'] = pd.to_datetime(batch_df['DT_UTC']).dt.tz_localize(None)

        # B. Merge Weather (Fresh Swiss data)
        batch_df = batch_df.merge(
            weather_df, 
            left_on="DT_UTC", 
            right_index=True, 
            how="left"
        )

        # C. Feature Preparation
        batch_df['date'] = batch_df['DT_UTC'].dt.date
        batch_df['dt_local'] = batch_df['DT_UTC']
        
        # ID column normalization
        id_col = 'ID' if 'ID' in batch_df.columns else 'id_customer'
        
        # Rename power column if it comes from Romande Energie Parquet as CONSO
        if 'CONSO_KWH' in batch_df.columns:
            batch_df = batch_df.rename(columns={'CONSO_KWH': 'value_kw_mean'})

        # D. Clean Missing Weather (Interpolate gaps)
        cols_to_fix = [c for c in ['temp', 'glob_rad'] if c in batch_df.columns]
        if len(cols_to_fix) < 2:
            print(f"Skipping batch: Missing weather columns. Found {batch_df.columns.tolist()}")
            continue
        batch_df[cols_to_fix] = batch_df[cols_to_fix].ffill().bfill()

        # E. Process Groups (Customer + Day)
        # feature_cols order MUST match the order used in train_and_calibrate
        feature_cols = ['value_kw_mean', 'glob_rad', 'temp']
        
        batch_suspects = 0
        for (c_id, date), day_data in batch_df.groupby([id_col, 'date']):
            if len(day_data) == 96:
                day_data = day_data.sort_values("dt_local")
                
                # F. Scale (Using DataFrame to avoid feature name warnings)
                day_data_to_scale = pd.DataFrame(day_data[feature_cols].values, columns=feature_cols)
                scaled = fitted_scaler.transform(day_data_to_scale)
                
                # G. Inference
                input_tensor = torch.FloatTensor(scaled).unsqueeze(0)
                with torch.no_grad():
                    # GCN-BiLSTM expects (batch, seq, feat) and adjacency (identity for single nodes)
                    recon = model(input_tensor, torch.eye(1))
                    
                    # Calculate error on column 0 (Load)
                    err = (recon[0, :, 0] - input_tensor[0, :, 0]) ** 2
                    mean_err = torch.mean(err).item()
                    num_events = (err > anomaly_threshold).sum().item()
                
                # H. Criteria: At least 1 hour (4 intervals) of anomalous behavior
                if num_events >= 4:
                    batch_suspects += 1
                    results_collector.append({
                        "ID": c_id, 
                        "Date": date, 
                        "Confidence": round(mean_err / anomaly_threshold, 2),
                        "Events": int(num_events),
                        "Mean_MSE": round(mean_err, 6)
                    })
        
      
        processed += 1
        if processed % 10 == 0: print(f"Progress: {processed}/{MAX_TEST}")

        if processed >= MAX_TEST: break
        del batch_df
        gc.collect()
        print(f"Batch {batch_idx+1} processed. Found {batch_suspects} anomalous days.")

        


    # 4. FINAL AGGREGATION
    if results_collector:
        final_df = pd.DataFrame(results_collector)
        # Aggregate by ID: count anomalous days and average the confidence
        summary = final_df.groupby('ID').agg({
            'Date': 'count', 
            'Confidence': 'mean',
            'Events': 'mean'
        }).rename(columns={'Date': 'Anomalous_Days'})
        
        # Calculate final suspicion score
        summary['Score'] = (summary['Anomalous_Days'] * summary['Confidence']).round(2)
        summary = summary.sort_values('Score', ascending=False)
        
        summary.to_csv("TOP_BATTERY_SUSPECTS.csv")
        print(f"\nDetection complete. Found {len(summary)} potential battery owners.")
        print("Results saved to TOP_BATTERY_SUSPECTS.csv")
    else:
        print("\nNo anomalies detected across the dataset.")
# %%
