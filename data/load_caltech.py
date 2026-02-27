import requests
import pandas as pd
import os
import time
from datetime import datetime

#session ID -> customer ID (only 1 charging event per customer ID)
#spaceID -> customer ID (parking space identifier, which is more consistent with the idea of a "customer" in this context)
#data is not measured in kW but in amps, so we need to convert it using the voltage (208V for Caltech):
#amps * voltage -> Val_KW_mean (in kW)
#all data is for EV charging 


# --- CONFIGURATION ---
TOKEN = 'FYwu13iH11r7fJB_iXOvP3hwjURr3CWyITcGkmYwVwE' 
SITE = 'caltech'
VOLTAGE = 208 
OUTPUT_PATH = "data/preprocessed_data/caltech_15min_kw.parquet"
MAX_SESSIONS = 50 # Adjust this number based on how many sessions you want to fetch (max 1000 for free tier)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def fetch_and_preprocess():
    url = f"https://ev.caltech.edu/api/v1/sessions/{SITE}/ts"
    all_rows = []
    page = 1
    
    print(f"Starting fetch (Limit: {MAX_SESSIONS} sessions)...")
    
    while page <= MAX_SESSIONS:
        response = requests.get(url, params={'page': page}, auth=(TOKEN, ''))
        if response.status_code != 200:
            print(f"Error at page {page}: {response.status_code}")
            break
            
        data = response.json()
        items = data.get('_items', [])
        if not items: break
            
        session = items[0]
        
        # ADJUSTMENT: Using spaceID instead of sessionID for the customer identifier
        space_id = session.get('spaceID')
        
        # Accessing the time series of measured current draw
        ts_data = session.get('chargingCurrent', {})
        times = ts_data.get('timestamps', [])
        amps = ts_data.get('current', []) 

        for t, a in zip(times, amps):
            all_rows.append({
                'dt_utc': t,
                'ID customer': space_id, # This now stores the space ID
                'Value_KW_mean': (a * VOLTAGE) / 1000.0
            })
            
        if page % 10 == 0:
            print(f"Progress: {page}/{MAX_SESSIONS} sessions collected...")
            
        if 'next' not in data.get('_links', {}): break
        page += 1
        time.sleep(0.1)

    if not all_rows:
        print("No data collected. Please verify your token and siteID.")
        return

    # --- Formatting to match schema ---
    df = pd.DataFrame(all_rows)
    df['dt_utc'] = pd.to_datetime(df['dt_utc'], utc=True)
    
    # Resample to 15-minute intervals per space ID to get the mean kW
    df_final = (
        df.groupby('ID customer')
        .resample('15min', on='dt_utc')
        .mean(numeric_only=True)
        .reset_index()
    )
    
    # Add constant labels
    df_final['Source'] = 'Caltech'
    df_final['type'] = 'EV'
    
    # Apply strict column ordering
    column_order = ['Source', 'ID customer', 'type', 'Value_KW_mean', 'dt_utc']
    df_final = df_final[column_order]

    # Save to Parquet format
    df_final.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
    print(f"Success! {len(df_final)} rows stored in {OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_and_preprocess()
    