import requests
import pandas as pd
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
API_TOKEN = 'FYwu13iH11r7fJB_iXOvP3hwjURr3CWyITcGkmYwVwE'  # Register at ev.caltech.edu/dataset
SITE_ID = 'caltech'            # 'caltech' or 'jpl'
BASE_URL = "https://ev.caltech.edu/api/v1"
FILENAME = "processed_data/caltech_ev_15min_full.parquet"
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

def get_last_timestamp(file_path):
    """Checks the existing parquet file for the most recent record."""
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        if not df.empty:
            # Ensure it's in datetime format to find the max
            return pd.to_datetime(df['dt_utc']).max()
    return None

def fetch_incremental_data():
    last_ts = get_last_timestamp(FILENAME)
    
    # Build the initial URL with a filter if we have a checkpoint
    if last_ts:
        # The API requires RFC 1123 date format for 'where' queries
        ts_str = last_ts.strftime('%a, %d %b %Y %H:%M:%S GMT')
        print(f"Checking for new data since: {ts_str}")
        query = f'where=connectionTime>="{ts_str}"'
        current_url = f"{BASE_URL}/sessions/{SITE_ID}/ts?{query}"
    else:
        print("No existing file found. Starting full history download...")
        current_url = f"{BASE_URL}/sessions/{SITE_ID}/ts"

    all_new_dfs = []
    
    while current_url:
        try:
            response = requests.get(current_url, headers=HEADERS)
            
            if response.status_code == 429:
                print("Rate limited by Caltech. Sleeping for 30s...")
                time.sleep(30); continue
            elif response.status_code != 200:
                print(f"API Error: {response.status_code}"); break
            
            data = response.json()
            items = data.get('_items', [])
            
            for session in items:
                # We use 'chargingCurrent' (Amps) as the primary time-series source
                if 'chargingCurrent' in session and session['chargingCurrent']:
                    # entry[0] = Unix timestamp, entry[1] = Amps
                    df_s = pd.DataFrame(session['chargingCurrent'], columns=['dt_utc', 'amps'])
                    df_s['dt_utc'] = pd.to_datetime(df_s['dt_utc'], unit='s', utc=True)
                    
                    # kW Calculation: (Amps * 240V) / 1000
                    df_s['Value_KW_mean'] = (df_s['amps'] * 240) / 1000
                    
                    # 15-minute Resampling (Mean power per window)
                    resampled = df_s.set_index('dt_utc').resample('15min').mean().reset_index()
                    
                    # Fill metadata
                    resampled['Source'] = 'Caltech EV'
                    resampled['ID customer'] = session.get('user_id', 'Unknown')
                    resampled['type'] = 'EV'
                    
                    all_new_dfs.append(resampled)

            # Follow pagination link
            next_link = data.get('_links', {}).get('next')
            current_url = f"{BASE_URL}/{next_link}" if next_link else None
            
            if len(all_new_dfs) % 50 == 0 and len(all_new_dfs) > 0:
                print(f"Fetched {len(all_new_dfs)} new sessions...")

        except Exception as e:
            print(f"Fetch failed: {e}"); break

    # --- MERGE AND SAVE ---
    if all_new_dfs:
        new_data = pd.concat(all_new_dfs, ignore_index=True)
        
        if os.path.exists(FILENAME):
            existing_data = pd.read_parquet(FILENAME)
            # Combine and remove overlap (just in case)
            final_df = pd.concat([existing_data, new_data]).drop_duplicates(subset=['ID customer', 'dt_utc'])
        else:
            final_df = new_data
            
        # Enforce exact column order from your requirements
        final_df = final_df[['Source', 'ID customer', 'type', 'Value_KW_mean', 'dt_utc']]
        final_df.to_parquet(FILENAME, engine='pyarrow', index=False)
        print(f"Success! {FILENAME} updated. Total rows: {len(final_df)}")
    else:
        print("No new data found on server.")

if __name__ == "__main__":
    fetch_incremental_data()