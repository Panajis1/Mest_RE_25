"Load Romande Energie smart meter parquet data with parallel processing and robust error handling."

import pandas as pd
import glob
import os
import gc
import logging
from typing import Generator, List, Set
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# 1. Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_loading.log"), # Saves to a file
        logging.StreamHandler()                 # Also prints to console
    ]
)
logger = logging.getLogger(__name__)

def process_single_file(file_path: str, valid_ids_set: Set[str], target_meta: pd.DataFrame) -> pd.DataFrame:
    """Worker function with detailed error logging."""
    try:
        df = pd.read_parquet(
            file_path, 
            columns=["ID", "DT_UTC", "CONSO_KWH", "PROD_KWH"],
            engine='pyarrow'
        )
        
        if df.empty:
            logger.warning(f"File skipped (Empty): {file_path}")
            return pd.DataFrame()

        df["ID"] = df["ID"].astype(str)
        filtered_df = df[df["ID"].isin(valid_ids_set)].copy()
        
        if filtered_df.empty:
            # Not an error, just a file with no matching customer types
            return pd.DataFrame()

        filtered_df["DT_UTC"] = pd.to_datetime(filtered_df["DT_UTC"])
        return filtered_df.merge(target_meta, on="ID", how="left")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return pd.DataFrame()


# Call the function using only the directory path
# If problems with ram occur, adjust the batch_size=1 when calling the function 
#Default function filters only for individual housholds named "Particuliers"

#use following code snippet to test the function and get a quick overview of the data. Adjust the batch_size for faster loading if needed.
"""
results_collector = []

data_gen = load_all_data_parallel_generator(data_dir, partner_type="Particuliers", batch_size=2)

for batch_df in data_gen:
    # 1. Update Global Metrics (e.g., Fleet Average)
    # batch_df["CONSO_KWH"].mean() 

    # 2. Process Individual Customers
    for customer_id, customer_data in batch_df.groupby("ID"):
        
        # --- PREPARE ---
        # Ensure DT_UTC is the index for time-based battery logic
        customer_data = customer_data.set_index("DT_UTC").sort_index()

        # --- ANALYZE ---
               
        # Run your battery probability logic
        # analysis = call application_detection_function(customer_data, weather_df)
        # 
        # --- STORE ---
        results_collector.append({
            "ID": customer_id,
            "is_zero_production": is_zero_prod,
            "avg_conso": customer_data["CONSO_KWH"].mean(),
            # "battery_prob": analysis["prob"]
        })

    # 3. CLEANUP (Crucial for Windows)
    del batch_df
    gc.collect()

# 4. SAVE FINAL OUTPUT
final_results = pd.DataFrame(results_collector)
final_results.to_parquet("final_analysis_report.parquet")
""" 

def load_all_data_parallel_generator(
    data_dir: str, 
    partner_type: str = "Particuliers",
    batch_size: int = 2
) -> Generator[pd.DataFrame, None, None]:
    
    logger.info(f"Starting data load for type: {partner_type}")

    # Load Metadata
    try:
        meta_df = pd.read_parquet(os.path.join(data_dir, "metadata"), engine='pyarrow')
        target_meta = meta_df[meta_df["TYPE_PARTENAIRE_LIBELLE"] == partner_type].copy()
        target_meta["ID"] = target_meta["ID"].astype(str)
        valid_ids_set = set(target_meta["ID"])
        logger.info(f"Metadata loaded. Found {len(valid_ids_set)} valid IDs for {partner_type}.")
    except Exception as e:
        logger.critical(f"Could not load metadata: {e}")
        return

    files = [f for f in glob.glob(os.path.join(data_dir, "*.parquet")) if "metadata" not in f.lower()]
    
    if not files:
        logger.error(f"No parquet files found in {data_dir}")
        return

    n_cores = max(1, cpu_count()//2)  # Leave one core free
    batch_size = batch_size or n_cores
    worker_func = partial(process_single_file, valid_ids_set=valid_ids_set, target_meta=target_meta)

    with tqdm(total=len(files), desc="Overall Progress") as pbar:
        with Pool(processes=n_cores) as pool:
            for i in range(0, len(files), batch_size):
                file_chunk = files[i : i + batch_size]
                
                results = pool.map(worker_func, file_chunk)
                pbar.update(len(file_chunk))
                
                batch_df = pd.concat([res for res in results if not res.empty], ignore_index=True)
                
                if not batch_df.empty:
                    yield batch_df
                
                del results
                gc.collect()

    logger.info("Data loading sequence complete.")

# --- EXECUTION ---
if __name__ == "__main__":
    DIR = "./data_folder"
    for batch in load_all_data_parallel_generator(DIR):
        # Do your work
        del batch
        gc.collect()