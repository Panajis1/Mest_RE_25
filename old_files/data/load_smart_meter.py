"""Load Romande Energie smart meter parquet data."""
import glob
import os
from typing import Generator, List

import pandas as pd


def get_parquet_files(data_dir: str) -> List[str]:
    """Return list of parquet files in directory."""
    return glob.glob(os.path.join(data_dir, "*.parquet"))


def load_customer_data(file_path: str) -> pd.DataFrame:
    """
    Load parquet file and ensure correct types.
    Returns DataFrame with columns: [ID, DT_UTC, CONSO_KWH, PROD_KWH]
    """
    try:
        df = pd.read_parquet(file_path)
        if "ID" in df.columns:
            df["ID"] = df["ID"].astype(str)
        if "DT_UTC" in df.columns:
            df["DT_UTC"] = pd.to_datetime(df["DT_UTC"])
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_all_data_generator(data_dir: str) -> Generator[pd.DataFrame, None, None]:
    """Yields DataFrames for each file to avoid OOM."""
    files = get_parquet_files(data_dir)
    for f in files:
        yield load_customer_data(f)
