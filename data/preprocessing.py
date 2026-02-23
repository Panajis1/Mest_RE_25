"""Clean and resample smart meter time series."""

from typing import List, Optional

import numpy as np
import pandas as pd


def clean_and_resample(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Resample to fixed frequency and fill gaps.
    - Small gaps (< 2 hours): linear interpolation.
    - Large gaps: fill with 0.
    - Clip negative values (sensor errors).
    """
    if df.empty:
        return df

    processed_dfs = []

    for cust_id, group in df.groupby("ID"):
        group = group.sort_values("DT_UTC").set_index("DT_UTC")
        group = group[~group.index.duplicated(keep="first")]

        full_idx = pd.date_range(
            start=group.index.min(), end=group.index.max(), freq=freq
        )
        resampled = group.reindex(full_idx)
        resampled["ID"] = cust_id

        resampled["CONSO_KWH"] = resampled["CONSO_KWH"].interpolate(
            method="linear", limit=8
        )
        resampled["PROD_KWH"] = resampled["PROD_KWH"].interpolate(
            method="linear", limit=8
        )
        resampled = resampled.fillna(0)

        resampled["CONSO_KWH"] = resampled["CONSO_KWH"].clip(lower=0)
        resampled["PROD_KWH"] = resampled["PROD_KWH"].clip(lower=0)

        processed_dfs.append(
            resampled.reset_index().rename(columns={"index": "DT_UTC"})
        )

    return pd.concat(processed_dfs) if processed_dfs else pd.DataFrame()


def normalize_zscore(
    df: pd.DataFrame, cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Apply Z-score normalization per customer."""
    if cols is None:
        cols = ["CONSO_KWH", "PROD_KWH"]
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f"{col}_norm"] = (df[col] - mean) / std
        else:
            df[f"{col}_norm"] = 0.0
    return df
