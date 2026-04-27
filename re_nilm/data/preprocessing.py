"""Clean and resample smart meter time series."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def clean_and_resample(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Resample to fixed frequency, interpolate short gaps, clip negatives.

    Args:
        df: DataFrame with columns [ID, DT_UTC, CONSO_KWH, PROD_KWH].
        freq: Target frequency string (default '15min').

    Returns:
        Cleaned DataFrame with the same columns.
    """
    if df.empty:
        return df

    processed_dfs = []
    for cust_id, group in df.groupby("ID"):
        group = group.sort_values("DT_UTC").set_index("DT_UTC")
        group = group[~group.index.duplicated(keep="first")]

        full_idx = pd.date_range(start=group.index.min(), end=group.index.max(), freq=freq)
        resampled = group.reindex(full_idx)
        resampled["ID"] = cust_id

        # Interpolate gaps up to 2 hours (8 × 15min), fill remaining with 0
        resampled["CONSO_KWH"] = resampled["CONSO_KWH"].interpolate(method="linear", limit=8)
        resampled["PROD_KWH"] = resampled["PROD_KWH"].interpolate(method="linear", limit=8)
        resampled = resampled.fillna(0)

        resampled["CONSO_KWH"] = resampled["CONSO_KWH"].clip(lower=0)
        resampled["PROD_KWH"] = resampled["PROD_KWH"].clip(lower=0)

        processed_dfs.append(
            resampled.reset_index().rename(columns={"index": "DT_UTC"})
        )

    return pd.concat(processed_dfs) if processed_dfs else pd.DataFrame()


def normalize_zscore(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply Z-score normalization per customer."""
    if cols is None:
        cols = ["CONSO_KWH", "PROD_KWH"]
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_norm"] = (df[col] - mean) / std if std > 0 else 0.0
    return df
