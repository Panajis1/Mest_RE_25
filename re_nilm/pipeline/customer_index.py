"""Customer-to-file index: maps customer_id → list of parquet file paths.

Migrated from pv_detection.py:build_customer_file_index().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def build_customer_file_index(
    data_dir: Path,
    cache_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Scan all parquet files in data_dir and build a {customer_id → [paths]} mapping.

    Skips the 'metadata' parquet. The index is cached as JSON if cache_path is given.

    Args:
        data_dir: Directory containing RE parquet files.
        cache_path: Optional path to save/load the JSON index.

    Returns:
        Dict mapping customer_id (str) to list of absolute file path strings.
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    index: Dict[str, List[str]] = {}
    parquet_files = sorted(
        p for p in Path(data_dir).glob("*.parquet") if "metadata" not in p.name.lower()
    )

    for path in parquet_files:
        try:
            df = pd.read_parquet(path, columns=["ID"])
            df["ID"] = df["ID"].astype(str)
            for cid in df["ID"].unique():
                index.setdefault(cid, []).append(str(path))
        except Exception as exc:
            print(f"[CustomerIndex] Skipping {path.name}: {exc}")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(index, f)

    return index


def load_customer_from_index(
    customer_id: str,
    index: Dict[str, List[str]],
) -> pd.DataFrame:
    """Load all rows for a single customer using a pre-built file index."""
    paths = index.get(str(customer_id), [])
    if not paths:
        return pd.DataFrame()

    frames = []
    for path in paths:
        try:
            df = pd.read_parquet(path)
            df["ID"] = df["ID"].astype(str)
            sub = df[df["ID"] == customer_id]
            if not sub.empty:
                frames.append(sub)
        except Exception as exc:
            print(f"[CustomerIndex] Error reading {path} for {customer_id}: {exc}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values("DT_UTC").reset_index(drop=True)
