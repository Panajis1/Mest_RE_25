"""Customer-to-file index: maps customer_id → list of parquet file paths.

Migrated from pv_detection.py:build_customer_file_index().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def _load_customer_type_ids(data_dir: Path, customer_type_filter: Optional[str]) -> Optional[Set[str]]:
    """Return customer IDs matching the metadata segment, or None when filtering is disabled/unavailable."""
    if customer_type_filter is None:
        return None

    meta_path = data_dir / "metadata"
    if not meta_path.exists():
        logger.warning(
            "[CustomerIndex] Metadata not found at %s — customer-type filtering disabled",
            meta_path,
        )
        return None

    try:
        meta = pd.read_parquet(meta_path, columns=["ID", "TYPE_PARTENAIRE_LIBELLE"])
    except Exception as exc:
        logger.warning(
            "[CustomerIndex] Could not read metadata at %s (%s) — customer-type filtering disabled",
            meta_path,
            exc,
        )
        return None

    if "TYPE_PARTENAIRE_LIBELLE" not in meta.columns or "ID" not in meta.columns:
        logger.warning(
            "[CustomerIndex] Metadata missing ID/TYPE_PARTENAIRE_LIBELLE — customer-type filtering disabled"
        )
        return None

    filtered = meta[meta["TYPE_PARTENAIRE_LIBELLE"] == customer_type_filter].copy()
    filtered["ID"] = filtered["ID"].astype(str)
    return set(filtered["ID"])


def build_customer_file_index(
    data_dir: Path,
    cache_path: Optional[Path] = None,
    customer_type_filter: Optional[str] = None,
    max_consumption_kwh: Optional[float] = None,
) -> Dict[str, List[str]]:
    """Scan all parquet files in data_dir and build a {customer_id → [paths]} mapping.

    Skips the 'metadata' parquet. The index is cached as JSON if cache_path is given.
    The cache embeds the resolved data_dir path and filter settings and is automatically
    invalidated when data_dir/filter settings change or when the cached index is empty.

    Args:
        data_dir: Directory containing RE parquet files.
        cache_path: Optional path to save/load the JSON index.
        customer_type_filter: Optional metadata segment filter, e.g. "Particuliers".
        max_consumption_kwh: Optional maximum total CONSO_KWH per customer over the
            scanned period. For one-year RE files this is the annual cap (100 MWh
            = 100_000 kWh).

    Returns:
        Dict mapping customer_id (str) to list of absolute file path strings.
    """
    data_dir = Path(data_dir)
    resolved_dir = str(data_dir.resolve())
    max_consumption_kwh = float(max_consumption_kwh) if max_consumption_kwh is not None else None

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            cached = json.load(f)
        cached_dir = cached.pop("__data_dir__", None)
        cached_customer_type = cached.pop("__customer_type_filter__", None)
        cached_max_consumption = cached.pop("__max_consumption_kwh__", None)
        # Invalidate if empty or built from a different directory
        filters_match = (
            cached_customer_type == customer_type_filter
            and cached_max_consumption == max_consumption_kwh
        )
        if cached and cached_dir == resolved_dir and filters_match:
            # Backward compatibility: normalize legacy relative paths.
            normalized: Dict[str, List[str]] = {}
            for cid, paths in cached.items():
                if cid.startswith("__"):
                    continue
                normalized[str(cid)] = [str(Path(p).resolve()) for p in paths]
            return normalized
        print(
            f"[CustomerIndex] Cache invalidated "
            f"({'empty' if not cached else 'data_dir/filter settings changed'}) "
            f"— rebuilding index."
        )

    valid_ids = _load_customer_type_ids(data_dir, customer_type_filter)
    if valid_ids is not None:
        logger.info(
            "[CustomerIndex] Customer type filter %r: %d metadata IDs",
            customer_type_filter,
            len(valid_ids),
        )
    index: Dict[str, List[str]] = {}
    consumption_kwh: Dict[str, float] = {}
    parquet_files = sorted(
        p for p in data_dir.glob("*.parquet") if "metadata" not in p.name.lower()
    )
    columns = ["ID", "CONSO_KWH"] if max_consumption_kwh is not None and max_consumption_kwh > 0 else ["ID"]

    for path in parquet_files:
        try:
            df = pd.read_parquet(path, columns=columns)
            df["ID"] = df["ID"].astype(str)
            if valid_ids is not None:
                df = df[df["ID"].isin(valid_ids)]
                if df.empty:
                    continue

            for cid in df["ID"].unique():
                index.setdefault(cid, []).append(str(path.resolve()))
            if "CONSO_KWH" in df.columns:
                sums = pd.to_numeric(df["CONSO_KWH"], errors="coerce").fillna(0.0).groupby(df["ID"]).sum()
                for cid, conso in sums.items():
                    consumption_kwh[cid] = consumption_kwh.get(cid, 0.0) + float(conso)
        except Exception as exc:
            print(f"[CustomerIndex] Skipping {path.name}: {exc}")

    if max_consumption_kwh is not None and max_consumption_kwh > 0:
        before_cap = len(index)
        keep_ids = {cid for cid, total in consumption_kwh.items() if total <= max_consumption_kwh}
        index = {cid: paths for cid, paths in index.items() if cid in keep_ids}
        logger.info(
            "[CustomerIndex] Consumption cap %.0f kWh: kept %d/%d customers",
            max_consumption_kwh,
            len(index),
            before_cap,
        )

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                **index,
                "__data_dir__": resolved_dir,
                "__customer_type_filter__": customer_type_filter,
                "__max_consumption_kwh__": max_consumption_kwh,
            }, f)

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
