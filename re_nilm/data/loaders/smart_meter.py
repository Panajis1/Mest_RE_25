"""Smart meter loader — merges load_smart_meter.py and fast_load_smart_meter.py."""

from __future__ import annotations

import gc
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Generator, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ["ID", "DT_UTC", "CONSO_KWH", "PROD_KWH"]


def _load_single_file(
    file_path: str,
    valid_ids: Optional[Set[str]],
    max_annual_kwh: float,
) -> pd.DataFrame:
    """Worker: load one parquet, filter by valid_ids, drop high-consumption outliers."""
    try:
        df = pd.read_parquet(file_path, columns=_REQUIRED_COLS, engine="pyarrow")
    except Exception:
        # Missing columns → try without column selection
        try:
            df = pd.read_parquet(file_path, engine="pyarrow")
            missing = [c for c in _REQUIRED_COLS if c not in df.columns]
            if missing:
                logger.warning("File %s missing columns %s — skipping", file_path, missing)
                return pd.DataFrame()
            df = df[_REQUIRED_COLS].copy()
        except Exception as exc:
            logger.error("Failed to read %s: %s", file_path, exc)
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["ID"] = df["ID"].astype(str)

    if valid_ids is not None:
        df = df[df["ID"].isin(valid_ids)]
        if df.empty:
            return pd.DataFrame()

    _dt = pd.to_datetime(df["DT_UTC"], errors="coerce")
    if _dt.dt.tz is not None:
        _dt = _dt.dt.tz_convert("UTC").dt.tz_localize(None)
    df["DT_UTC"] = _dt

    # Drop customers whose annual consumption exceeds the threshold
    if max_annual_kwh > 0:
        yearly = df.groupby("ID")["CONSO_KWH"].sum()
        keep = yearly[yearly <= max_annual_kwh].index
        df = df[df["ID"].isin(keep)]

    return df


_NO_FILTER_SENTINEL: Optional[Set[str]] = None  # used to mean "load all customers"


def _load_metadata(data_dir: Path, customer_type: str) -> tuple[Optional[Set[str]], pd.DataFrame]:
    """Load metadata parquet and return (valid_ids_set, metadata_df).

    Returns (None, empty_df) when metadata file is missing, which is interpreted
    downstream as "no filter — load all customers".
    """
    meta_path = data_dir / "metadata"
    if not meta_path.exists():
        logger.warning("Metadata not found at %s — no customer-type filtering applied", meta_path)
        return _NO_FILTER_SENTINEL, pd.DataFrame()

    try:
        meta = pd.read_parquet(meta_path, engine="pyarrow")
    except Exception as exc:
        logger.error("Failed to read metadata at %s: %s — no customer-type filtering applied", meta_path, exc)
        return _NO_FILTER_SENTINEL, pd.DataFrame()

    meta["ID"] = meta["ID"].astype(str)
    filtered = meta[meta["TYPE_PARTENAIRE_LIBELLE"] == customer_type]
    return set(filtered["ID"]), filtered


class SmartMeterLoader:
    """Load Romande Energie smart meter parquet files, optionally filtered by customer type.

    Args:
        data_dir: Directory containing parquet files and a 'metadata' parquet.
        customer_type: Filter to customers of this type (e.g. 'Particuliers'). Pass None
            to load all customers.
        max_annual_kwh: Drop customers whose total CONSO_KWH exceeds this value.
        n_workers: Number of parallel worker processes for batch loading. Use 1 for
            sequential (safer for debugging and low-RAM machines).
        batch_size: Files per parallel batch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        customer_type: Optional[str] = "Particuliers",
        max_annual_kwh: float = 100_000.0,
        n_workers: int = 1,
        batch_size: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.customer_type = customer_type
        self.max_annual_kwh = max_annual_kwh
        self.n_workers = n_workers
        self.batch_size = batch_size
        self._valid_ids: Optional[Set[str]] = None
        self._metadata: pd.DataFrame = pd.DataFrame()

    _METADATA_LOADED = "_metadata_loaded"

    def _ensure_metadata(self) -> None:
        if getattr(self, "_metadata_loaded_flag", False):
            return
        self._metadata_loaded_flag = True
        if self.customer_type is not None:
            self._valid_ids, self._metadata = _load_metadata(self.data_dir, self.customer_type)
            # _load_metadata returns None when metadata is missing → no filter
        else:
            self._valid_ids = None  # explicit no filter

    def parquet_files(self) -> list[Path]:
        """Return sorted list of data parquet files (excludes 'metadata')."""
        return sorted(
            p for p in self.data_dir.glob("*.parquet") if "metadata" not in p.name.lower()
        )

    def load_file(self, path: Path) -> pd.DataFrame:
        """Load a single parquet file with filtering applied."""
        self._ensure_metadata()
        return _load_single_file(str(path), self._valid_ids, self.max_annual_kwh)

    def stream_files(self) -> Generator[pd.DataFrame, None, None]:
        """Yield one DataFrame per parquet file (sequential, memory-efficient)."""
        self._ensure_metadata()
        for path in self.parquet_files():
            df = _load_single_file(str(path), self._valid_ids, self.max_annual_kwh)
            if not df.empty:
                yield df

    def stream_batches(self, batch_size: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """Yield concatenated DataFrames for parallel-loaded batches of files.

        Uses multiprocessing when n_workers > 1 for faster I/O on large datasets.
        """
        self._ensure_metadata()
        bs = batch_size or self.batch_size
        files = [str(p) for p in self.parquet_files()]
        worker = partial(
            _load_single_file,
            valid_ids=self._valid_ids,
            max_annual_kwh=self.max_annual_kwh,
        )
        n = max(1, min(self.n_workers, cpu_count() // 2 or 1))

        for i in range(0, len(files), bs):
            chunk = files[i : i + bs]
            if n > 1:
                with Pool(processes=n) as pool:
                    results = pool.map(worker, chunk)
            else:
                results = [worker(f) for f in chunk]

            non_empty = [r for r in results if not r.empty]
            if not non_empty:
                del results
                gc.collect()
                continue
            batch = pd.concat(non_empty, ignore_index=True)
            if not batch.empty:
                yield batch

            del results
            gc.collect()

    def load_customer(self, customer_id: str) -> pd.DataFrame:
        """Load all rows for a single customer, scanning all parquet files."""
        frames = []
        for df in self.stream_files():
            sub = df[df["ID"] == customer_id]
            if not sub.empty:
                frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=_REQUIRED_COLS)
        return pd.concat(frames, ignore_index=True).sort_values("DT_UTC").reset_index(drop=True)
