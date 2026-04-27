"""Unified streaming engine for processing thousands of customers from parquet files.

Replaces the duplicated streaming/autosave/resume logic in:
  - model/pv_detection.py (process_customers_streaming, stream_pv_indicators)
  - hp_model/hp_detection_RE.py
  - model/ac_batch_processing.py
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_memory_mb() -> float:
    if not _PSUTIL_AVAILABLE:
        return 0.0
    import os
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _load_checkpoint(checkpoint_path: Path) -> set:
    """Return set of customer_ids already processed."""
    if checkpoint_path.exists():
        try:
            done = pd.read_parquet(checkpoint_path, columns=["customer_id"])
            return set(done["customer_id"].astype(str))
        except Exception:
            pass
    return set()


def _write_checkpoint(results: List[dict], checkpoint_path: Path) -> None:
    """Append result rows to checkpoint parquet."""
    df = pd.DataFrame(results)[["customer_id"]]
    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path, columns=["customer_id"])
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(checkpoint_path, index=False)


def _append_results(results: List[dict], output_path: Path) -> None:
    """Append a batch of result dicts to the output parquet."""
    if not results:
        return
    df = pd.DataFrame(results)
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


class StreamingEngine:
    """Streaming pipeline engine for per-customer appliance detection.

    Iterates customer IDs in configurable batches, calls processor_fn per
    customer, autosaves results, and supports checkpoint-based resume.

    Args:
        n_workers: Parallel workers (1 = sequential, >1 = ProcessPoolExecutor).
        batch_size: Customers per batch before autosave.
        resume: If True, skip already-processed customers using checkpoint.
        checkpoint_path: Path for checkpoint parquet.
        memory_limit_mb: If RAM usage exceeds this, reduce batch size (0 = no limit).
    """

    def __init__(
        self,
        n_workers: int = 1,
        batch_size: int = 500,
        resume: bool = True,
        checkpoint_path: Optional[Path] = None,
        memory_limit_mb: float = 0.0,
    ):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.memory_limit_mb = memory_limit_mb

    def run(
        self,
        customer_ids: Iterable[str],
        processor_fn: Callable[[str], Optional[dict]],
        output_path: Path,
    ) -> pd.DataFrame:
        """Process all customer IDs and collect results.

        Args:
            customer_ids: Iterable of customer ID strings to process.
            processor_fn: Callable(customer_id) → dict | None. Must be picklable
                for multiprocessing (i.e. a top-level function or a class with __call__).
            output_path: Where to write/append results parquet.

        Returns:
            Final results DataFrame.
        """
        all_ids = list(customer_ids)
        done_ids: set = set()

        if self.resume and self.checkpoint_path is not None:
            done_ids = _load_checkpoint(self.checkpoint_path)
            n_skip = len(done_ids)
            if n_skip:
                logger.info("[StreamingEngine] Resuming: skipping %d already-processed customers", n_skip)

        pending = [cid for cid in all_ids if cid not in done_ids]
        n_total = len(pending)
        logger.info("[StreamingEngine] Processing %d customers (workers=%d, batch=%d)",
                    n_total, self.n_workers, self.batch_size)

        iter_ids = tqdm(range(0, n_total, self.batch_size), desc="Batches") if _TQDM_AVAILABLE else range(0, n_total, self.batch_size)

        batch_results: List[dict] = []

        for batch_start in iter_ids:
            # Memory ceiling check
            if self.memory_limit_mb > 0 and _PSUTIL_AVAILABLE:
                mem_mb = _check_memory_mb()
                if mem_mb > self.memory_limit_mb:
                    logger.warning("[StreamingEngine] RAM %.0f MB exceeds limit %.0f MB — forcing GC", mem_mb, self.memory_limit_mb)
                    gc.collect()

            batch_ids = pending[batch_start: batch_start + self.batch_size]
            results_this_batch: List[dict] = []

            if self.n_workers <= 1:
                for cid in batch_ids:
                    try:
                        result = processor_fn(cid)
                        if result is not None:
                            results_this_batch.append(result)
                    except Exception as exc:
                        logger.error("[StreamingEngine] Error processing %s: %s", cid, exc)
            else:
                with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                    futures = {pool.submit(processor_fn, cid): cid for cid in batch_ids}
                    for future in as_completed(futures):
                        cid = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                results_this_batch.append(result)
                        except Exception as exc:
                            logger.error("[StreamingEngine] Error processing %s: %s", cid, exc)

            if results_this_batch:
                _append_results(results_this_batch, output_path)
                if self.checkpoint_path is not None:
                    _write_checkpoint(results_this_batch, self.checkpoint_path)
                batch_results.extend(results_this_batch)

            del results_this_batch
            gc.collect()

        if output_path.exists():
            return pd.read_parquet(output_path)
        return pd.DataFrame(batch_results)
