"""Unified streaming engine for processing thousands of customers from parquet files.

Replaces the duplicated streaming/autosave/resume logic in:
  - model/pv_detection.py (process_customers_streaming, stream_pv_indicators)
  - hp_model/hp_detection_RE.py
  - model/ac_batch_processing.py
"""

from __future__ import annotations

import gc
import logging
import shutil
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


def _checkpoint_log_path(checkpoint_path: Path) -> Path:
    """Return append-only text checkpoint path beside the legacy parquet path."""
    return checkpoint_path.with_suffix(".txt")


def _load_checkpoint(checkpoint_path: Path, checkpoint_format: str = "parquet") -> set:
    """Return set of customer_ids already processed.

    Text checkpoint mode still reads legacy parquet checkpoints for backwards
    compatibility, then unions any append-only text log entries.
    """
    done: set = set()
    if checkpoint_path.exists():
        try:
            done = pd.read_parquet(checkpoint_path, columns=["customer_id"])
            done = set(done["customer_id"].astype(str))
        except Exception:
            pass
    if checkpoint_format == "text":
        log_path = _checkpoint_log_path(checkpoint_path)
        if log_path.exists():
            try:
                with open(log_path) as f:
                    done.update(line.strip() for line in f if line.strip())
            except Exception:
                pass
    return done


def _write_checkpoint(results: List[dict], checkpoint_path: Path, checkpoint_format: str = "parquet") -> None:
    """Append result rows to checkpoint parquet."""
    if not results:
        return
    df = pd.DataFrame(results)[["customer_id"]]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_format == "text":
        log_path = _checkpoint_log_path(checkpoint_path)
        with open(log_path, "a") as f:
            for cid in df["customer_id"].astype(str).drop_duplicates():
                f.write(f"{cid}\n")
        return
    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path, columns=["customer_id"])
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates()
    df.to_parquet(checkpoint_path, index=False)


def _append_results_legacy(results: List[dict], output_path: Path) -> None:
    """Append a batch of result dicts to the output parquet."""
    if not results:
        return
    df = pd.DataFrame(results)
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def _parts_dir(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_parts")


def _next_part_index(parts_dir: Path) -> int:
    indices = []
    for path in parts_dir.glob("part-*.parquet"):
        try:
            indices.append(int(path.stem.split("-")[-1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def _write_result_part(results: List[dict], parts_dir: Path, part_index: int) -> None:
    """Write one batch of result dicts as an immutable parquet part."""
    if not results:
        return
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"part-{part_index:06d}.parquet"
    tmp_path = parts_dir / f"part-{part_index:06d}.tmp.parquet"
    pd.DataFrame(results).to_parquet(tmp_path, index=False)
    tmp_path.replace(part_path)


def _compact_parts(output_path: Path, parts_dir: Path, keep_part_files: bool = False) -> pd.DataFrame:
    """Compact parquet parts into the public output path once at step completion."""
    part_paths = sorted(parts_dir.glob("part-*.parquet"))
    if not part_paths:
        if output_path.exists():
            return pd.read_parquet(output_path)
        return pd.DataFrame()

    frames = []
    if output_path.exists():
        frames.append(pd.read_parquet(output_path))
    frames.extend(pd.read_parquet(path) for path in part_paths)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["customer_id"], keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    if not keep_part_files:
        shutil.rmtree(parts_dir, ignore_errors=True)
    return out


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
        write_mode: 'parts' writes per-batch part files and compacts once; 'legacy'
            preserves read/concat/rewrite behaviour.
        keep_part_files: Keep part files after successful compaction.
        checkpoint_format: 'text' uses an append-only checkpoint log; 'parquet'
            preserves legacy rewrite behaviour.
    """

    def __init__(
        self,
        n_workers: int = 1,
        batch_size: int = 500,
        resume: bool = True,
        checkpoint_path: Optional[Path] = None,
        memory_limit_mb: float = 0.0,
        write_mode: str = "parts",
        keep_part_files: bool = False,
        checkpoint_format: str = "text",
    ):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.memory_limit_mb = memory_limit_mb
        self.write_mode = write_mode
        self.keep_part_files = keep_part_files
        self.checkpoint_format = checkpoint_format

    def run(
        self,
        customer_ids: Iterable[str],
        processor_fn: Callable[[str], Optional[dict]],
        output_path: Path,
        worker_initializer: Optional[Callable] = None,
        worker_initargs: tuple = (),
    ) -> pd.DataFrame:
        """Process all customer IDs and collect results.

        Args:
            customer_ids: Iterable of customer ID strings to process.
            processor_fn: Callable(customer_id) → dict | None. Must be a
                module-level function (not a closure) for multiprocessing.
            output_path: Where to write/append results parquet.
            worker_initializer: Optional callable passed to ProcessPoolExecutor
                initializer — called once per worker process on startup.
                Use this to set up per-process state (detector, index, weather)
                instead of embedding large objects in processor_fn, which would
                be pickled once per task submission.
            worker_initargs: Positional arguments forwarded to worker_initializer.

        Returns:
            Final results DataFrame.
        """
        all_ids = list(customer_ids)
        done_ids: set = set()
        output_path = Path(output_path)
        parts_dir = _parts_dir(output_path)

        if not self.resume:
            output_path.unlink(missing_ok=True)
            if self.checkpoint_path is not None:
                self.checkpoint_path.unlink(missing_ok=True)
                _checkpoint_log_path(self.checkpoint_path).unlink(missing_ok=True)
            shutil.rmtree(parts_dir, ignore_errors=True)

        if self.resume and self.checkpoint_path is not None:
            done_ids = _load_checkpoint(self.checkpoint_path, checkpoint_format=self.checkpoint_format)
            n_skip = len(done_ids)
            if n_skip:
                logger.info("[StreamingEngine] Resuming: skipping %d already-processed customers", n_skip)

        pending = [cid for cid in all_ids if cid not in done_ids]
        n_total = len(pending)
        logger.info("[StreamingEngine] Processing %d customers (workers=%d, batch=%d)",
                    n_total, self.n_workers, self.batch_size)

        iter_ids = tqdm(range(0, n_total, self.batch_size), desc="Batches") if _TQDM_AVAILABLE else range(0, n_total, self.batch_size)

        batch_results: List[dict] = []
        next_part_index = _next_part_index(parts_dir)

        def _process_batch(batch_ids: list[str], pool: Optional[ProcessPoolExecutor] = None) -> List[dict]:
            results_this_batch: List[dict] = []
            if pool is None:
                for cid in batch_ids:
                    try:
                        result = processor_fn(cid)
                        if result is not None:
                            results_this_batch.append(result)
                    except Exception as exc:
                        logger.error("[StreamingEngine] Error processing %s: %s", cid, exc)
                return results_this_batch

            futures = {pool.submit(processor_fn, cid): cid for cid in batch_ids}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_this_batch.append(result)
                except Exception as exc:
                    logger.error("[StreamingEngine] Error processing %s: %s", cid, exc)
            return results_this_batch

        def _run_loop(pool: Optional[ProcessPoolExecutor] = None) -> None:
            nonlocal next_part_index
            for batch_start in iter_ids:
                # Memory ceiling check
                if self.memory_limit_mb > 0 and _PSUTIL_AVAILABLE:
                    mem_mb = _check_memory_mb()
                    if mem_mb > self.memory_limit_mb:
                        logger.warning("[StreamingEngine] RAM %.0f MB exceeds limit %.0f MB — forcing GC", mem_mb, self.memory_limit_mb)
                        gc.collect()

                batch_ids = pending[batch_start: batch_start + self.batch_size]
                results_this_batch = _process_batch(batch_ids, pool=pool)

                if results_this_batch:
                    if self.write_mode == "parts":
                        _write_result_part(results_this_batch, parts_dir, next_part_index)
                        next_part_index += 1
                    else:
                        _append_results_legacy(results_this_batch, output_path)
                    if self.checkpoint_path is not None:
                        _write_checkpoint(
                            results_this_batch,
                            self.checkpoint_path,
                            checkpoint_format=self.checkpoint_format,
                        )
                    batch_results.extend(results_this_batch)

                del results_this_batch
                gc.collect()

        if self.n_workers <= 1:
            if worker_initializer is not None:
                worker_initializer(*worker_initargs)
            _run_loop(pool=None)
        else:
            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=worker_initializer,
                initargs=worker_initargs,
            ) as pool:
                _run_loop(pool=pool)

        if self.write_mode == "parts":
            return _compact_parts(output_path, parts_dir, keep_part_files=self.keep_part_files)
        if output_path.exists():
            return pd.read_parquet(output_path)
        return pd.DataFrame(batch_results)
