"""Regression test for the streaming compactor used by disaggregation steps.

The previous implementation read every part file into pandas at once and
concatenated them, which caused 30+ GB peaks on large 15-min disagg runs.
The new implementation streams part-by-part via pyarrow.parquet.ParquetWriter.

This test verifies the new path produces a parquet output with the exact
same rows (and column dtypes) as a pandas concat of the same parts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from re_nilm.pipeline.orchestrator import PipelineOrchestrator


def _minimal_orchestrator(tmp_path: Path) -> PipelineOrchestrator:
    """Build an orchestrator just well-formed enough to call the compactor."""
    cfg = {
        "data": {"re_data_dir": str(tmp_path), "metadata_dir": str(tmp_path)},
        "output": {"results_dir": str(tmp_path)},
        "pipeline": {"n_workers": 1, "batch_size": 10, "resume_from_checkpoint": False,
                     "checkpoint_path": str(tmp_path / "ckpt.parquet")},
        "models": {},
        "detectors": {"enabled": []},
    }
    return PipelineOrchestrator(cfg, enabled_detectors=[])


def _make_disagg_part(parts_dir: Path, part_idx: int, n_customers: int, n_rows_per_customer: int) -> pd.DataFrame:
    """Write one realistic disagg part file. Returns the in-memory equivalent."""
    rng = np.random.default_rng(part_idx)
    rows = []
    for k in range(n_customers):
        cid = f"cust_{part_idx:03d}_{k:03d}"
        ts = pd.date_range("2022-01-01", periods=n_rows_per_customer, freq="15min")
        rows.append(pd.DataFrame({
            "dt_utc": ts,
            "customer_id": cid,
            "ac_kw_pred": rng.uniform(0, 2, n_rows_per_customer).astype("float32"),
            "ac_on_prob": rng.uniform(0, 1, n_rows_per_customer).astype("float32"),
        }))
    df = pd.concat(rows, ignore_index=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parts_dir / f"part-{part_idx:06d}.parquet", index=False)
    return df


def test_compact_timeseries_parts_streams_correctly(tmp_path):
    """The streaming compactor must produce the same rows as a pandas concat."""
    orch = _minimal_orchestrator(tmp_path)
    out_path = tmp_path / "ac_disagg_15min.parquet"
    parts_dir = orch._timeseries_parts_dir(out_path)

    # 4 part files × 3 customers × 96 rows (one day at 15min). Small but
    # enough to exercise the schema-once-then-stream code path.
    expected_frames = [
        _make_disagg_part(parts_dir, i, n_customers=3, n_rows_per_customer=96)
        for i in range(4)
    ]
    expected = pd.concat(expected_frames, ignore_index=True)

    n_rows = orch._compact_timeseries_parts(out_path)
    assert n_rows == len(expected)

    actual = pd.read_parquet(out_path)
    # Streaming compaction concatenates parts in sorted order, same as pandas.
    pd.testing.assert_frame_equal(
        actual.sort_values(["customer_id", "dt_utc"]).reset_index(drop=True),
        expected.sort_values(["customer_id", "dt_utc"]).reset_index(drop=True),
        check_dtype=True,
    )


def test_compact_timeseries_parts_no_parts_returns_zero(tmp_path):
    orch = _minimal_orchestrator(tmp_path)
    out_path = tmp_path / "missing.parquet"
    assert orch._compact_timeseries_parts(out_path) == 0
    assert not out_path.exists()


def test_compact_timeseries_parts_cleans_up_parts_dir(tmp_path):
    """After successful compaction, the parts/ dir should be removed unless
    keep_part_files is set in config."""
    orch = _minimal_orchestrator(tmp_path)
    out_path = tmp_path / "ac_disagg_15min.parquet"
    parts_dir = orch._timeseries_parts_dir(out_path)
    _make_disagg_part(parts_dir, 0, n_customers=2, n_rows_per_customer=24)

    orch._compact_timeseries_parts(out_path)
    assert out_path.exists()
    assert not parts_dir.exists()
