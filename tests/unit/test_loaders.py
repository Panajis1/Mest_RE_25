"""Unit tests for re_nilm/data/loaders/smart_meter.py edge cases."""

import gc
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from re_nilm.data.loaders.smart_meter import SmartMeterLoader, _load_single_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def _make_meter_df(customer_ids=("A", "B"), n_rows=48) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    rows = []
    for cid in customer_ids:
        dt = pd.date_range("2022-01-01", periods=n_rows, freq="15min")
        rows.append(pd.DataFrame({
            "ID": cid,
            "DT_UTC": dt,
            "CONSO_KWH": np.random.default_rng(0).uniform(0.1, 1.0, n_rows),
            "PROD_KWH": 0.0,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# _load_single_file
# ---------------------------------------------------------------------------

def test_load_single_file_basic():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        _write_parquet(_make_meter_df(), path)
        result = _load_single_file(str(path), valid_ids=None, max_annual_kwh=0)
        assert set(result["ID"].unique()) == {"A", "B"}


def test_load_single_file_valid_ids_filter():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        _write_parquet(_make_meter_df(), path)
        result = _load_single_file(str(path), valid_ids={"A"}, max_annual_kwh=0)
        assert list(result["ID"].unique()) == ["A"]


def test_load_single_file_bad_path_returns_empty():
    result = _load_single_file("/nonexistent/path.parquet", valid_ids=None, max_annual_kwh=0)
    assert result.empty


# ---------------------------------------------------------------------------
# SmartMeterLoader — missing metadata → no filter
# ---------------------------------------------------------------------------

def test_loader_no_metadata_loads_all():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_parquet(_make_meter_df(["X", "Y"]), tmp_path / "batch1.parquet")
        # No metadata file

        loader = SmartMeterLoader(
            data_dir=tmp_path,
            customer_type="Particuliers",
            max_annual_kwh=0,
            n_workers=1,
        )
        frames = list(loader.stream_files())
        assert len(frames) > 0
        ids = set(pd.concat(frames)["ID"].unique())
        # Without metadata, all customers should be loaded (no filter applied)
        assert "X" in ids and "Y" in ids


def test_loader_with_metadata_filters():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_parquet(_make_meter_df(["X", "Y"]), tmp_path / "batch1.parquet")

        # Write minimal metadata
        meta = pd.DataFrame({
            "ID": ["X", "Y"],
            "TYPE_PARTENAIRE_LIBELLE": ["Particuliers", "Entreprises"],
        })
        _write_parquet(meta, tmp_path / "metadata")

        loader = SmartMeterLoader(
            data_dir=tmp_path,
            customer_type="Particuliers",
            max_annual_kwh=0,
        )
        frames = list(loader.stream_files())
        assert len(frames) > 0
        ids = set(pd.concat(frames)["ID"].unique())
        assert "X" in ids
        assert "Y" not in ids


# ---------------------------------------------------------------------------
# stream_batches — empty batch guard
# ---------------------------------------------------------------------------

def test_stream_batches_no_crash_on_empty_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # File with wrong schema (will be skipped)
        pd.DataFrame({"foo": [1, 2]}).to_parquet(tmp_path / "bad.parquet")
        # File with correct schema but no matching IDs
        _write_parquet(_make_meter_df(["Z"]), tmp_path / "good.parquet")

        loader = SmartMeterLoader(
            data_dir=tmp_path,
            customer_type=None,
            max_annual_kwh=0,
            n_workers=1,
            batch_size=2,
        )
        batches = list(loader.stream_batches())
        # No crash expected; may yield 0 or 1 batch
        assert isinstance(batches, list)
