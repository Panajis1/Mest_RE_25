from __future__ import annotations

from pathlib import Path

import pandas as pd

from re_nilm.pipeline.customer_index import build_customer_file_index


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def test_build_customer_file_index_filters_segment_and_consumption(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    meta = pd.DataFrame({
        "ID": ["res_ok", "biz", "res_high"],
        "TYPE_PARTENAIRE_LIBELLE": ["Particuliers", "Entreprises", "Particuliers"],
    })
    _write_parquet(meta, data_dir / "metadata")

    rows = []
    for cid, conso in [("res_ok", 10.0), ("biz", 10.0), ("res_high", 60.0)]:
        rows.append(pd.DataFrame({
            "ID": cid,
            "DT_UTC": pd.date_range("2024-01-01", periods=2, freq="15min"),
            "CONSO_KWH": conso,
            "PROD_KWH": 0.0,
        }))
    _write_parquet(pd.concat(rows, ignore_index=True), data_dir / "batch_a.parquet")
    _write_parquet(pd.concat(rows, ignore_index=True), data_dir / "batch_b.parquet")

    index = build_customer_file_index(
        data_dir,
        customer_type_filter="Particuliers",
        max_consumption_kwh=100.0,
    )

    assert set(index) == {"res_ok"}


def test_customer_file_index_cache_includes_filter_settings(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_path = tmp_path / "customer_file_index.json"

    _write_parquet(pd.DataFrame({
        "ID": ["A", "B"],
        "TYPE_PARTENAIRE_LIBELLE": ["Particuliers", "Entreprises"],
    }), data_dir / "metadata")
    _write_parquet(pd.DataFrame({
        "ID": ["A", "A", "B", "B"],
        "DT_UTC": pd.date_range("2024-01-01", periods=2, freq="15min").tolist() * 2,
        "CONSO_KWH": [1.0, 1.0, 1.0, 1.0],
        "PROD_KWH": [0.0, 0.0, 0.0, 0.0],
    }), data_dir / "batch.parquet")

    residential = build_customer_file_index(
        data_dir,
        cache_path=cache_path,
        customer_type_filter="Particuliers",
        max_consumption_kwh=100_000,
    )
    all_customers = build_customer_file_index(
        data_dir,
        cache_path=cache_path,
        customer_type_filter=None,
        max_consumption_kwh=100_000,
    )

    assert set(residential) == {"A"}
    assert set(all_customers) == {"A", "B"}
