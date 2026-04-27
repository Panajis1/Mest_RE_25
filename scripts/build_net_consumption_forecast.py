"""
Build net consumption forecast time series per customer:

    net_consumption_kwh_15min = CONSO_KWH - PROD_KWH + pv_forecast_kwh_15min

Reads meter data from ``data/re_data/ETHZ_ALL`` and PV forecast from
``data/out/pv_forecast_2024_15min.parquet``, writes sharded parquets under
``data/out/net_consumption_forecast_2024_15min/``.

Environment (optional):
    NET_MAX_PARTS — if set to a positive integer, only write that many part
    files (for debugging). Default: process all batches.
    NET_SCAN_FORECAST_IDS — if set to ``1``, discover customers by scanning the
    entire ``customer_id`` column of the forecast parquet (slow on large files).
    Default: load IDs from ``data/out/capacity_autosave.parquet`` with the same
    capacity threshold as ``forecast_pv_for_customers_streaming`` (>= 0.1 kWp),
    which matches how the forecast was built.
    NET_CUSTOMER_IDS_PATH — optional path to a CSV (column ``customer_id``) or
    a text file with one customer id per line (overrides autosave when set).

Run from repo root:
    cd Mest_RE_25 && .venv/bin/python scripts/build_net_consumption_forecast.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from model.pv_detection import build_customer_file_index  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore
        return x


# Tunable: target ~100–400 customers per output file
BATCH_SIZE = 200
YEAR = 2024

DATA_DIR = _repo_root / "data" / "re_data" / "ETHZ_ALL"
FORECAST_PATH = _repo_root / "data" / "out" / "pv_forecast_2024_15min.parquet"
OUT_DIR = _repo_root / "data" / "out" / "net_consumption_forecast_2024_15min"
CAPACITY_AUTOSAVE_PATH = _repo_root / "data" / "out" / "capacity_autosave.parquet"

# Same default as ``forecast_pv_for_customers_streaming(..., min_capacity_kwp=0.1)``
MIN_CAPACITY_KWP = 0.1


def _collect_unique_customer_ids(fc_dset: ds.Dataset) -> list[str]:
    """
    Distinct ``customer_id`` values by scanning the forecast column (slow).

    Per record batch we call ``pa.compute.unique`` so we only iterate ~one id per
    customer per batch instead of every 15-minute row.
    """
    seen: set[str] = set()
    for batch in fc_dset.to_batches(columns=["customer_id"], batch_size=262_144):
        col = batch.column(0)
        uniq = pa.compute.unique(col)
        for cid in uniq.to_pylist():
            if cid is not None:
                seen.add(str(cid))
    return sorted(seen)


def _load_customer_ids_from_capacity_autosave(
    path: Path,
    min_capacity_kwp: float = MIN_CAPACITY_KWP,
) -> list[str]:
    """Fast path: same customer cohort as the streaming PV forecast writer."""
    t = pq.read_table(path, columns=["customer_id", "pv_capacity_kwp"])
    df = t.to_pandas()
    df["pv_capacity_kwp"] = pd.to_numeric(df["pv_capacity_kwp"], errors="coerce")
    df = df.dropna(subset=["customer_id", "pv_capacity_kwp"])
    df = df.loc[df["pv_capacity_kwp"] >= float(min_capacity_kwp)]
    return sorted(df["customer_id"].astype(str).unique().tolist())


def _load_customer_ids_from_path(path: Path) -> list[str]:
    if path.suffix.lower() in (".csv",):
        df = pd.read_csv(path)
        if "customer_id" not in df.columns:
            raise ValueError(f"CSV must contain column 'customer_id': {path}")
        return sorted(df["customer_id"].astype(str).unique().tolist())
    lines = path.read_text(encoding="utf-8").splitlines()
    return sorted({ln.strip() for ln in lines if ln.strip()})


def _resolve_customer_id_list(fc_dset: ds.Dataset) -> list[str]:
    custom_path = os.environ.get("NET_CUSTOMER_IDS_PATH", "").strip()
    if custom_path:
        p = Path(custom_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"NET_CUSTOMER_IDS_PATH not found: {p}")
        print(f"Loading customer_id list from NET_CUSTOMER_IDS_PATH={p}")
        return _load_customer_ids_from_path(p)

    if os.environ.get("NET_SCAN_FORECAST_IDS", "").strip() in ("1", "true", "yes"):
        print("NET_SCAN_FORECAST_IDS=1: scanning full forecast for distinct customer_id (slow)...")
        return _collect_unique_customer_ids(fc_dset)

    cap_path = Path(CAPACITY_AUTOSAVE_PATH)
    if cap_path.is_file():
        print(f"Loading customer_id list from capacity autosave (fast): {cap_path}")
        return _load_customer_ids_from_capacity_autosave(cap_path)

    print(
        f"No {cap_path.name}; falling back to full forecast scan for distinct customer_id (slow). "
        f"Add capacity autosave at {cap_path}, or set NET_CUSTOMER_IDS_PATH, to avoid this scan."
    )
    return _collect_unique_customer_ids(fc_dset)


def main() -> None:
    forecast_path = Path(FORECAST_PATH)
    if not forecast_path.is_file():
        raise FileNotFoundError(f"Missing forecast parquet: {forecast_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_parts_env = os.environ.get("NET_MAX_PARTS", "").strip()
    max_parts: int | None = None
    if max_parts_env:
        max_parts = int(max_parts_env)
        if max_parts <= 0:
            max_parts = None

    print("Opening forecast dataset...")
    fc_dset = ds.dataset(str(forecast_path), format="parquet")

    all_ids = _resolve_customer_id_list(fc_dset)
    print(f"  {len(all_ids):,} customers (batch list)")

    print("Building customer → file index (ETHZ_ALL)...")
    cust_file_index = build_customer_file_index(str(DATA_DIR), restrict_to_ids=set(all_ids))
    print(f"  {len(cust_file_index):,} customers mapped to raw parquet files")

    manifest_rows: list[dict] = []
    n_batches = (len(all_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_ranges = list(range(0, len(all_ids), BATCH_SIZE))
    if max_parts is not None:
        batch_ranges = batch_ranges[:max_parts]
        print(f"NET_MAX_PARTS={max_parts}: processing {len(batch_ranges)} batch(es) only")

    for k, start in enumerate(tqdm(batch_ranges, desc="Batches")):
        batch_ids = all_ids[start : start + BATCH_SIZE]

        fc_tbl = fc_dset.to_table(
            columns=["customer_id", "DT_UTC", "pv_forecast_kwh_15min"],
            filter=ds.field("customer_id").isin(batch_ids),
        )
        fc = fc_tbl.to_pandas()
        if fc.empty:
            print(f"  [part {k:04d}] no forecast rows for batch; skipping")
            continue

        fc["customer_id"] = fc["customer_id"].astype(str)
        fc["DT_UTC"] = pd.to_datetime(fc["DT_UTC"], utc=True).dt.tz_convert(None)

        files_needed = sorted({f for cid in batch_ids for f in cust_file_index.get(cid, [])})
        id_set = set(batch_ids)
        re_chunks: list[pd.DataFrame] = []

        for fpath in files_needed:
            df = pd.read_parquet(fpath, columns=["ID", "DT_UTC", "CONSO_KWH", "PROD_KWH"])
            df["ID"] = df["ID"].astype(str)
            df = df.loc[df["ID"].isin(id_set)]
            if df.empty:
                continue
            dt = pd.to_datetime(df["DT_UTC"], utc=True).dt.tz_convert(None)
            df["DT_UTC"] = dt
            df = df.loc[df["DT_UTC"].dt.year == YEAR]
            if df.empty:
                continue
            re_chunks.append(df)

        if re_chunks:
            re_df = pd.concat(re_chunks, ignore_index=True)
            re_df = re_df.rename(columns={"ID": "customer_id"})
            re_df = re_df.drop_duplicates(subset=["customer_id", "DT_UTC"], keep="last")
        else:
            re_df = pd.DataFrame(columns=["customer_id", "DT_UTC", "CONSO_KWH", "PROD_KWH"])

        merged = fc.merge(re_df, on=["customer_id", "DT_UTC"], how="left")
        for col in ("CONSO_KWH", "PROD_KWH"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float32").fillna(0.0)
        merged["pv_forecast_kwh_15min"] = pd.to_numeric(
            merged["pv_forecast_kwh_15min"], errors="coerce"
        ).astype("float32")
        merged["net_consumption_kwh_15min"] = (
            merged["CONSO_KWH"] - merged["PROD_KWH"] + merged["pv_forecast_kwh_15min"]
        ).astype("float32")

        out_cols = [
            "customer_id",
            "DT_UTC",
            "CONSO_KWH",
            "PROD_KWH",
            "pv_forecast_kwh_15min",
            "net_consumption_kwh_15min",
        ]
        out_path = OUT_DIR / f"net_consumption_2024_15min_part_{k:04d}.parquet"
        table = pa.Table.from_pandas(merged[out_cols], preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy")

        cust_in_batch = merged["customer_id"].nunique()
        manifest_rows.append(
            {
                "part_file": out_path.name,
                "n_rows": len(merged),
                "n_customers": int(cust_in_batch),
                "customer_id_first": batch_ids[0],
                "customer_id_last": batch_ids[-1],
            }
        )

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(OUT_DIR / "manifest.csv", index=False)
        print(f"Wrote {len(manifest_rows)} part file(s) under {OUT_DIR}")
        print(f"Manifest: {OUT_DIR / 'manifest.csv'}")
    else:
        print("No output files written (empty forecast batches?).")


if __name__ == "__main__":
    main()
