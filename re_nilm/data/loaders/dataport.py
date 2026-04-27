"""Load Pecan Street Dataport data: 15-min circuit data, EV metadata, weather, pricing events.

Migrated from data/load_dataport.py. All interactive notebook cells removed.
The default data directory is now data/raw/dataport/ relative to the repo root.
Pass data_dir explicitly to override.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd

_DEFAULT_DATAPORT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "dataport"

DATAPORT_15MIN_FILES = [
    "15minute_data_california",
    "15minute_data_austin",
    "15minute_data_newyork",
]

APPLIANCE_COLUMNS = [
    "grid", "solar", "solar2",
    "car1", "car2",
    "air1", "air2", "air3", "airwindowunit1",
    "furnace1", "furnace2", "heater1", "heater2", "heater3",
]

COLS_EV = ["car1", "car2"]
COLS_HP = ["furnace1", "furnace2", "heater1", "heater2", "heater3"]
COLS_PV = ["solar", "solar2"]
COLS_AC = ["air1", "air2", "air3", "airwindowunit1"]


def _data_dir_path(data_dir: Optional[Union[Path, str]] = None) -> Path:
    return Path(data_dir or _DEFAULT_DATAPORT_DIR)


def get_available_regions(data_dir: Optional[Union[Path, str]] = None) -> List[str]:
    """Return list of region names for which 15minute_data_<region>.sqlite3 exists."""
    base = _data_dir_path(data_dir)
    return [
        stem.replace("15minute_data_", "")
        for stem in DATAPORT_15MIN_FILES
        if (base / f"{stem}.sqlite3").exists()
    ]


def load_15min_data(
    data_dir: Optional[Union[Path, str]] = None,
    region: Optional[str] = None,
    dataids: Optional[List[int]] = None,
    appliance_columns_only: bool = False,
) -> pd.DataFrame:
    """Load 15-minute circuit-level data from one Dataport region."""
    region = (region or "california").lower()
    base = _data_dir_path(data_dir)
    path = base / f"15minute_data_{region}.sqlite3"
    if not path.exists():
        raise FileNotFoundError(f"Dataport SQLite not found: {path}")

    query = f'SELECT * FROM "15minute_data_{region}"'
    if dataids is not None:
        query += f" WHERE dataid IN ({','.join(str(i) for i in dataids)})"
    query += " ORDER BY dataid, local_15min"

    df = pd.read_sql_query(query, f"sqlite:///{path}")
    # Strip trailing fixed-offset from local_15min (e.g. "-05") — keep as naive local time
    s = df["local_15min"].astype(str).str.replace(r"([+-]\d{2})$", "", regex=True)
    df["local_15min"] = pd.to_datetime(s, errors="coerce")

    for col in df.columns:
        if col not in ("dataid", "local_15min"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if appliance_columns_only:
        keep = ["dataid", "local_15min"] + [c for c in APPLIANCE_COLUMNS if c in df.columns]
        df = df[keep].copy()
    return df


def load_all_15min_data(
    data_dir: Optional[Union[Path, str]] = None,
    regions: Optional[List[str]] = None,
    dataids: Optional[List[int]] = None,
    appliance_columns_only: bool = False,
) -> pd.DataFrame:
    """Load and concatenate all (or selected) Dataport regions. Adds 'region' column."""
    base = _data_dir_path(data_dir)
    to_load = regions if regions is not None else get_available_regions(data_dir)
    if not to_load:
        return pd.DataFrame()

    parts = []
    for reg in to_load:
        if not (base / f"15minute_data_{reg}.sqlite3").exists():
            continue
        one = load_15min_data(base, reg, dataids, appliance_columns_only)
        one["region"] = reg
        parts.append(one)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    priority = ["region", "dataid", "local_15min"]
    rest = [c for c in df.columns if c not in priority]
    return df[[c for c in priority + rest if c in df.columns]]


def _households_with_appliance(df: pd.DataFrame, cols: List[str]) -> Set[Tuple[str, int]]:
    present: Set[Tuple[str, int]] = set()
    for (region, dataid), g in df.groupby(["region", "dataid"]):
        if any(c in g.columns and (g[c].fillna(0) != 0).any() for c in cols):
            present.add((region, dataid))
    return present


def get_appliance_households(
    df: pd.DataFrame,
    manifest: bool = False,
) -> Union[Set[Tuple[str, int]], Tuple[Set[Tuple[str, int]], pd.DataFrame]]:
    """Return (region, dataid) households that have HP, PV, EV, or AC."""
    ev = _households_with_appliance(df, COLS_EV)
    hp = _households_with_appliance(df, COLS_HP)
    pv = _households_with_appliance(df, COLS_PV)
    ac = _households_with_appliance(df, COLS_AC)
    any_appliance = ev | hp | pv | ac

    if not manifest:
        return any_appliance

    rows = [
        {"region": r, "dataid": d,
         "has_ev": (r, d) in ev, "has_pv": (r, d) in pv,
         "has_ac": (r, d) in ac, "has_hp": (r, d) in hp}
        for r, d in sorted(any_appliance)
    ]
    return any_appliance, pd.DataFrame(rows)


def load_electric_vehicles(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Load EV metadata from ev_and_weather.zip."""
    zip_path = _data_dir_path(data_dir) / "ev_and_weather.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"ev_and_weather.zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ev_and_weather/electric_vehicles.csv") as f:
            return pd.read_csv(io.BytesIO(f.read()))


def load_weather(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Load hourly weather from ev_and_weather.zip."""
    zip_path = _data_dir_path(data_dir) / "ev_and_weather.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"ev_and_weather.zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("ev_and_weather/weather.csv") as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df["localhour"] = pd.to_datetime(df["localhour"])
    return df
