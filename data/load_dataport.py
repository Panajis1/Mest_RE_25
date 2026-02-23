"""Load Pecan Street Dataport data: 15-min circuit data, EV metadata, weather, pricing events."""

import gzip
import io
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd

# Default path to Dataport data (relative to this file)
_DEFAULT_DATA_PORT_DIR = Path(__file__).resolve().parent / "raw_data" / "data_port"

# Known 15-min SQLite files: filename stem (without .sqlite3) -> region label
DATAPORT_15MIN_FILES = [
    "15minute_data_california",
    "15minute_data_austin",
    "15minute_data_newyork",
]

# Appliance columns of interest: load curve, PV, EV, AC, heat
APPLIANCE_COLUMNS = [
    "grid",
    "solar",
    "solar2",
    "car1",
    "car2",
    "air1",
    "air2",
    "air3",
    "airwindowunit1",
    "furnace1",
    "furnace2",
    "heater1",
    "heater2",
    "heater3",
]

# Column groups for appliance-household detection (EV, HP, PV, AC)
COLS_EV = ["car1", "car2"]
COLS_HP = ["furnace1", "furnace2", "heater1", "heater2", "heater3"]
COLS_PV = ["solar", "solar2"]
COLS_AC = ["air1", "air2", "air3", "airwindowunit1"]


def _data_dir_path(data_dir: Optional[Union[Path, str]] = None) -> Path:
    return Path(data_dir or _DEFAULT_DATA_PORT_DIR)


def get_available_regions(data_dir: Optional[Union[Path, str]] = None) -> List[str]:
    """Return list of region names for which 15minute_data_<region>.sqlite3 exists."""
    base = _data_dir_path(data_dir)
    available = []
    for stem in DATAPORT_15MIN_FILES:
        # stem is e.g. "15minute_data_california" -> region "california"
        if (base / f"{stem}.sqlite3").exists():
            region = stem.replace("15minute_data_", "")
            available.append(region)
    return available


def load_15min_data(
    data_dir: Optional[Union[Path, str]] = None,
    region: Optional[str] = None,
    dataids: Optional[List[int]] = None,
    appliance_columns_only: bool = False,
) -> pd.DataFrame:
    """
    Load 15-minute circuit-level data from one Dataport region (California, Austin, or New York).

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing 15minute_data_<region>.sqlite3. Defaults to data/raw_data/data_port.
    region : str, optional
        One of "california", "austin", "newyork". If None, uses "california" for backward compatibility.
    dataids : list of int, optional
        If provided, only load these household IDs. Otherwise load all.
    appliance_columns_only : bool, default False
        If True, return only dataid, local_15min, and APPLIANCE_COLUMNS (grid, solar, car, air, furnace/heater).

    Returns
    -------
    pd.DataFrame
        Columns include dataid, local_15min, and circuit columns (kW). local_15min is parsed to datetime.
    """
    if region is None:
        region = "california"
    region = region.lower()
    table_name = f"15minute_data_{region}"
    base = _data_dir_path(data_dir)
    path = base / f"{table_name}.sqlite3"
    if not path.exists():
        raise FileNotFoundError(f"Dataport SQLite not found: {path}")

    query = f'SELECT * FROM "{table_name}"'
    if dataids is not None:
        ids_str = ",".join(str(i) for i in dataids)
        query += f" WHERE dataid IN ({ids_str})"
    query += " ORDER BY dataid, local_15min"

    df = pd.read_sql_query(query, f"sqlite:///{path}")
    # Parse timestamps as UTC to avoid mixed tz-aware/naive issues; downstream code
    # treats this as the UTC time base.
    df["local_15min"] = pd.to_datetime(df["local_15min"], utc=True, errors="coerce")

    for col in df.columns:
        if col in ("dataid", "local_15min"):
            continue
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
    """
    Load 15-minute circuit-level data from all (or selected) Dataport regions and concatenate.

    Adds a "region" column so households are uniquely (region, dataid). Skips regions whose
    SQLite file is missing.

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing 15minute_data_<region>.sqlite3 files.
    regions : list of str, optional
        Regions to load, e.g. ["california", "austin", "newyork"]. If None, loads all available.
    dataids : list of int, optional
        If provided, only load these household IDs per region.
    appliance_columns_only : bool, default False
        If True, return only dataid, local_15min, region, and APPLIANCE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Columns include region, dataid, local_15min, and circuit columns.
    """
    base = _data_dir_path(data_dir)
    to_load = regions if regions is not None else get_available_regions(data_dir)
    if not to_load:
        return pd.DataFrame()

    parts = []
    for reg in to_load:
        path = base / f"15minute_data_{reg}.sqlite3"
        if not path.exists():
            continue
        one = load_15min_data(
            data_dir=base, region=reg, dataids=dataids, appliance_columns_only=appliance_columns_only
        )
        one["region"] = reg
        parts.append(one)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # Ensure region is first column for clarity
    cols = ["region", "dataid", "local_15min"] + [c for c in df.columns if c not in ("region", "dataid", "local_15min")]
    return df[[c for c in cols if c in df.columns]]


# PR real power CSV (gzipped): 15-min data, September 2023, same schema as SQLite
PR_REALPOWER_CSV = "pr_realpower_09-2023_15min.csv.gz"
PR_REGION_LABEL = "pr_2023"


def load_pr_realpower_15min(
    data_dir: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    Load 15-minute circuit-level data from pr_realpower_09-2023_15min.csv.gz.

    Same semantic schema as SQLite 15-min data. Adds region = "pr_2023" so
    (region, dataid) is unique. Returns only dataid, local_15min, region, and
    APPLIANCE_COLUMNS. If the file is missing, returns an empty DataFrame.

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing the gzipped CSV. Defaults to data/raw_data/data_port.

    Returns
    -------
    pd.DataFrame
        Columns: region, dataid, local_15min, and appliance columns (kW).
    """
    base = _data_dir_path(data_dir)
    path = base / PR_REALPOWER_CSV
    if not path.exists():
        return pd.DataFrame()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    # Parse timestamps as UTC for consistency with SQLite loader.
    df["local_15min"] = pd.to_datetime(df["local_15min"], utc=True, errors="coerce")

    for col in df.columns:
        if col in ("dataid", "local_15min"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    keep = ["dataid", "local_15min"] + [c for c in APPLIANCE_COLUMNS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["region"] = PR_REGION_LABEL
    cols = ["region", "dataid", "local_15min"] + [c for c in df.columns if c not in ("region", "dataid", "local_15min")]
    return df[[c for c in cols if c in df.columns]]


def _households_with_appliance(df: pd.DataFrame, cols: List[str]) -> Set[Tuple[str, int]]:
    """Set of (region, dataid) that have at least one non-zero 15-min reading in any of cols."""
    present: Set[Tuple[str, int]] = set()
    for (region, dataid), g in df.groupby(["region", "dataid"]):
        for c in cols:
            if c in g.columns and (g[c].fillna(0) != 0).any():
                present.add((region, dataid))
                break
    return present


def get_appliance_households(
    df: pd.DataFrame,
    manifest: bool = False,
) -> Union[Set[Tuple[str, int]], Tuple[Set[Tuple[str, int]], pd.DataFrame]]:
    """
    Return (region, dataid) households that have at least one of HP, PV, EV, or AC.

    Parameters
    ----------
    df : pd.DataFrame
        15-min data with columns region, dataid, and appliance columns.
    manifest : bool, default False
        If True, also return a manifest DataFrame with columns region, dataid,
        has_ev, has_pv, has_ac, has_hp.

    Returns
    -------
    set of (region, dataid), or (set, manifest DataFrame) if manifest=True.
    """
    households_ev = _households_with_appliance(df, COLS_EV)
    households_hp = _households_with_appliance(df, COLS_HP)
    households_pv = _households_with_appliance(df, COLS_PV)
    households_ac = _households_with_appliance(df, COLS_AC)
    any_appliance = households_ev | households_hp | households_pv | households_ac

    if not manifest:
        return any_appliance

    rows = []
    for (region, dataid) in sorted(any_appliance):
        rows.append({
            "region": region,
            "dataid": dataid,
            "has_ev": (region, dataid) in households_ev,
            "has_pv": (region, dataid) in households_pv,
            "has_ac": (region, dataid) in households_ac,
            "has_hp": (region, dataid) in households_hp,
        })
    manifest_df = pd.DataFrame(rows)
    return any_appliance, manifest_df


def load_electric_vehicles(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """
    Load EV metadata from ev_and_weather.zip (electric_vehicles.csv).

    Columns: dataid, vehicle_type, model_year, quick_charge_port, delivery_date,
    ownership_status, lease_end_date. Join to 15-min data by dataid.
    """
    data_dir = Path(data_dir or _DEFAULT_DATA_PORT_DIR)
    zip_path = data_dir / "ev_and_weather.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"ev_and_weather.zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("ev_and_weather/electric_vehicles.csv") as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    return df


def load_weather(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """
    Load hourly weather from ev_and_weather.zip (weather.csv).

    Columns include localhour, temperature, humidity, pressure, wind_speed,
    cloud_cover, irradiance, etc. One location (Austin area). Join to 15-min
    data by rounding local_15min to hour and matching localhour.
    """
    data_dir = Path(data_dir or _DEFAULT_DATA_PORT_DIR)
    zip_path = data_dir / "ev_and_weather.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"ev_and_weather.zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("ev_and_weather/weather.csv") as f:
            df = pd.read_csv(io.BytesIO(f.read()), nrows=None)
    df["localhour"] = pd.to_datetime(df["localhour"])
    return df


def load_pricing_events(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """
    Load pricing events from project_specific_datasets.zip (pricing_events.csv).

    Columns: eventid, eventdate, eventtype, notes. Use to mark event dates on load curves.
    """
    data_dir = Path(data_dir or _DEFAULT_DATA_PORT_DIR)
    zip_path = data_dir / "project_specific_datasets.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"project_specific_datasets.zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("project_specific_datasets/pricing_events.csv") as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df["eventdate"] = pd.to_datetime(df["eventdate"])
    return df


def load_pricing_events_notifications(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """
    Load pricing event notifications from project_specific_datasets.zip.

    Columns: notificationID, pricingeventID, notification_time, notification_type,
    notification_notes, group_name.
    """
    data_dir = Path(data_dir or _DEFAULT_DATA_PORT_DIR)
    zip_path = data_dir / "project_specific_datasets.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"project_specific_datasets.zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("project_specific_datasets/pricing_events_notifications.csv") as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df["notification_time"] = pd.to_datetime(df["notification_time"])
    return df


def load_civita_text_messages(data_dir: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """
    Load demand-response text messages from project_specific_datasets.zip (civita_text_messages.csv).

    Columns: id, datetime_sent_cstcdt, message_text. Use to mark message dates on load curves.
    """
    data_dir = Path(data_dir or _DEFAULT_DATA_PORT_DIR)
    zip_path = data_dir / "project_specific_datasets.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"project_specific_datasets.zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("project_specific_datasets/civita_text_messages.csv") as f:
            df = pd.read_csv(io.BytesIO(f.read()))
    df["datetime_sent_cstcdt"] = pd.to_datetime(df["datetime_sent_cstcdt"])
    return df
