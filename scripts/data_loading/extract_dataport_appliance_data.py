"""
Extract Dataport load curves and appliance loads for households with HP, PV, EV, or AC.

Output: long-format Parquet and manifest CSV in data/processed_data/.
Run from repo root: python scripts/extract_dataport_appliance_data.py
"""
# %%
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from data.load_dataport import (
    COLS_AC,
    COLS_EV,
    COLS_HP,
    COLS_PV,
    get_available_regions,
    get_appliance_households,
    load_all_15min_data,
    load_electric_vehicles,
    load_weather,
    load_pr_realpower_15min,
)
from data.get_weather_timeseries import get_weather_timeseries
# %%
# Region -> timezone for local_15min to UTC (SQLite stores local wall-clock time)
REGION_TZ = {
    "austin": "America/Chicago",
    "california": "America/Los_Angeles",
    "newyork": "America/New_York",
    "pr_2023": "America/Chicago",  # CSV timestamps are -05
}

SOURCE_LABEL = "dataport"
OUTPUT_DIR = _repo_root / "data" / "processed_data"

# Type -> list of columns to sum for Value_KW_mean
TYPE_COLUMNS = {
    "TOT": ["grid"],
    "EV": COLS_EV,
    "PV": COLS_PV,
    "AC": COLS_AC,
    "HP": COLS_HP,
}

# Representative coordinates per Dataport region for Open‑Meteo fallback
REGION_COORDS = {
    "austin": (30.292432, -97.699662),
    "california": (32.778033, -117.151885),
    "newyork": (42.421658, -76.498564),
    # Rough centroid for Puerto Rico (if present)
    "pr_2023": (18.2208, -66.5901),
}


def local_to_utc(df: pd.DataFrame) -> pd.Series:
    """
    Normalize local_15min to datetime64[ns, UTC] using region-specific timezones.

    `local_15min` from the Dataport loaders is parsed as a *naive local wall-clock*
    timestamp (timezone/offset stripped upstream). Here we:
    1) Interpret `local_15min` as local time in the region's IANA timezone.
    2) Convert to UTC for a consistent global time base.
    """
    # Start from a datetime representation of local_15min (local wall-clock).
    # If it already has a timezone for any reason, strip it so we always work
    # with naive local wall-clock values here.
    local_ts = pd.to_datetime(df["local_15min"], errors="coerce")
    if getattr(local_ts.dt, "tz", None) is not None:
        local_ts = local_ts.dt.tz_localize(None)

    # Prepare an empty timezone-aware UTC Series aligned with df
    dt_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    for region, tz_name in REGION_TZ.items():
        mask = df["region"] == region
        if not mask.any():
            continue
        sub = local_ts[mask]
        # Localize naive local times to the region timezone, then convert to UTC.
        # Any ambiguous/nonexistent times around DST transitions are marked as NaT.
        sub_local = sub.dt.tz_localize(
            ZoneInfo(tz_name),
            nonexistent="NaT",
            ambiguous="NaT",
        )
        dt_utc.loc[mask] = sub_local.dt.tz_convert("UTC")

    return dt_utc


def wide_to_long(appliance_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform filtered wide 15-min data to long format: Source, ID customer, type, Value_KW_mean, dt_utc.
    Only emit types that each household has (plus TOT for all).
    """
    # Add dt_utc
    appliance_df = appliance_df.copy()
    appliance_df["dt_utc"] = local_to_utc(appliance_df)

    # Build type columns that exist in data
    for t, cols in TYPE_COLUMNS.items():
        if t == "TOT":
            appliance_df["_TOT"] = appliance_df["grid"] if "grid" in appliance_df.columns else 0.0
        else:
            existing = [c for c in cols if c in appliance_df.columns]
            appliance_df[f"_{t}"] = appliance_df[existing].sum(axis=1) if existing else 0.0

    # Manifest: which types each (region, dataid) has
    type_flags = {}
    for _, row in manifest_df.iterrows():
        key = (row["region"], row["dataid"])
        type_flags[key] = ["TOT"]
        if row.get("has_ev"):
            type_flags[key].append("EV")
        if row.get("has_pv"):
            type_flags[key].append("PV")
        if row.get("has_ac"):
            type_flags[key].append("AC")
        if row.get("has_hp"):
            type_flags[key].append("HP")

    rows = []
    for (region, dataid), g in appliance_df.groupby(["region", "dataid"]):
        types_to_emit = type_flags.get((region, dataid), ["TOT"])
        for _, r in g.iterrows():
            dt_utc = r["dt_utc"]
            for t in types_to_emit:
                val = r.get(f"_{t}", 0.0)
                if pd.isna(val):
                    val = 0.0
                rows.append({
                    "Source": SOURCE_LABEL,
                    "ID customer": dataid,
                    "region": region,
                    "type": t,
                    "Value_KW_mean": float(val),
                    "dt_utc": dt_utc,
                })

    out = pd.DataFrame(rows)
    out["ID customer"] = out["ID customer"].astype("int64")
    return out


def main() -> None:
    data_dir = _repo_root / "data" / "raw_data" / "data_port"

    # Load 15-min from SQLite
    available = get_available_regions(data_dir)
    if not available:
        raise FileNotFoundError(f"No Dataport SQLite files found in {data_dir}")
    df = load_all_15min_data(
        data_dir=data_dir,
        regions=None,
        dataids=None,
        appliance_columns_only=True,
    )

    # Optionally add PR CSV
    pr_df = load_pr_realpower_15min(data_dir)
    if not pr_df.empty:
        df = pd.concat([df, pr_df], ignore_index=True)
        print(f"Loaded PR CSV: {len(pr_df)} rows, region {pr_df['region'].iloc[0]}")

    # Appliance households and manifest
    any_appliance, manifest_df = get_appliance_households(df, manifest=True)
    household_keys = pd.DataFrame(sorted(any_appliance), columns=["region", "dataid"])
    appliance_df = df.merge(household_keys, on=["region", "dataid"]).copy()

    print(f"Appliance households: {len(any_appliance)}")
    print(f"Filtered rows: {len(appliance_df)}")

    # Optional: EV overlap report
    try:
        ev = load_electric_vehicles(data_dir)
        ev_dataids = set(ev["dataid"].unique())
        df_dataids = set(df["dataid"].unique())
        in_both = ev_dataids & df_dataids
        ev_metered = manifest_df[manifest_df["has_ev"]].shape[0]
        print("\n--- EV metadata vs 15-min overlap ---")
        print(f"EV metadata dataids: {len(ev_dataids)}")
        print(f"In 15-min data: {len(in_both)}")
        print(f"Appliance households with EV (metered): {ev_metered}")
    except FileNotFoundError:
        pass

    # Long format (per-household, per-type)
    long_df = wide_to_long(appliance_df, manifest_df)

    # Derive local time per region from dt_utc.
    #
    # Note: pandas does not support a single datetime64 column with mixed timezones.
    # We therefore store dt_local as naive local wall-clock time (no tz) while keeping
    # dt_utc as timezone-aware UTC. The mapping from `region` -> timezone is in REGION_TZ.
    long_df["dt_utc"] = pd.to_datetime(long_df["dt_utc"], utc=True, errors="coerce")
    long_df["dt_local"] = pd.NaT
    for region, tz_name in REGION_TZ.items():
        mask = long_df["region"] == region
        if mask.any():
            long_df.loc[mask, "dt_local"] = (
                long_df.loc[mask, "dt_utc"]
                .dt.tz_convert(ZoneInfo(tz_name))
                .dt.tz_localize(None)
            )

    # 1) Attach Dataport packaged hourly weather where available
    try:
        weather = load_weather(data_dir)
    except FileNotFoundError:
        weather = pd.DataFrame()

    if not weather.empty and "localhour" in weather.columns:
        # Build hourly key from local time (drop tz for join with naive localhour)
        long_df["local_hour"] = (
            long_df["dt_local"].dt.floor("H").dt.tz_localize(None)
        )
        weather_hourly = weather.rename(
            columns={
                "temperature": "temp_station",
                "irradiance": "glob_rad_station",
            }
        )
        cols_keep = [
            c
            for c in ["localhour", "temp_station", "glob_rad_station"]
            if c in weather_hourly.columns
        ]
        weather_hourly = weather_hourly[cols_keep].drop_duplicates(subset=["localhour"])

        long_df = long_df.merge(
            weather_hourly,
            how="left",
            left_on="local_hour",
            right_on="localhour",
        )
        long_df = long_df.drop(columns=["local_hour", "localhour"])
    else:
        long_df["temp_station"] = pd.NA
        long_df["glob_rad_station"] = pd.NA

    # 2) Open‑Meteo fallback per region to fill gaps or full coverage
    fallback_parts: list[pd.DataFrame] = []
    for region in sorted(long_df["region"].dropna().unique()):
        coords = REGION_COORDS.get(region)
        if coords is None:
            continue
        lat, lon = coords

        sub = long_df[long_df["region"] == region]
        if sub.empty:
            continue

        # Use local dates for the Open‑Meteo query
        dt_local_min = sub["dt_local"].min()
        dt_local_max = sub["dt_local"].max()
        if pd.isna(dt_local_min) or pd.isna(dt_local_max):
            continue

        start_date = dt_local_min.date().isoformat()
        end_date = dt_local_max.date().isoformat()

        try:
            w = get_weather_timeseries(
                lat=lat,
                lon=lon,
                start=start_date,
                end=end_date,
                location_name=f"dataport_{region}",
                unit="metric",
            )
        except Exception as exc:
            print(f"Open‑Meteo fallback failed for region {region}: {exc}")
            continue

        w = w.rename(
            columns={
                "temperature_2m": "temp_fallback",
                "shortwave_radiation": "glob_rad_fallback",
            }
        )
        cols_keep = [
            c
            for c in ["dt_utc", "temp_fallback", "glob_rad_fallback"]
            if c in w.columns
        ]
        if not cols_keep:
            continue

        w = w[cols_keep].copy()
        w["dt_utc"] = pd.to_datetime(w["dt_utc"], utc=True, errors="coerce")
        w["region"] = region
        fallback_parts.append(w)

    if fallback_parts:
        fallback_weather = pd.concat(fallback_parts, ignore_index=True)
        long_df = long_df.merge(
            fallback_weather,
            how="left",
            on=["region", "dt_utc"],
        )
    else:
        long_df["temp_fallback"] = pd.NA
        long_df["glob_rad_fallback"] = pd.NA

    # 3) Final weather columns: prefer Dataport station weather, fall back to Open‑Meteo
    for col_station, col_fb, col_final in [
        ("temp_station", "temp_fallback", "temp"),
        ("glob_rad_station", "glob_rad_fallback", "glob_rad"),
    ]:
        if col_station not in long_df.columns:
            long_df[col_station] = pd.NA
        if col_fb not in long_df.columns:
            long_df[col_fb] = pd.NA
        long_df[col_final] = long_df[col_station]
        mask_missing = long_df[col_final].isna()
        long_df.loc[mask_missing, col_final] = long_df.loc[mask_missing, col_fb]

    # 4) Standardize column names and persist parquet
    long_df = long_df.rename(
        columns={
            "Source": "source",
            "ID customer": "id_customer",
            "Value_KW_mean": "value_kw_mean",
        }
    )

    # Ensure dtypes
    long_df["id_customer"] = long_df["id_customer"].astype("int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / "dataport_appliance_households.parquet"
    manifest_path = OUTPUT_DIR / "dataport_appliance_manifest.csv"

    out_cols = [
        "dt_utc",
        "dt_local",
        "source",
        "id_customer",
        "type",
        "temp",
        "glob_rad",
        "value_kw_mean",
    ]
    # Keep any extra metadata columns (e.g. region) but write core schema first
    for c in ["region"]:
        if c in long_df.columns and c not in out_cols:
            out_cols.append(c)

    long_df[out_cols].to_parquet(parquet_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nWrote {parquet_path}")
    print(f"Wrote {manifest_path}")

# %%
if __name__ == "__main__":
    main()

# %%
dp = pd.read_parquet(OUTPUT_DIR / "dataport_appliance_households.parquet")
# %%
path = '/Users/alan/Desktop/ETH/cs_re/Mest_RE_25/data/raw_data/data_port'
dp = load_all_15min_data(data_dir=path)
# %%