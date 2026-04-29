#%%

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from data.get_weather_timeseries import get_weather_timeseries

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed_data"

DATAPORT_PATH = PROCESSED_DIR / "dataport_appliance_households.parquet"
GREENFLUX_PATH = PROCESSED_DIR / "uk_national_grid_green_flux.parquet"
PIERPAOLO_PATH = PROCESSED_DIR / "dataset_pierpaolo.parquet"
CALTECH_PATH = PROCESSED_DIR / "caltech_15min_kw.parquet"
HEAPO_PATH = PROCESSED_DIR / "heapo_hp_long.parquet"
UNIFIED_PATH = PROCESSED_DIR / "all_sources_load_with_weather.parquet"


# %%
def load_dataport() -> pd.DataFrame:
    if not DATAPORT_PATH.exists():
        raise FileNotFoundError(f"Dataport parquet not found: {DATAPORT_PATH}")
    df = pd.read_parquet(DATAPORT_PATH)
    # Ensure expected columns are present
    expected = {
        "dt_utc",
        "dt_local",
        "source",
        "id_customer",
        "type",
        "temp",
        "glob_rad",
        "value_kw_mean",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Dataport parquet missing columns: {missing}")
    return df[list(expected) ]


def load_greenflux() -> pd.DataFrame:
    if not GREENFLUX_PATH.exists():
        raise FileNotFoundError(f"GreenFlux parquet not found: {GREENFLUX_PATH}")
    df = pd.read_parquet(GREENFLUX_PATH)
    expected = {
        "dt_utc",
        "dt_local",
        "source",
        "id_customer",
        "type",
        "temp",
        "glob_rad",
        "value_kw_mean",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"GreenFlux parquet missing columns: {missing}")
    return df[list(expected) ]

def fill_weather_data(dataset: str,) -> pd.DataFrame:
    if dataset == "heapo":
        df = load_heapo()
        heapo_mask = df["source"].str.lower() == "heapo"
        heapo_df = df.loc[heapo_mask].copy()
        if not heapo_df.empty:
            heapo_df["dt_utc"] = pd.to_datetime(heapo_df["dt_utc"], utc=True)
            start_date = heapo_df["dt_utc"].min().date().isoformat()
            end_date = heapo_df["dt_utc"].max().date().isoformat()
            heapo_weather = get_weather_timeseries(
                lat=47.3769,
                lon=8.5417,
                start=start_date,
                end=end_date,
            ).rename(
                columns={
                    "temperature_2m": "temp",
                    "shortwave_radiation": "glob_rad",
                }
            )
            heapo_df = heapo_df.merge(heapo_weather[["dt_utc", "temp", "glob_rad"]], on="dt_utc", how="left")
            df.loc[heapo_mask, ["temp", "glob_rad"]] = heapo_df[["temp", "glob_rad"]].values
        return df

    if dataset == "caltech":
        df = load_caltech_data()
        # 34.0549° N, 118.2426° W
        caltech_mask = df["source"].str.lower() == "caltech"
        caltech_df = df.loc[caltech_mask].copy()

        if not caltech_df.empty:
            caltech_df["dt_utc"] = pd.to_datetime(caltech_df["dt_utc"], utc=True)
            start_date = caltech_df["dt_utc"].min().date().isoformat()
            end_date = caltech_df["dt_utc"].max().date().isoformat()

            caltech_weather = get_weather_timeseries(
                lat=34.0549,
                lon=118.2426,
                start=start_date,
                end=end_date,
            ).rename(
                columns={
                    "temperature_2m": "temp",
                    "shortwave_radiation": "glob_rad",
                }
            )

            caltech_df = caltech_df.merge(
                caltech_weather[["dt_utc", "temp", "glob_rad"]],
                on="dt_utc",
                how="left",
            )
            df.loc[caltech_mask, ["temp", "glob_rad"]] = caltech_df[["temp", "glob_rad"]].values
        return df

    if dataset == "pierpaolo":
    
        df = read_pierpaolo_data()

        # if source 'pv estonia' then tallinn : 59.4370° N, 24.7536° E
        estonia_mask = df["source"].str.lower() == "pv estonia"
        estonia_df = df.loc[estonia_mask].copy()

        if not estonia_df.empty:
            # ensure dt_utc is timezone-aware UTC so it matches get_weather_timeseries
            estonia_df["dt_utc"] = pd.to_datetime(estonia_df["dt_utc"], utc=True)

            start_date = estonia_df["dt_utc"].min().date().isoformat()
            end_date = estonia_df["dt_utc"].max().date().isoformat()

            estonia_weather = get_weather_timeseries(
                lat=59.4370,
                lon=24.7536,
                start=start_date,
                end=end_date,
            ).rename(
                columns={
                    "temperature_2m": "temp",
                    "shortwave_radiation": "glob_rad",
                }
            )

            # ensure weather dt_utc has the same UTC dtype
            estonia_weather["dt_utc"] = pd.to_datetime(
                estonia_weather["dt_utc"], utc=True
            )

            # drop any existing temp/glob_rad to avoid _x/_y suffixes, then merge on dt_utc
            estonia_df = (
                estonia_df.drop(columns=["temp", "glob_rad"], errors="ignore")
                .merge(
                    estonia_weather[["dt_utc", "temp", "glob_rad"]],
                    on="dt_utc",
                    how="left",
                )
            )

            # write the weather-based temp and glob_rad back into the original dataframe
            df.loc[estonia_mask, ["temp", "glob_rad"]] = estonia_df[
                ["temp", "glob_rad"]
            ].values

        # if source 'Smartnialmeter' then mix of bern, zurich, luzern:
        # 46.9480° N, 7.4474° E (Bern)
        # 47.3769° N, 8.5417° E (Zurich)
        # 47.0502° N, 8.3093° E (Luzern)
        smart_mask = df["source"].str.lower() == "smartnialmeter"
        smart_df = df.loc[smart_mask].copy()

        if not smart_df.empty:
            smart_df["dt_utc"] = pd.to_datetime(smart_df["dt_utc"], utc=True)
            start_date_ch = smart_df["dt_utc"].min().date().isoformat()
            end_date_ch = smart_df["dt_utc"].max().date().isoformat()

            bern_weather = get_weather_timeseries(
                lat=46.9480,
                lon=7.4474,
                start=start_date_ch,
                end=end_date_ch,
            ).rename(
                columns={
                    "temperature_2m": "temp_bern",
                    "shortwave_radiation": "glob_rad_bern",
                }
            )

            zurich_weather = get_weather_timeseries(
                lat=47.3769,
                lon=8.5417,
                start=start_date_ch,
                end=end_date_ch,
            ).rename(
                columns={
                    "temperature_2m": "temp_zurich",
                    "shortwave_radiation": "glob_rad_zurich",
                }
            )

            luzern_weather = get_weather_timeseries(
                lat=47.0502,
                lon=8.3093,
                start=start_date_ch,
                end=end_date_ch,
            ).rename(
                columns={
                    "temperature_2m": "temp_luzern",
                    "shortwave_radiation": "glob_rad_luzern",
                }
            )

            for wdf in (bern_weather, zurich_weather, luzern_weather):
                wdf["dt_utc"] = pd.to_datetime(wdf["dt_utc"], utc=True)

            swiss_weather = (
                bern_weather[["dt_utc", "temp_bern", "glob_rad_bern"]]
                .merge(
                    zurich_weather[["dt_utc", "temp_zurich", "glob_rad_zurich"]],
                    on="dt_utc",
                    how="inner",
                )
                .merge(
                    luzern_weather[["dt_utc", "temp_luzern", "glob_rad_luzern"]],
                    on="dt_utc",
                    how="inner",
                )
            )

            swiss_weather["temp"] = swiss_weather[
                ["temp_bern", "temp_zurich", "temp_luzern"]
            ].mean(axis=1)
            swiss_weather["glob_rad"] = swiss_weather[
                ["glob_rad_bern", "glob_rad_zurich", "glob_rad_luzern"]
            ].mean(axis=1)
            swiss_weather = swiss_weather[["dt_utc", "temp", "glob_rad"]]

            smart_df = (
                smart_df.drop(columns=["temp", "glob_rad"], errors="ignore")
                .merge(
                    swiss_weather,
                    on="dt_utc",
                    how="left",
                )
            )

            df.loc[smart_mask, ["temp", "glob_rad"]] = smart_df[
                ["temp", "glob_rad"]
            ].values

        return df



def read_pierpaolo_data() -> pd.DataFrame:
    if not PIERPAOLO_PATH.exists():
        raise FileNotFoundError(f"Pierpaolo parquet not found: {PIERPAOLO_PATH}")
    df = pd.read_parquet(PIERPAOLO_PATH)
    # Make column names case insensitive
    col_map = {col.lower(): col for col in df.columns}

    # Define mapping with possible input colname variants (lowercase)
    columns_to_rename = {}
    if "house" in col_map:
        columns_to_rename[col_map["house"]] = "id_customer"
    if "consumption_kw" in col_map:
        columns_to_rename[col_map["consumption_kw"]] = "value_kw_mean"
    if "temperature_total" in col_map:
        columns_to_rename[col_map["temperature_total"]] = "temp"
    if "solar_irradiance" in col_map:
        columns_to_rename[col_map["solar_irradiance"]] = "glob_rad"
    # Change 'type' from 'TOTAL' to 'TOT' if present
    if "type" in df.columns:
        df.loc[df["type"] == "TOTAL", "type"] = "TOT"

    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)

    # For rows where type is EV1 or EV2, create an additional aggregated
    # row per (id_customer, dt_utc) with type == 'EV' and
    # value_kw_mean equal to the sum of EV1 and EV2 at that timestep.
    required_cols = {"id_customer", "dt_utc", "type", "value_kw_mean"}
    if required_cols.issubset(df.columns):
        ev_mask = df["type"].isin(["EV1", "EV2"])
        if ev_mask.any():
            group_cols = ["id_customer", "dt_utc"]
            agg_spec: dict[str, str] = {"value_kw_mean": "sum"}
            # Carry over representative values for other useful columns if present.
            for col in ["source", "dt_local", "temp", "glob_rad"]:
                if col in df.columns:
                    agg_spec[col] = "first"

            ev_agg = (
                df.loc[ev_mask]
                .groupby(group_cols, as_index=False)
                .agg(agg_spec)
            )
            ev_agg["type"] = "EV"

            # Ensure the aggregated frame has the same columns/order as df
            for col in df.columns:
                if col not in ev_agg.columns:
                    ev_agg[col] = pd.NA
            ev_agg = ev_agg[df.columns]

            df = pd.concat([df, ev_agg], ignore_index=True)

    return df

def load_caltech_data() -> pd.DataFrame:
    if not CALTECH_PATH.exists():
        raise FileNotFoundError(f"Caltech parquet not found: {CALTECH_PATH}")
    df = pd.read_parquet(CALTECH_PATH)
    # Make column names case insensitive
    col_map = {col.lower(): col for col in df.columns}
    # Also handle possible 'ID customer' variant
    if "id_customer" in col_map:
        df = df.rename(columns={col_map["id_customer"]: "id_customer"})
    elif "id customer" in col_map:
        df = df.rename(columns={col_map["id customer"]: "id_customer"})
    if "value_kw_mean" in col_map:
        df = df.rename(columns={col_map["value_kw_mean"]: "value_kw_mean"})
    if "temp" in col_map:
        df = df.rename(columns={col_map["temp"]: "temp"})
    if "glob_rad" in col_map:
        df = df.rename(columns={col_map["glob_rad"]: "glob_rad"})
    if "source" in col_map:
        df = df.rename(columns={col_map["source"]: "source"})
    return df

def load_heapo() -> pd.DataFrame:
    if not HEAPO_PATH.exists():
        raise FileNotFoundError(f"Heapo parquet not found: {HEAPO_PATH}")
    df = pd.read_parquet(HEAPO_PATH)
    # Make column names case insensitive
    col_map = {col.lower(): col for col in df.columns}
    if "id_customer" in col_map:
        df = df.rename(columns={col_map["id_customer"]: "id_customer"})
    if "id customer" in col_map:
        df = df.rename(columns={col_map["id customer"]: "id_customer"})
    if "value_kw_mean" in col_map:
        df = df.rename(columns={col_map["value_kw_mean"]: "value_kw_mean"})
    if "temp" in col_map:
        df = df.rename(columns={col_map["temp"]: "temp"})
    if "glob_rad" in col_map:
        df = df.rename(columns={col_map["glob_rad"]: "glob_rad"})
    if "source" in col_map:
        df = df.rename(columns={col_map["source"]: "source"})
    
    df['dt_local'] = df['dt_utc'].dt.tz_convert('Europe/Zurich')
    df['dt_local'] = df['dt_local'].dt.tz_localize(None)

    return df
# %%
def main() -> None:
    dp = load_dataport()
    gf = load_greenflux()
    pp = fill_weather_data(dataset="pierpaolo")
    ct = fill_weather_data(dataset="caltech")
    hp = fill_weather_data(dataset="heapo")
    # Align dtypes where reasonable
    for col in ["id_customer"]:
        if col in dp.columns and col in gf.columns:
            dp[col] = dp[col].astype("str")
            gf[col] = gf[col].astype("str")
            hp[col] = hp[col].astype("str")
    unified = pd.concat([dp, gf, pp, ct, hp], ignore_index=True)
    unified = unified.sort_values([ "id_customer", "dt_utc"]).reset_index(
        drop=True
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(UNIFIED_PATH, index=False)
    print(f"Wrote unified parquet to {UNIFIED_PATH}")


if __name__ == "__main__":
    main()


# %%
all_sources = pd.read_parquet(UNIFIED_PATH)
# %%
caltech_rows = all_sources[all_sources["source"].str.lower() == "dataport"]
print(caltech_rows)
# %%
print('AMUNT OF DATA PER TYPE (QUARTER HOUR)' )
for t in all_sources["type"].unique():
    mask = all_sources["type"] == t
    # For each column, count non-NaN/nonzero values in that type
    print(f"Type: {t}")
    for col in all_sources.columns:
        if col in ["dt_utc", "dt_local", "type", "source", "id_customer", 'temp', 'glob_rad', ]:  # skip meta columns
            continue
        s = all_sources.loc[mask, col]
        if pd.api.types.is_numeric_dtype(s):
            count = s[(~s.isna()) & (s != 0)].count()
        else:
            count = s[~s.isna()].count()
        print(f"  {col}: {count}")
# %%
