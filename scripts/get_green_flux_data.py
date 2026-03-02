# %%
from zoneinfo import ZoneInfo

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the repository root to sys.path to allow absolute imports from 'data'
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from data.get_weather_timeseries import get_weather_timeseries
# %%
# 1. LOAD THE DATA
# To save memory and speed up loading, we only read the columns we actually need.
file_path = "/Users/alan/Desktop/ETH/cs_re/Mest_RE_25/data/raw_data/uk_national_grid/GreenFlux15Minute.csv"
cols_to_keep = ["ChargerID", "TransactionID", "StartTime", "Avg_Amps_Drawn"]
processed_data_path = "/Users/alan/Desktop/ETH/cs_re/Mest_RE_25/data/processed_data/uk_national_grid_green_flux.parquet"
#%%

# ============================================================
# New unified pipeline with local time and weather enrichment
# ============================================================

df2 = pd.read_csv(file_path, usecols=cols_to_keep, parse_dates=["StartTime"])

# Vectorized power calculation
df2["Avg_Amps_Drawn"] = df2["Avg_Amps_Drawn"].fillna(0)
df2["value_kw_mean"] = np.where(
    df2["TransactionID"] == 0,
    0.0,
    (df2["Avg_Amps_Drawn"] * 230) / 1000,
)

df2["source"] = "uk_na_grid"
df2["type"] = "EV"

df2 = df2.rename(
    columns={
        "ChargerID": "id_customer",
        "StartTime": "dt_utc",
    }
)

df2["dt_utc"] = pd.to_datetime(df2["dt_utc"], utc=True, errors="coerce")
# dt_local represents UK local time (Europe/London). For winter periods this
# will numerically match UTC, while during DST it will be offset by +1 hour.
df2["dt_local"] = df2["dt_utc"].dt.tz_convert(ZoneInfo("Europe/London"))

gf = df2[["source", "id_customer", "type", "value_kw_mean", "dt_utc", "dt_local"]]

min_time_2 = gf["dt_utc"].min()
max_time_2 = gf["dt_utc"].max()
full_time_index_2 = pd.date_range(
    start=min_time_2,
    end=max_time_2,
    freq="15min",
    tz="UTC",
)

unique_customers_2 = gf["id_customer"].unique()
full_multi_index_2 = pd.MultiIndex.from_product(
    [full_time_index_2, unique_customers_2],
    names=["dt_utc", "id_customer"],
)

gf_indexed = (
    gf.drop_duplicates(subset=["dt_utc", "id_customer"])
    .set_index(["dt_utc", "id_customer"])
)
gf_continuous = gf_indexed.reindex(full_multi_index_2)
gf_continuous["value_kw_mean"] = gf_continuous["value_kw_mean"].fillna(0.0)
gf_continuous["source"] = "uk_na_grid"
gf_continuous["type"] = "EV"
gf_continuous["dt_local"] = gf_continuous.index.get_level_values("dt_utc").tz_convert(
    ZoneInfo("Europe/London")
)

gf_final = gf_continuous.reset_index()

# Weather enrichment for representative UK coordinates
start_date_2 = gf_final["dt_local"].min().date().isoformat()
end_date_2 = gf_final["dt_local"].max().date().isoformat()

weather_uk = get_weather_timeseries(
    lat=51.5,
    lon=-0.1,
    start=start_date_2,
    end=end_date_2,
    location_name="UK_national_grid",
    unit="metric",
)

weather_uk = weather_uk.rename(
    columns={
        "temperature_2m": "temp",
        "shortwave_radiation": "glob_rad",
    }
)
weather_uk["dt_utc"] = pd.to_datetime(
    weather_uk["dt_utc"],
    utc=True,
    errors="coerce",
)
weather_keep_2 = ["dt_utc", "dt_local", "temp", "glob_rad"]
weather_uk = weather_uk[weather_keep_2]

gf_final = gf_final.merge(
    weather_uk,
    how="left",
    on="dt_utc",
    suffixes=("", "_weather"),
)

if "dt_local_weather" in gf_final.columns:
    gf_final = gf_final.drop(columns=["dt_local_weather"])

gf_final = gf_final.sort_values(by=["id_customer", "dt_utc"]).reset_index(drop=True)

gf_final = gf_final[
    [
        "dt_utc",
        "dt_local",
        "source",
        "id_customer",
        "type",
        "temp",
        "glob_rad",
        "value_kw_mean",
    ]
]

# Overwrite the original parquet with the enriched schema
gf_final.to_parquet(processed_data_path)

# %%
path_gf = "/Users/alan/Desktop/ETH/cs_re/Mest_RE_25/data/processed_data/uk_national_grid_green_flux.parquet"
gf = pd.read_parquet(path_gf)
gf.head()
# %%