import pandas as pd


def env_data(stations=None):
    """Download MeteoSwiss station data and return a regional average time series.

    Returns
    -------
    combined_df : DataFrame
        Stacked per-station records with a UTC-aware ``timestamp`` column.
    average_df : DataFrame
        Station-averaged series resampled to 15-min intervals with linear
        interpolation.  Index is a tz-naive ``DatetimeIndex`` so it aligns
        directly with the Romande Energie parquet timestamps.
    """
    if stations is None:
        stations = [
            {"name": "Biere", "id": "BIE"},
            {"name": "St.Prex", "id": "PSI"},
            {"name": "Vevey / Corseaux", "id": "VEV"},
            {"name": "Villars-Tiercelin", "id": "VIT"},
            {"name": "Mathod", "id": "MAR"},
            {"name": "Bullet / La Fretaz", "id": "FRU"},
        ]

    dict_all = {}
    for station in stations:
        try:
            station_name = station["name"]
            sid = station["id"]
            url = (
                f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/"
                f"{sid.lower()}/ogd-smn_{sid.lower()}_t_historical_2020-2029.csv"
            )
            df_all = pd.read_csv(url, delimiter=";")
            df_all["timestamp"] = pd.to_datetime(
                df_all["reference_timestamp"],
                format="%d.%m.%Y %H:%M",
                utc=True,
            )
            df_all["t_2m_C"] = df_all["tre200s0"]
            df_all["global_rad_W"] = df_all["gre000z0"]
            df_all["wind_speed_10m_ms"] = df_all["fkl010z0"]
            df_all["snow_depth_cm"] = df_all["htoauts0"]
            df_all = df_all[
                ["timestamp", "t_2m_C", "global_rad_W", "wind_speed_10m_ms", "snow_depth_cm"]
            ]
            dict_all[station_name] = df_all
        except Exception as e:
            print(
                f"Could not process station {station['name']} "
                f"with id {station['id']}. Error: {e}"
            )
            continue

    if not dict_all:
        return pd.DataFrame(), pd.DataFrame()

    # Stacked per-station records (UTC-aware timestamp column)
    combined_df = pd.concat(dict_all.values(), ignore_index=True)
    combined_df = combined_df.sort_values("timestamp")

    # Regional average indexed by timestamp
    average_df = (
        combined_df
        .set_index("timestamp")
        .groupby(level=0)
        .mean(numeric_only=True)
        .sort_index()
    )

    # Resample to a regular 15-min grid (matches RE parquet resolution)
    # and interpolate gaps left by the 10-min native MeteoSwiss cadence.
    if not average_df.empty:
        average_df = average_df.resample("15min").interpolate()
        if (
            isinstance(average_df.index, pd.DatetimeIndex)
            and average_df.index.tz is not None
        ):
            average_df.index = average_df.index.tz_convert(None)

    return combined_df, average_df
