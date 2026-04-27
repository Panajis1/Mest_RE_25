"""Weather loaders — merges data/envdata.py (MeteoSwiss) and data/get_weather_timeseries.py (Open-Meteo)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── MeteoSwiss default stations (Romande Energie coverage area) ───────────────

_DEFAULT_STATIONS = [
    {"name": "Biere", "id": "BIE"},
    {"name": "St.Prex", "id": "PSI"},
    {"name": "Vevey / Corseaux", "id": "VEV"},
    {"name": "Villars-Tiercelin", "id": "VIT"},
    {"name": "Mathod", "id": "MAR"},
    {"name": "Bullet / La Fretaz", "id": "FRU"},
]

_METEOSWISS_URL = (
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/"
    "{sid_lower}/ogd-smn_{sid_lower}_t_historical_2020-2029.csv"
)

_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ── Shared normalisation ──────────────────────────────────────────────────────

def _standardize(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return DataFrame with tz-naive UTC index and columns [dt_utc, t_2m_C, global_rad_W]."""
    df = df.copy()
    dt = pd.to_datetime(df[time_col], errors="coerce")
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    df["dt_utc"] = dt
    return df[["dt_utc", "t_2m_C", "global_rad_W"]].sort_values("dt_utc").reset_index(drop=True)


# ── MeteoSwiss backend ────────────────────────────────────────────────────────

def load_meteoswiss(
    stations: Optional[list] = None,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download MeteoSwiss station data and return a 15-min regional average.

    Args:
        stations: List of dicts with 'name' and 'id' keys. Defaults to Romande
            Energie coverage stations.
        cache_path: If provided, save/load the result as parquet to avoid
            re-downloading on subsequent calls.

    Returns:
        DataFrame with columns [dt_utc, t_2m_C, global_rad_W] at 15-min resolution,
        tz-naive UTC.
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    if stations is None:
        stations = _DEFAULT_STATIONS

    station_dfs = []
    for station in stations:
        sid = station["id"]
        url = _METEOSWISS_URL.format(sid_lower=sid.lower())
        try:
            raw = pd.read_csv(url, delimiter=";")
            raw["t_2m_C"] = raw["tre200s0"]
            raw["global_rad_W"] = raw["gre000z0"]
            raw["timestamp"] = pd.to_datetime(raw["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True)
            station_dfs.append(raw[["timestamp", "t_2m_C", "global_rad_W"]])
        except Exception as exc:
            print(f"[WeatherLoader] Skipping station {station['name']} ({sid}): {exc}")

    if not station_dfs:
        raise RuntimeError("No MeteoSwiss stations could be loaded.")

    combined = pd.concat(station_dfs, ignore_index=True).sort_values("timestamp")
    average = (
        combined.set_index("timestamp")
        .groupby(level=0)
        .mean(numeric_only=True)
        .resample("15min")
        .interpolate()
    )
    if average.index.tz is not None:
        average.index = average.index.tz_convert("UTC").tz_localize(None)

    result = average.reset_index().rename(columns={"timestamp": "dt_utc"})

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)

    return result


# ── Open-Meteo backend ────────────────────────────────────────────────────────

def load_open_meteo(
    lat: float,
    lon: float,
    start: str,
    end: str,
    unit: str = "metric",
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch hourly temperature and shortwave radiation from the Open-Meteo archive API.

    Resamples to 15-min resolution via linear interpolation.

    Args:
        lat, lon: Coordinates.
        start, end: ISO date strings 'YYYY-MM-DD' (inclusive).
        unit: 'metric'/'celsius' or 'imperial'/'fahrenheit'.
        cache_path: If provided, cache result as parquet.

    Returns:
        DataFrame with columns [dt_utc, t_2m_C, global_rad_W], tz-naive UTC, 15-min.
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    temperature_unit = "celsius" if unit in ("metric", "celsius") else "fahrenheit"
    start_dt, end_dt = date.fromisoformat(start), date.fromisoformat(end)
    if start_dt > end_dt:
        raise ValueError(f"start ({start}) must be on or before end ({end})")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ["temperature_2m", "shortwave_radiation"],
        "timezone": "UTC",
        "temperature_unit": temperature_unit,
    }
    resp = requests.get(_OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")

    hourly = data.get("hourly", {})
    df = pd.DataFrame({
        "dt_utc": pd.to_datetime(hourly["time"]).tz_localize("UTC").tz_localize(None),
        "t_2m_C": hourly["temperature_2m"],
        "global_rad_W": hourly["shortwave_radiation"],
    })

    # Resample to 15-min
    df_15 = (
        df.set_index("dt_utc")
        .resample("15min")
        .interpolate("time")
        .reset_index()
    )

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df_15.to_parquet(cache_path, index=False)

    return df_15


# ── Convenience class ─────────────────────────────────────────────────────────

class WeatherLoader:
    """Thin wrapper that selects MeteoSwiss or Open-Meteo backend via config.

    Args:
        backend: 'meteoswiss' or 'open_meteo'.
        cache_dir: Directory to cache downloaded data. Pass None to disable.
        open_meteo_coords: (lat, lon) tuple required for open_meteo backend.
        stations: Custom station list for meteoswiss backend.
    """

    def __init__(
        self,
        backend: str = "meteoswiss",
        cache_dir: Optional[Path] = None,
        open_meteo_coords: Optional[tuple] = None,
        stations: Optional[list] = None,
    ):
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.open_meteo_coords = open_meteo_coords
        self.stations = stations

    def load(self, dt_start: Optional[str] = None, dt_end: Optional[str] = None) -> pd.DataFrame:
        """Load weather data for the given date range.

        Returns DataFrame with [dt_utc, t_2m_C, global_rad_W].
        """
        cache_path = None
        if self.cache_dir is not None:
            fname = f"weather_{self.backend}_{dt_start}_{dt_end}.parquet"
            cache_path = self.cache_dir / fname

        if self.backend == "meteoswiss":
            return load_meteoswiss(stations=self.stations, cache_path=cache_path)

        if self.backend == "open_meteo":
            if self.open_meteo_coords is None:
                raise ValueError("open_meteo_coords=(lat, lon) is required for 'open_meteo' backend")
            lat, lon = self.open_meteo_coords
            if dt_start is None or dt_end is None:
                raise ValueError("dt_start and dt_end are required for 'open_meteo' backend")
            return load_open_meteo(lat, lon, dt_start, dt_end, cache_path=cache_path)

        raise ValueError(f"Unknown backend: {self.backend!r}. Use 'meteoswiss' or 'open_meteo'.")
