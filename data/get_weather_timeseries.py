#%%
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
#%%


def _map_temperature_unit(unit: str) -> str:
    """Map user-friendly unit string to Open-Meteo temperature_unit."""
    if unit in ("metric", "celsius"):
        return "celsius"
    if unit in ("imperial", "fahrenheit"):
        return "fahrenheit"
    raise ValueError(f"Unsupported unit '{unit}', use 'metric'/'celsius' or 'imperial'/'fahrenheit'.")


def get_weather_timeseries(
    lat: float,
    lon: float,
    start: str,
    end: str,
    location_name: Optional[str] = None,
    unit: str = "metric",
    out_path: Optional[str] = None,
    show_plot: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly local-time temperature and solar irradiance (shortwave_radiation)
    from Open-Meteo for a given location and date range.

    Parameters
    ----------
    lat, lon : float
        Latitude [-90, 90] and longitude [-180, 180].
    start, end : str
        ISO dates 'YYYY-MM-DD' (inclusive).
    location_name : str, optional
        For metadata only; not sent to the API.
    unit : {'metric','imperial','celsius','fahrenheit'}
        Temperature unit (maps to Open-Meteo 'celsius' or 'fahrenheit').
    out_path : str, optional
        If provided, path to save the timeseries as CSV.
    show_plot : bool
        If True, show a quick matplotlib plot (if matplotlib is installed).

    Returns
    -------
    df : pandas.DataFrame
        Index: local time (as returned by Open-Meteo, hourly).
        Columns: 'temperature_2m', 'shortwave_radiation'.
        Metadata (timezone, elevation, etc.) is stored in df.attrs['meta'].
    """
    # Basic validation
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}.")

    try:
        start_dt = date.fromisoformat(start)
    except ValueError as exc:
        raise ValueError(f"Invalid start date '{start}': {exc}") from exc

    try:
        end_dt = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError(f"Invalid end date '{end}': {exc}") from exc

    if start_dt > end_dt:
        raise ValueError(f"start ({start_dt}) must be on or before end ({end_dt}).")

    today = date.today()
    if end_dt > today:
        raise ValueError(
            f"End date {end_dt.isoformat()} is in the future; "
            "the historical archive only supports past data."
        )

    temperature_unit = _map_temperature_unit(unit)

    # Build API request
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "hourly": ["temperature_2m", "shortwave_radiation"],
        # Use UTC to avoid DST ambiguity in local timestamps; we can always
        # convert to a local timezone downstream if needed.
        "timezone": "UTC",
        "temperature_unit": temperature_unit,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason', 'unknown error')}")

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    sw_rad = hourly.get("shortwave_radiation") or []

    if not times:
        raise RuntimeError("No hourly 'time' values returned by Open-Meteo.")

    if not temps or not sw_rad:
        raise RuntimeError(
            "Missing expected hourly variables 'temperature_2m' and/or 'shortwave_radiation'."
        )

    if not (len(times) == len(temps) == len(sw_rad)):
        raise RuntimeError(
            "Length mismatch in hourly arrays: "
            f"time={len(times)}, temperature_2m={len(temps)}, shortwave_radiation={len(sw_rad)}."
        )

    # Build timezone-aware DatetimeIndex. Times are returned in UTC (no DST ambiguity).
    tz_name = data.get("timezone") or "UTC"
    dt_local = pd.to_datetime(times).tz_localize(tz_name)

    df_local = pd.DataFrame(
        {
            "temperature_2m": temps,
            "shortwave_radiation": sw_rad,
        },
        index=dt_local,
    )

    # Linear interpolation to a 15‑minute grid
    df_15min = df_local.resample("15min").interpolate("time")

    # Create explicit local and UTC timestamp columns and drop time index
    dt_local_resampled = df_15min.index
    dt_utc_resampled = dt_local_resampled.tz_convert("UTC")

    out = df_15min.copy()
    out.insert(0, "dt_local", dt_local_resampled)
    out.insert(1, "dt_utc", dt_utc_resampled)
    out = out.reset_index(drop=True)

    # Attach metadata (including 'near station' interpretation)
    out.attrs["meta"] = {
        "location_name": location_name,
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "elevation": data.get("elevation"),
        "timezone": data.get("timezone"),
        "utc_offset_seconds": data.get("utc_offset_seconds"),
        "temperature_unit": (data.get("hourly_units") or {}).get("temperature_2m"),
        "shortwave_radiation_unit": (data.get("hourly_units") or {}).get("shortwave_radiation"),
        "note": (
            "Data is from a high-resolution reanalysis grid cell near the requested "
            "coordinates (not a single physical station sensor)."
        ),
    }

    # Optional: save to CSV
    if out_path is not None:
        out.to_csv(out_path, index=False)

    # Optional: quick plot
    if show_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            # Silent fallback; caller still gets df
            pass
        else:
            ax = df_15min["temperature_2m"].plot(
                figsize=(10, 5),
                color="tab:red",
                label="Temperature 2m",
            )
            ax.set_ylabel(out.attrs["meta"]["temperature_unit"] or "Temperature")
            ax2 = ax.twinx()
            df_15min["shortwave_radiation"].plot(
                ax=ax2,
                color="tab:blue",
                label="Shortwave radiation",
            )
            ax2.set_ylabel(
                out.attrs["meta"]["shortwave_radiation_unit"] or "Shortwave radiation"
            )
            ax.set_xlabel("Time (local)")
            ax.set_title("Hourly temperature and shortwave solar radiation")
            plt.tight_layout()
            plt.show()

    return out

#%%

# Example usage
df = get_weather_timeseries(
    lat=48.8566,
    lon=2.3522,
    start="2024-01-01",
    end="2024-01-03",
    location_name="Paris",
    unit="metric",
)
df
# %%
