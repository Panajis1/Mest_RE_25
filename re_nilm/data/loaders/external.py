"""External EV charging dataset loaders (Caltech ACN-Data).

Migrated from data/load_caltech.py. The API token must be passed explicitly
or set via the CALTECH_API_TOKEN environment variable — it is never hardcoded.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CALTECH_VOLTAGE_V = 208.0
CALTECH_SITE = "caltech"
_ACN_API_URL = "https://ev.caltech.edu/api/v1/sessions/{site}/ts"


def fetch_caltech_ev(
    token: Optional[str] = None,
    site: str = CALTECH_SITE,
    max_sessions: int = 50,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch Caltech ACN-Data EV charging sessions and return 15-min kW timeseries.

    Args:
        token: ACN-Data API token. Falls back to the CALTECH_API_TOKEN env var.
        site: ACN site name (default 'caltech').
        max_sessions: Max session pages to fetch (1 page ≈ 1 session).
        out_path: If provided, save result as parquet.

    Returns:
        DataFrame with columns [dt_utc, id_customer, value_kw_mean, type, source],
        resampled to 15-min intervals per parking space.
    """
    if token is None:
        token = os.environ.get("CALTECH_API_TOKEN")
    if not token:
        raise ValueError(
            "API token required. Pass token= or set CALTECH_API_TOKEN environment variable."
        )

    url = _ACN_API_URL.format(site=site)
    all_rows = []
    site_timezone = None

    for page in range(1, max_sessions + 1):
        resp = requests.get(url, params={"page": page}, auth=(token, ""), timeout=30)
        if resp.status_code != 200:
            print(f"[Caltech] Error at page {page}: {resp.status_code}")
            break

        data = resp.json()
        items = data.get("_items", [])
        if not items:
            break

        session = items[0]
        if site_timezone is None:
            site_timezone = session.get("timezone") or "UTC"

        space_id = session.get("spaceID")
        ts_data = session.get("chargingCurrent", {})
        times = ts_data.get("timestamps", [])
        amps = ts_data.get("current", [])

        for t, a in zip(times, amps):
            all_rows.append({
                "dt_utc": t,
                "id_customer": space_id,
                "value_kw_mean": (a * CALTECH_VOLTAGE_V) / 1000.0,
            })

        if "_links" in data and "next" not in data["_links"]:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["dt_utc", "id_customer", "value_kw_mean", "type", "source"])

    df = pd.DataFrame(all_rows)
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True)

    df_15 = (
        df.groupby("id_customer")
        .resample("15min", on="dt_utc")
        .mean(numeric_only=True)
        .reset_index()
    )
    df_15["dt_local"] = df_15["dt_utc"].dt.tz_convert(site_timezone or "UTC")
    df_15["type"] = "EV"
    df_15["source"] = "Caltech"

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_15.to_parquet(out_path, engine="pyarrow", index=False)

    return df_15
