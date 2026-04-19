"""
ingest_buoys.py — Pull hourly SST observations from NDBC buoys.

NDBC standard meteorological data:
  https://www.ndbc.noaa.gov/data/realtime2/<station_id>.txt

West Coast stations (extend list as needed):
  46026 San Francisco, 46042 Monterey, 46047 Tanner Banks,
  46086 San Clemente, 46025 Santa Monica, 46011 Santa Maria
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/buoys")
NDBC_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"

WEST_COAST_STATIONS = [
    "46026",  # San Francisco
    "46042",  # Monterey
    "46047",  # Tanner Banks
    "46086",  # San Clemente Basin
    "46025",  # Santa Monica Basin
    "46011",  # Santa Maria Basin
    "46054",  # West Santa Barbara
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_station(station_id: str) -> pd.DataFrame | None:
    """Download the last 45 days of standard met data for one NDBC buoy."""
    url = NDBC_URL.format(station=station_id)
    log.info("fetching %s", url)
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        log.warning("station %s not found", station_id)
        return None
    resp.raise_for_status()

    # NDBC files have two header rows; the second row contains units
    lines = resp.text.splitlines()
    header = lines[0].lstrip("#").split()
    # skip units row (index 1)
    data_text = "\n".join(lines[2:])
    df = pd.read_csv(
        StringIO(data_text),
        names=header,
        sep=r"\s+",
        na_values=["MM", "99.0", "9999"],
    )
    df["station_id"] = station_id
    # Parse timestamp columns into a single datetime
    time_cols = [c for c in ["YY", "MM", "DD", "hh", "mm"] if c in df.columns]
    if len(time_cols) == 5:
        df["time"] = pd.to_datetime(
            df[["YY", "MM", "DD", "hh", "mm"]].rename(
                columns={"YY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}
            )
        )
        df.drop(columns=time_cols, inplace=True)
    return df


def ingest(stations: list[str] = WEST_COAST_STATIONS) -> pd.DataFrame:
    """Fetch all stations and concatenate into a single DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid in stations:
        df = fetch_station(sid)
        if df is not None:
            out = RAW_DIR / f"{sid}.parquet"
            df.to_parquet(out, index=False)
            log.info("saved %s (%d rows)", out.name, len(df))
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined


if __name__ == "__main__":
    df = ingest()
    print(f"Ingested {len(df)} buoy observations across {df['station_id'].nunique()} stations")
