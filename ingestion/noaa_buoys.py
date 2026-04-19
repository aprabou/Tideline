from __future__ import annotations

import logging
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from google.cloud import storage
from tqdm import tqdm

START_YEAR = 2014
END_YEAR = 2023
NDBC_URL_TEMPLATE = (
    "https://www.ndbc.noaa.gov/view_text_file.php?"
    "filename={station}h{year}.txt.gz&dir=data/historical/stdmet/"
)
OUTPUT_PATH = Path("data/raw/ingestion/noaa_buoys_sst.parquet")
GCS_BLOB_PATH = "ingestion/noaa_buoys_sst.parquet"

# Southern California Bight stations.
STATION_COORDS: dict[str, tuple[float, float]] = {
    "46025": (33.749, -119.053),
    "46069": (32.120, -120.780),
    "46086": (32.499, -118.052),
    "46218": (34.452, -120.780),
    "46219": (34.267, -119.839),
    "46221": (34.279, -120.068),
    "46222": (33.618, -118.317),
    "46223": (33.460, -117.767),
    "46224": (33.190, -117.472),
    "46225": (33.749, -119.053),
    "46231": (32.933, -117.391),
    "46232": (32.560, -117.500),
    "46242": (36.785, -122.469),
    "46253": (34.178, -119.435),
    "46254": (34.208, -119.221),
    "46256": (33.700, -118.200),
    "46258": (32.750, -117.500),
    "46259": (34.400, -120.120),
    "46266": (33.911, -118.440),
    "46268": (33.650, -118.130),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def gcs_blob_exists(bucket_name: str, blob_name: str) -> bool:
    """Return True when the target blob exists in the configured bucket."""
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_name).exists(client)


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> None:
    """Upload file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    log.info("uploaded gs://%s/%s", bucket_name, blob_name)


def fetch_historical_text(station_id: str, year: int, timeout: int = 60) -> str | None:
    """Download one station-year NDBC stdmet text file."""
    url = NDBC_URL_TEMPLATE.format(station=station_id, year=year)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    if "not found" in resp.text.lower():
        return None
    return resp.text


def _normalize_year(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(np.where(values < 100, 2000 + values, values), index=series.index)
    return normalized


def parse_stdmet_text(raw_text: str) -> pd.DataFrame:
    """Parse NDBC standard meteorological text into timestamp + SST rows."""
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if len(lines) < 3:
        return pd.DataFrame(columns=["date", "sst"])

    header = lines[0].lstrip("#").split()
    data_text = "\n".join(lines[2:])

    df = pd.read_csv(StringIO(data_text), sep=r"\s+", names=header, dtype=str)
    year_col = "YYYY" if "YYYY" in df.columns else "YY"
    if year_col not in df.columns:
        raise ValueError("NDBC file missing YYYY/YY column")

    required_cols = [year_col, "MM", "DD", "hh", "mm", "WTMP"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"NDBC file missing required column: {col}")

    dt_frame = pd.DataFrame(
        {
            "year": _normalize_year(df[year_col]),
            "month": pd.to_numeric(df["MM"], errors="coerce"),
            "day": pd.to_numeric(df["DD"], errors="coerce"),
            "hour": pd.to_numeric(df["hh"], errors="coerce"),
            "minute": pd.to_numeric(df["mm"], errors="coerce"),
        }
    )
    parsed = pd.DataFrame()
    parsed["date"] = pd.to_datetime(dt_frame, errors="coerce").dt.floor("D")
    parsed["sst"] = pd.to_numeric(df["WTMP"], errors="coerce")
    parsed["sst"] = parsed["sst"].replace(99.0, np.nan)
    parsed = parsed.dropna(subset=["date"])
    return parsed


def build_station_daily_sst(station_id: str, years: range) -> pd.DataFrame:
    """Download + parse all requested years and resample to daily mean SST."""
    frames: list[pd.DataFrame] = []
    for year in tqdm(years, desc=f"years {station_id}", leave=False):
        raw_text = fetch_historical_text(station_id=station_id, year=year)
        if not raw_text:
            continue
        parsed = parse_stdmet_text(raw_text)
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "date", "sst"])

    all_rows = pd.concat(frames, ignore_index=True)
    daily = (
        all_rows.groupby("date", as_index=False)["sst"]
        .mean()
        .assign(station_id=station_id, lat=STATION_COORDS[station_id][0], lon=STATION_COORDS[station_id][1])
        [["station_id", "lat", "lon", "date", "sst"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def run_ingestion() -> Path | None:
    """Run full buoy ingestion and upload output to GCS if configured."""
    bucket_name = os.environ.get("GCS_BUCKET")
    if bucket_name:
        try:
            if gcs_blob_exists(bucket_name, GCS_BLOB_PATH):
                log.info("skipping; gs://%s/%s already exists", bucket_name, GCS_BLOB_PATH)
                return None
        except Exception as exc:  # noqa: BLE001
            log.warning("could not check GCS blob existence: %s", exc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    years = range(START_YEAR, END_YEAR + 1)

    station_frames: list[pd.DataFrame] = []
    for station_id in tqdm(STATION_COORDS, desc="stations"):
        station_df = build_station_daily_sst(station_id=station_id, years=years)
        if not station_df.empty:
            station_frames.append(station_df)

    result = pd.concat(station_frames, ignore_index=True) if station_frames else pd.DataFrame(
        columns=["station_id", "lat", "lon", "date", "sst"]
    )
    result.to_parquet(OUTPUT_PATH, index=False)
    log.info("wrote %s (%d rows)", OUTPUT_PATH, len(result))

    if bucket_name:
        try:
            upload_to_gcs(OUTPUT_PATH, bucket_name, GCS_BLOB_PATH)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to upload to GCS: %s", exc)
    else:
        log.warning("GCS_BUCKET not set; skipping upload")
    return OUTPUT_PATH


def main() -> None:
    run_ingestion()


if __name__ == "__main__":
    main()
