"""
satellite_sst.py — Download NOAA OISST v2.1 SST for the California Current region.

Downloads one annual NetCDF per year via direct HTTP (bypassing OPeNDAP), subsets
to the Southern California Bight bounding box, computes anomaly against the
multi-year mean, and saves one per-day NetCDF under OUTPUT_DIR.
The zlev dummy dimension is preserved for compatibility with satellite_features.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from google.cloud import storage
from tqdm import tqdm

PSL_BASE = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"

START_YEAR = 2014
END_YEAR = 2023
BBOX = {"lat_min": 32.0, "lat_max": 36.0, "lon_min": -122.0, "lon_max": -117.0}
OUTPUT_DIR = Path("data/raw/oisst")
DOWNLOAD_DIR = Path("data/raw/oisst_annual")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def gcs_blob_exists(bucket_name: str, blob_name: str) -> bool:
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_name).exists(client)


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    log.info("uploaded gs://%s/%s", bucket_name, blob_name)


def _lon_to_360(lon: float) -> float:
    return lon % 360.0


def _download_annual_nc(year: int) -> Path | None:
    """Download the full-year OISST NetCDF file via direct HTTP."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOAD_DIR / f"sst.day.mean.{year}.nc"
    if dest.exists():
        log.info("annual file already present: %s", dest)
        return dest

    url = f"{PSL_BASE}/sst.day.mean.{year}.nc"
    log.info("downloading %s → %s", url, dest)
    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with dest.open("wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"OISST {year}"
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
        return dest
    except Exception as exc:
        log.error("failed to download OISST %d: %s", year, exc)
        if dest.exists():
            dest.unlink()
        return None


def _subset_year(nc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]] | None:
    """Open an annual NetCDF, subset to BBOX, return (sst_array, lats, lons, dates)."""
    lon_min_360 = _lon_to_360(BBOX["lon_min"])
    lon_max_360 = _lon_to_360(BBOX["lon_max"])
    try:
        ds = xr.open_dataset(nc_path)
        subset = ds["sst"].sel(
            lat=slice(BBOX["lat_min"], BBOX["lat_max"]),
            lon=slice(lon_min_360, lon_max_360),
        ).load()
        ds.close()
        lats = subset.lat.to_numpy()
        lons = subset.lon.to_numpy()
        dates = [pd.Timestamp(t) for t in pd.to_datetime(subset.time.to_numpy())]
        return subset.to_numpy(), lats, lons, dates
    except Exception as exc:
        log.error("failed to subset %s: %s", nc_path, exc)
        return None


def _save_daily(
    date: pd.Timestamp,
    sst: np.ndarray,
    anom: np.ndarray,
    lats: np.ndarray,
    lons_neg: np.ndarray,
    out_path: Path,
) -> None:
    ds = xr.Dataset(
        {
            "sst": xr.DataArray(
                sst[np.newaxis, np.newaxis, :, :],
                dims=["time", "zlev", "lat", "lon"],
            ),
            "anom": xr.DataArray(
                anom[np.newaxis, np.newaxis, :, :],
                dims=["time", "zlev", "lat", "lon"],
            ),
        },
        coords={
            "time": [date.to_datetime64()],
            "zlev": [0.0],
            "lat": lats,
            "lon": lons_neg,
        },
    )
    ds.to_netcdf(out_path)
    ds.close()


def run_ingestion() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bucket_name = os.environ.get("GCS_BUCKET")
    written: list[Path] = []

    # Pass 1: download annual files and subset to bbox
    log.info("downloading OISST %d-%d via direct HTTP...", START_YEAR, END_YEAR)
    year_data: dict[int, tuple] = {}
    for year in range(START_YEAR, END_YEAR + 1):
        nc_path = _download_annual_nc(year)
        if nc_path is None:
            continue
        result = _subset_year(nc_path)
        if result is not None:
            year_data[year] = result
            log.info("year %d: %d days subsetted", year, len(result[3]))

    if not year_data:
        log.error("no OISST data downloaded — aborting")
        return []

    # Compute climatology (simple mean over all days across all years)
    first = next(iter(year_data.values()))
    lats, lons = first[1], first[2]
    lons_neg = lons - 360.0
    all_sst = np.concatenate([v[0] for v in year_data.values()], axis=0)
    clim_mean = np.nanmean(all_sst, axis=0)  # (lat, lon)
    log.info("climatology computed from %d total days", len(all_sst))

    # Pass 2: save per-day NetCDF files with anomaly
    for year, (sst_arr, _, _, dates) in year_data.items():
        for i, date in enumerate(tqdm(dates, desc=f"saving {year}", leave=False)):
            out_path = OUTPUT_DIR / f"{date.strftime('%Y-%m-%d')}.nc"
            if out_path.exists():
                written.append(out_path)
                continue
            sst_day = sst_arr[i]
            anom_day = sst_day - clim_mean
            _save_daily(date, sst_day, anom_day, lats, lons_neg, out_path)
            written.append(out_path)

            if bucket_name:
                try:
                    upload_to_gcs(out_path, bucket_name, f"oisst/{out_path.name}")
                except Exception as exc:
                    log.warning("GCS upload failed for %s: %s", out_path.name, exc)

        log.info("year %d saved", year)

    log.info("total daily files written: %d", len(written))
    return written


def main() -> None:
    run_ingestion()


if __name__ == "__main__":
    main()
