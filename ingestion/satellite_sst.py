from __future__ import annotations

import logging
import os
from pathlib import Path

import xarray as xr
from google.cloud import storage
from tqdm import tqdm

ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc"
VARIABLES = ["CRW_SST", "CRW_SSTANOMALY", "CRW_DHW"]
START_YEAR = 2014
END_YEAR = 2023
BBOX = {"lat_min": 32.0, "lat_max": 36.0, "lon_min": -122.0, "lon_max": -117.0}
OUTPUT_DIR = Path("sst_rasters")

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


def _resolve_coord_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"None of candidate coordinates found: {candidates}")


def _build_slice(values: xr.DataArray, low: float, high: float) -> slice:
    first = float(values.values[0])
    last = float(values.values[-1])
    if first <= last:
        return slice(low, high)
    return slice(high, low)


def subset_year_dataset(ds: xr.Dataset, year: int) -> xr.Dataset:
    available_vars = [name for name in VARIABLES if name in ds.data_vars]
    if len(available_vars) != len(VARIABLES):
        missing = set(VARIABLES) - set(available_vars)
        raise ValueError(f"missing expected variables in dataset: {sorted(missing)}")

    lat_name = _resolve_coord_name(ds, ["latitude", "lat"])
    lon_name = _resolve_coord_name(ds, ["longitude", "lon"])
    time_name = _resolve_coord_name(ds, ["time"])

    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"

    subset = ds[available_vars].sel(
        {
            time_name: slice(year_start, year_end),
            lat_name: _build_slice(ds[lat_name], BBOX["lat_min"], BBOX["lat_max"]),
            lon_name: _build_slice(ds[lon_name], BBOX["lon_min"], BBOX["lon_max"]),
        }
    )
    return subset


def fetch_remote_dataset() -> xr.Dataset:
    return xr.open_dataset(ERDDAP_URL, engine="netcdf4")


def run_ingestion() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bucket_name = os.environ.get("GCS_BUCKET")
    written: list[Path] = []

    with fetch_remote_dataset() as remote_ds:
        for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="satellite years"):
            blob_name = f"sst_rasters/{year}.nc"
            out_path = OUTPUT_DIR / f"{year}.nc"

            if bucket_name:
                try:
                    if gcs_blob_exists(bucket_name, blob_name):
                        log.info("skipping %s; already present in gs://%s/%s", year, bucket_name, blob_name)
                        continue
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not check GCS blob existence for %s: %s", year, exc)

            year_ds = subset_year_dataset(remote_ds, year)
            if year_ds.sizes.get("time", 0) == 0:
                log.warning("no data found for year %s", year)
                continue

            year_ds.to_netcdf(out_path)
            written.append(out_path)
            log.info("wrote %s", out_path)

            if bucket_name:
                try:
                    upload_to_gcs(out_path, bucket_name, blob_name)
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to upload %s to GCS: %s", out_path.name, exc)
            else:
                log.warning("GCS_BUCKET not set; skipping upload for %s", out_path.name)

    return written


def main() -> None:
    run_ingestion()


if __name__ == "__main__":
    main()
