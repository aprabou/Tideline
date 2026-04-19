from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from google.cloud import storage
from tqdm import tqdm

CALCOFI_ZIP_URL = "https://calcofi.org/downloads/database/CalCOFI_Database_194903-202105_csv_16October2023.zip"
OUTPUT_PATH = Path("data/raw/ingestion/calcofi_bottle.parquet")
DOWNLOAD_PATH = Path("data/raw/ingestion/calcofi_source.zip")
GCS_BLOB_PATH = "ingestion/calcofi_bottle.parquet"

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


def download_zip(url: str = CALCOFI_ZIP_URL, out_path: Path = DOWNLOAD_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        with out_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="calcofi zip",
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))

    return out_path


def load_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        bottle_name = next(n for n in names if n.endswith("Bottle.csv"))
        cast_name = next(n for n in names if n.endswith("Cast.csv"))
        bottle = pd.read_csv(zf.open(bottle_name), low_memory=False, encoding="latin-1")
        cast = pd.read_csv(
            zf.open(cast_name), low_memory=False, encoding="latin-1",
            usecols=["Cst_Cnt", "Date", "Lat_Dec", "Lon_Dec", "St_Line", "St_Station"],
        )
        return bottle.merge(cast, on="Cst_Cnt", how="left")


def transform_bottle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    date_col = "Date"
    lat_col = "Lat_Dec" if "Lat_Dec" in df.columns else "lat"
    lon_col = "Lon_Dec" if "Lon_Dec" in df.columns else "lon"

    required = [
        "St_Line",
        "St_Station",
        "Cst_Cnt",
        "Depthm",
        "T_degC",
        "Salnty",
        "ChlorA",
        date_col,
        lat_col,
        lon_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CalCOFI dataset missing required columns: {missing}")

    out = df[required].copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.floor("D")
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")

    out = out[(out["date"] >= pd.Timestamp("2014-01-01"))]
    out = out[(out["lat"] >= 32) & (out["lat"] <= 35) & (out["lon"] >= -121) & (out["lon"] <= -117)]

    out["station"] = out["St_Line"].astype(str).str.strip() + "-" + out["St_Station"].astype(str).str.strip()
    out["depth_m"] = pd.to_numeric(out["Depthm"], errors="coerce")
    out["temp_c"] = pd.to_numeric(out["T_degC"], errors="coerce")
    out["salinity"] = pd.to_numeric(out["Salnty"], errors="coerce")
    out["chlorophyll_a"] = pd.to_numeric(out["ChlorA"], errors="coerce")

    out = out[["date", "station", "lat", "lon", "depth_m", "temp_c", "salinity", "chlorophyll_a"]]
    out = out.dropna(subset=["date", "lat", "lon"]).sort_values("date").reset_index(drop=True)
    return out


def run_ingestion() -> Path | None:
    bucket_name = os.environ.get("GCS_BUCKET")
    if bucket_name:
        try:
            if gcs_blob_exists(bucket_name, GCS_BLOB_PATH):
                log.info("skipping; gs://%s/%s already exists", bucket_name, GCS_BLOB_PATH)
                return None
        except Exception as exc:  # noqa: BLE001
            log.warning("could not check GCS blob existence: %s", exc)

    zip_path = download_zip()
    raw_df = load_csv_from_zip(zip_path)
    result = transform_bottle_dataframe(raw_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
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
