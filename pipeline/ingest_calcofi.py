"""
ingest_calcofi.py — Download Scripps Institution of Oceanography CalCOFI data.

CalCOFI bottle data (CTD + nutrients):
  https://calcofi.org/data/oceanographic-data/bottle-database/

The CSV files are large; this script downloads a recent subset and caches it
to data/raw/calcofi/ as parquet for fast downstream access.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/calcofi")
# CalCOFI publicly available bottle data (most recent decade)
CALCOFI_BOTTLE_URL = "https://calcofi.org/downloads/databases/CalCOFIBottleDatabase.zip"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_bottle_db(out_dir: Path = RAW_DIR) -> Path:
    """Download the CalCOFI bottle database zip to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = out_dir / "CalCOFIBottleDatabase.zip"
    if dest_zip.exists():
        log.info("already cached: %s", dest_zip)
        return dest_zip
    log.info("downloading CalCOFI bottle database (may be large)...")
    with requests.get(CALCOFI_BOTTLE_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest_zip.open("wb") as f:
            for chunk in r.iter_content(chunk_size=4 << 20):
                f.write(chunk)
    log.info("saved %.1f MB", dest_zip.stat().st_size / 1e6)
    return dest_zip


def extract_and_parse(zip_path: Path, out_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Extract CSV from zip, parse, and save as parquet."""
    import zipfile

    parquet_path = out_dir / "calcofi_bottles.parquet"
    if parquet_path.exists():
        log.info("loading cached parquet: %s", parquet_path)
        return pd.read_parquet(parquet_path)

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV found in CalCOFI zip")
        log.info("parsing %s", csv_names[0])
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    # Keep core oceanographic columns
    keep = [
        "Cst_Cnt", "Sta_ID", "Depth_ID", "Depthm", "T_degC", "Salnty",
        "O2ml_L", "Phspht", "NO3uM", "SiO3uM", "Lat_Dec", "Lon_Dec",
        "Date", "Quarter",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df.to_parquet(parquet_path, index=False)
    log.info("saved %s (%d rows, %d cols)", parquet_path.name, len(df), len(df.columns))
    return df


def ingest() -> pd.DataFrame:
    zip_path = download_bottle_db()
    return extract_and_parse(zip_path)


if __name__ == "__main__":
    df = ingest()
    print(df.head())
    print(f"\nCalCOFI: {len(df):,} bottle samples, date range: {df['Date'].min()} – {df['Date'].max()}")
