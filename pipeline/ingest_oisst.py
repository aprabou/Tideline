"""
ingest_oisst.py — Download NOAA OISST v2.1 daily SST (NetCDF) to data/raw/.

Dataset: NOAA 1/4° Daily Optimum Interpolation SST (OISST) v2.1
Source:  https://www.ncei.noaa.gov/products/optimum-interpolation-sst
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import requests

RAW_DIR = Path("data/raw/oisst")
BASE_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr/{year}{month:02d}/"
    "oisst-avhrr-v02r01.{year}{month:02d}{day:02d}.nc"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_day(target: date, out_dir: Path = RAW_DIR) -> Path:
    """Download a single day of OISST and return the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(year=target.year, month=target.month, day=target.day)
    dest = out_dir / Path(url).name
    if dest.exists():
        log.info("already cached: %s", dest.name)
        return dest
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def ingest(days: int = 30) -> list[Path]:
    """Download the last *days* days of OISST."""
    today = date.today()
    # OISST is typically 2-3 days behind real-time
    end = today - timedelta(days=2)
    start = end - timedelta(days=days - 1)
    paths = []
    for offset in range(days):
        target = start + timedelta(days=offset)
        try:
            paths.append(download_day(target))
        except requests.HTTPError as exc:
            log.warning("skipping %s: %s", target, exc)
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NOAA OISST data")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch")
    args = parser.parse_args()
    files = ingest(days=args.days)
    print(f"Downloaded {len(files)} file(s) to {RAW_DIR}")
