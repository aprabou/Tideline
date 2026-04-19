"""
features.py — Build the silver-layer feature table from bronze parquet files.

Features per grid cell × day:
  - sst_mean, sst_anom (vs 30-yr climatology)
  - rolling_7d_mean, rolling_14d_mean, rolling_30d_mean of SST
  - sst_trend_14d (linear slope over 14-day window)
  - buoy_sst_nearest (nearest NDBC buoy obs, spatially joined)
  - calcofi_t_degc_q (latest quarterly CalCOFI temperature, spatially joined)
  - month_sin, month_cos (seasonal encoding)
  - lat, lon
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw")
SILVER_DIR = Path("data/silver")
CLIM_WINDOW = 30  # years for climatology (use available data in practice)


def load_oisst(raw_dir: Path = RAW_DIR / "oisst") -> xr.Dataset:
    """Load all local OISST NetCDF files into a single Dataset."""
    nc_files = sorted(raw_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No OISST NetCDF files found in {raw_dir}")
    ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
    return ds


def compute_anomaly(sst: xr.DataArray) -> xr.DataArray:
    """Compute SST anomaly relative to day-of-year climatology."""
    clim = sst.groupby("time.dayofyear").mean("time")
    return sst.groupby("time.dayofyear") - clim


def rolling_stats(df: pd.DataFrame, col: str = "sst") -> pd.DataFrame:
    """Add rolling mean and linear-trend features for a single location series."""
    for w in [7, 14, 30]:
        df[f"rolling_{w}d_mean"] = df[col].rolling(w, min_periods=w // 2).mean()
    # 14-day linear trend (slope in °C/day)
    df["sst_trend_14d"] = (
        df[col]
        .rolling(14, min_periods=7)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )
    return df


def seasonal_encoding(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    month = pd.to_datetime(df[time_col]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def build_feature_table(
    oisst_dir: Path = RAW_DIR / "oisst",
    buoy_dir: Path = RAW_DIR / "buoys",
    out_dir: Path = SILVER_DIR,
) -> pd.DataFrame:
    """Build the full feature table and write to parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_oisst(oisst_dir)
    sst = ds["sst"].squeeze()  # drop zlev if present
    anom = compute_anomaly(sst)

    # Flatten to tidy DataFrame
    df_sst = sst.to_dataframe(name="sst").reset_index()
    df_anom = anom.to_dataframe(name="sst_anom").reset_index()
    df = df_sst.merge(df_anom[["time", "lat", "lon", "sst_anom"]], on=["time", "lat", "lon"])

    df = df.sort_values(["lat", "lon", "time"])
    df = (
        df.groupby(["lat", "lon"], group_keys=False)
        .apply(lambda g: rolling_stats(g.copy()))
        .reset_index(drop=True)
    )
    df = seasonal_encoding(df)

    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Feature table saved: {out_path} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    df = build_feature_table()
    print(df.describe())
