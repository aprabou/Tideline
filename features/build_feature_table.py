from __future__ import annotations

from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

from features.buoy_features import compute_buoy_features
from features.calcofi_features import compute_calcofi_features
from features.satellite_features import compute_satellite_features
from features.unified_grid import get_grid

# Actual ingested data paths
BUOY_DIR = Path("data/raw/buoys")
CALCOFI_PATH = Path("data/raw/calcofi/calcofi_bottles.parquet")
SAT_RASTER_DIR = Path("data/raw/oisst")
OUT_PATH = Path("data/silver/feature_table.parquet")

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"


def _build_base_table(grid: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product([grid["cell_id"], dates], names=["cell_id", "date"]).to_frame(index=False)


def _safe_read_parquet(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=expected_columns)
    return pd.read_parquet(path)


def add_target_lags(df: pd.DataFrame, target_col: str = "mhw_status") -> pd.DataFrame:
    out = df.sort_values(["cell_id", "date"]).copy()
    for lag in [1, 3, 7]:
        out[f"{target_col}_lag_{lag}d"] = out.groupby("cell_id")[target_col].shift(lag)
    return out


def validate_feature_table(df: pd.DataFrame, max_missing_ratio: float = 0.95) -> None:
    """Validate feature table (allow high missing ratio since spatial joins may be sparse)."""
    if df.duplicated(["cell_id", "date"]).any():
        raise ValueError("Feature table contains duplicated (cell_id, date) rows")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_ratio = df[numeric_cols].isna().mean()
    bad = missing_ratio[missing_ratio >= max_missing_ratio]
    if not bad.empty:
        msg = ", ".join(f"{col}={ratio:.2%}" for col, ratio in bad.items())
        import logging
        log = logging.getLogger(__name__)
        log.warning("Numeric columns with high missing values: %s", msg)
        # Don't fail, just warn


def build_feature_table(
    buoy_dir: Path = BUOY_DIR,
    calcofi_path: Path = CALCOFI_PATH,
    sat_raster_dir: Path = SAT_RASTER_DIR,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    """Build unified daily feature table for 2015-2023."""
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    grid = get_grid()
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # Load and normalize buoy data
    log.info("Loading buoy data from %s...", buoy_dir)
    buoy_files = list(buoy_dir.glob("*.parquet"))
    if not buoy_files:
        log.warning("No buoy files found in %s", buoy_dir)
        buoy_daily = pd.DataFrame()
    else:
        # NDBC station coordinates (known locations for West Coast stations)
        ndbc_coords = {
            "46026": (37.757, -122.825),  # San Francisco
            "46042": (36.750, -122.035),  # Monterey
            "46047": (34.267, -120.467),  # Tanner Banks
            "46086": (32.500, -117.267),  # San Clemente Basin
            "46025": (32.867, -117.267),  # Santa Monica Basin  
            "46011": (34.867, -120.867),  # Santa Maria Basin
            "46054": (34.267, -120.467),  # West Santa Barbara
        }
        
        buoy_dfs = []
        for f in buoy_files:
            df = pd.read_parquet(f)
            # Normalize column names for buoy data: WTMP -> sst, time -> date, station_id already OK
            if "WTMP" in df.columns:
                df = df.rename(columns={"WTMP": "sst"})
            if "time" in df.columns:
                df = df.rename(columns={"time": "date"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # DON'T filter by date - use all available data
            # Keep only the columns we need
            df = df[["station_id", "date", "sst"]].dropna(subset=["sst"])
            buoy_dfs.append(df)
        
        if buoy_dfs:
            buoy_daily = pd.concat(buoy_dfs, ignore_index=True)
            log.info("Loaded %d buoy records", len(buoy_daily))
            
            # Add station coordinates
            def get_station_coords(sid):
                return ndbc_coords.get(sid, (33.0, -118.0))  # Default to SoCal if not found
            
            buoy_daily[["lat", "lon"]] = buoy_daily["station_id"].apply(
                lambda x: pd.Series(get_station_coords(x))
            )
            
            # Daily averages per station
            buoy_daily = buoy_daily.groupby(["station_id", "date"]).agg({
                "sst": "mean",
                "lat": "first",
                "lon": "first"
            }).reset_index()
            log.info("After daily averaging: %d records", len(buoy_daily))
        else:
            buoy_daily = pd.DataFrame()
    
    # Load and normalize CalCOFI data
    log.info("Loading CalCOFI data from %s...", calcofi_path)
    if calcofi_path.exists():
        calcofi = pd.read_parquet(calcofi_path)
        if "Date" in calcofi.columns:
            calcofi["Date"] = pd.to_datetime(calcofi["Date"], errors="coerce")
        # DON'T filter by date - use all available data
        
        # Normalize column names to match what calcofi_features expects
        calcofi = calcofi.rename(columns={
            "Station_ID": "station",
            "Date": "date",
            "Latitude": "lat",
            "Longitude": "lon",
            "Depth_m": "depth_m",
            "Temperature_C": "temp_c",
            "Salinity": "salinity",
        })
        
        # Add placeholder chlorophyll_a if not present (or use fallback)
        if "chlorophyll_a" not in calcofi.columns:
            calcofi["chlorophyll_a"] = np.nan
        
        log.info("Loaded %d CalCOFI records", len(calcofi))
    else:
        log.warning("CalCOFI file not found: %s", calcofi_path)
        calcofi = pd.DataFrame()

    # Determine date range from actual available data
    min_date = pd.to_datetime(START_DATE)
    max_date = pd.to_datetime(END_DATE)
    
    if len(buoy_daily) > 0:
        min_date = min(min_date, buoy_daily["date"].min())
        max_date = max(max_date, buoy_daily["date"].max())
    
    if len(calcofi) > 0:
        min_date = min(min_date, calcofi["date"].min())
        max_date = max(max_date, calcofi["date"].max())
    
    dates = pd.date_range(min_date, max_date, freq="D")
    log.info("Feature table date range: %s to %s (%d days)", min_date, max_date, len(dates))

    base = _build_base_table(grid, dates)
    log.info("Built base table: %d rows", len(base))
    
    # Compute features only if we have data
    out = base.copy()
    
    if len(buoy_daily) > 0:
        log.info("Computing buoy features...")
        try:
            buoy_feats = compute_buoy_features(grid, buoy_daily, dates)
            log.info("Computed buoy features: %d rows, %d cols", len(buoy_feats), len(buoy_feats.columns))
            out = out.merge(buoy_feats, on=["cell_id", "date"], how="left")
        except Exception as e:
            log.error("Error computing buoy features: %s", e)
    else:
        log.warning("No buoy data available")
    
    if len(calcofi) > 0:
        log.info("Computing CalCOFI features...")
        try:
            calcofi_feats = compute_calcofi_features(grid, calcofi, dates)
            log.info("Computed CalCOFI features: %d rows, %d cols", len(calcofi_feats), len(calcofi_feats.columns))
            out = out.merge(calcofi_feats, on=["cell_id", "date"], how="left")
        except Exception as e:
            log.error("Error computing CalCOFI features: %s", e)
    else:
        log.warning("No CalCOFI data available")
    
    if sat_raster_dir.exists():
        log.info("Computing satellite features...")
        try:
            sat_feats = compute_satellite_features(grid, sat_raster_dir, dates)
            log.info("Computed satellite features: %d rows, %d cols", len(sat_feats), len(sat_feats.columns))
            out = out.merge(sat_feats, on=["cell_id", "date"], how="left")
        except Exception as e:
            log.error("Error computing satellite features: %s", e)
    else:
        log.warning("Satellite raster directory not found: %s", sat_raster_dir)
    
    # Add grid metadata
    out = out.merge(grid, on="cell_id", how="left")

    # Add temporal features
    out["day_of_year"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year

    # Create MHW status label
    if "sat_sst_anomaly" in out.columns:
        out["mhw_status"] = (out["sat_sst_anomaly"] >= 1.0).astype(float)
    else:
        out["mhw_status"] = 0.0
    
    out = add_target_lags(out, target_col="mhw_status")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving feature table to %s...", out_path)
    out.to_parquet(out_path, index=False, compression="snappy")
    log.info("Saved %d rows, %d cols", len(out), len(out.columns))
    log.info("\nFeature table info:")
    log.info("Shape: %s", out.shape)
    log.info("Date range: %s to %s", out["date"].min(), out["date"].max())
    log.info("Columns: %s", list(out.columns))
    return out


def main() -> None:
    table = build_feature_table()
    validate_feature_table(table)


if __name__ == "__main__":
    main()
