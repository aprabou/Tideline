from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.buoy_features import compute_buoy_features
from features.calcofi_features import compute_calcofi_features
from features.satellite_features import compute_satellite_features
from features.unified_grid import get_grid

BUOY_PATH = Path("data/raw/ingestion/noaa_buoys_sst.parquet")
CALCOFI_PATH = Path("data/raw/ingestion/calcofi_bottle.parquet")
SAT_RASTER_DIR = Path("sst_rasters")
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


def validate_feature_table(df: pd.DataFrame, max_missing_ratio: float = 0.3) -> None:
    if df.duplicated(["cell_id", "date"]).any():
        raise ValueError("Feature table contains duplicated (cell_id, date) rows")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_ratio = df[numeric_cols].isna().mean()
    bad = missing_ratio[missing_ratio >= max_missing_ratio]
    if not bad.empty:
        msg = ", ".join(f"{col}={ratio:.2%}" for col, ratio in bad.items())
        raise ValueError(f"Numeric columns exceed missing-value threshold: {msg}")


def build_feature_table(
    buoy_path: Path = BUOY_PATH,
    calcofi_path: Path = CALCOFI_PATH,
    sat_raster_dir: Path = SAT_RASTER_DIR,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    """Build unified daily feature table for 2015-2023."""
    grid = get_grid()
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    buoy_daily = _safe_read_parquet(buoy_path, ["station_id", "lat", "lon", "date", "sst"])
    calcofi = _safe_read_parquet(
        calcofi_path,
        ["date", "station", "lat", "lon", "depth_m", "temp_c", "salinity", "chlorophyll_a"],
    )

    base = _build_base_table(grid, dates)
    buoy_feats = compute_buoy_features(grid, buoy_daily, dates)
    calcofi_feats = compute_calcofi_features(grid, calcofi, dates)
    sat_feats = compute_satellite_features(grid, sat_raster_dir, dates)

    out = (
        base.merge(buoy_feats, on=["cell_id", "date"], how="left")
        .merge(calcofi_feats, on=["cell_id", "date"], how="left")
        .merge(sat_feats, on=["cell_id", "date"], how="left")
        .merge(grid, on="cell_id", how="left")
    )

    out["day_of_year"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year

    out["mhw_status"] = (out["sat_sst_anomaly"] >= 1.0).astype(float)
    out = add_target_lags(out, target_col="mhw_status")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def main() -> None:
    table = build_feature_table()
    validate_feature_table(table)


if __name__ == "__main__":
    main()
