from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial as spspatial
import xarray as xr
from scipy.ndimage import sobel

from models.climatology import attach_climatology, compute_cell_climatology


SAT_VARIABLE_MAP = {
    "sst": "sat_sst",
    "anom": "sat_sst_anomaly",
}


def _ckdtree(coords: np.ndarray):
    tree_cls = getattr(spspatial, "cKDTree", spspatial.KDTree)
    return tree_cls(coords)


def _resolve_coord_name(ds: xr.Dataset, names: list[str]) -> str:
    for name in names:
        if name in ds.coords:
            return name
    raise ValueError(f"Could not find coordinate in {names}")


def _normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    return ((lons + 180.0) % 360.0) - 180.0


def _days_since_last_true(series: pd.Series) -> pd.Series:
    n = len(series)
    idx = np.arange(n)
    last = np.where(series.to_numpy(dtype=bool), idx, np.nan).astype(float)
    last = pd.Series(last).ffill().to_numpy(dtype=float)
    out = idx.astype(float) - last
    out[np.isnan(last)] = np.nan
    return pd.Series(out, index=series.index, dtype=float)


def _lookup_nearest_cells(ds: xr.Dataset, grid: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    lat_name = _resolve_coord_name(ds, ["latitude", "lat"])
    lon_name = _resolve_coord_name(ds, ["longitude", "lon"])
    time_name = _resolve_coord_name(ds, ["time"])

    lats = ds[lat_name].to_numpy()
    lons = _normalize_longitudes(ds[lon_name].to_numpy())
    times = pd.to_datetime(ds[time_name].to_numpy())
    if len(times) == 0:
        return pd.DataFrame(columns=["date", "cell_id", "sat_sst", "sat_sst_anomaly", "sat_sst_gradient"])

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    pix_coords = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    tree = _ckdtree(pix_coords)
    _, pix_idx = tree.query(grid[["lat", "lon"]].to_numpy(), k=1)

    n_lat = len(lats)
    n_lon = len(lons)

    # OISST files store a single time slice and a single z-level.
    sst_var = ds["sst"].isel(time=0, zlev=0).to_numpy()
    anom_var = ds["anom"].isel(time=0, zlev=0).to_numpy()
    grad_lat = sobel(sst_var, axis=0, mode="nearest")
    grad_lon = sobel(sst_var, axis=1, mode="nearest")
    grad = np.hypot(grad_lat, grad_lon).reshape(n_lat * n_lon)

    sst_flat = sst_var.reshape(n_lat * n_lon)
    anom_flat = anom_var.reshape(n_lat * n_lon)

    cell_ids = grid["cell_id"].to_numpy()
    out = pd.DataFrame(
        {
            "date": np.full(len(cell_ids), target_date),
            "cell_id": cell_ids,
            "sat_sst": sst_flat[pix_idx],
            "sat_sst_anomaly": anom_flat[pix_idx],
            "sat_sst_gradient": grad[pix_idx],
        }
    )
    return out


def compute_satellite_features(
    grid: pd.DataFrame,
    raster_dir: Path,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Nearest-pixel satellite features for each cell-date."""
    nc_files = sorted(raster_dir.glob("*.nc"))
    base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
    if not nc_files:
        for col in ["sat_sst", "sat_sst_anomaly", "sat_dhw", "sat_sst_gradient", "sat_days_since_cold"]:
            base[col] = np.nan
        return base

    frames: list[pd.DataFrame] = []
    wanted_dates = set(pd.to_datetime(date_index).normalize())
    for nc_file in nc_files:
        with xr.open_dataset(nc_file) as ds:
            time_name = _resolve_coord_name(ds, ["time"])
            target_date = pd.to_datetime(ds[time_name].to_numpy()).normalize()[0]
            if target_date not in wanted_dates:
                continue
            frames.append(_lookup_nearest_cells(ds, grid, pd.Timestamp(target_date)))

    if not frames:
        for col in ["sat_sst", "sat_sst_anomaly", "sat_dhw", "sat_sst_gradient", "sat_days_since_cold"]:
            base[col] = np.nan
        return base

    sat = pd.concat(frames, ignore_index=True)
    sat = base.merge(sat, on=["cell_id", "date"], how="left")

    clim = compute_cell_climatology(sat, value_col="sat_sst", date_col="date", group_col="cell_id")
    sat = attach_climatology(sat, clim, date_col="date", group_col="cell_id", out_col="sat_climatology")
    cold = sat["sat_sst"] < (sat["sat_climatology"] - 0.5)
    sat["sat_days_since_cold"] = (
        sat.assign(_cold=cold)
        .sort_values(["cell_id", "date"])
        .groupby("cell_id", group_keys=False)["_cold"]
        .transform(_days_since_last_true)
    )
    sat = sat.drop(columns=["sat_climatology"], errors="ignore")

    # Degree heating weeks derived from positive anomaly accumulation.
    sat["sat_dhw"] = (
        sat.sort_values(["cell_id", "date"])
        .groupby("cell_id", group_keys=False)["sat_sst_anomaly"]
        .transform(lambda s: s.clip(lower=0).rolling(84, min_periods=1).sum() / 7.0)
    )

    return sat[
        ["cell_id", "date", "sat_sst", "sat_sst_anomaly", "sat_dhw", "sat_sst_gradient", "sat_days_since_cold"]
    ]
