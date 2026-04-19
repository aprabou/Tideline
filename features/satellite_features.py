from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial as spspatial
import xarray as xr
from scipy.ndimage import sobel

from models.climatology import attach_climatology, compute_cell_climatology


SAT_VARIABLE_MAP = {
    "CRW_SST": "sat_sst",
    "CRW_SSTANOMALY": "sat_sst_anomaly",
    "CRW_DHW": "sat_dhw",
}


def _ckdtree(coords: np.ndarray):
    tree_cls = getattr(spspatial, "cKDTree", spspatial.KDTree)
    return tree_cls(coords)


def _resolve_coord_name(ds: xr.Dataset, names: list[str]) -> str:
    for name in names:
        if name in ds.coords:
            return name
    raise ValueError(f"Could not find coordinate in {names}")


def _days_since_last_true(series: pd.Series) -> pd.Series:
    n = len(series)
    idx = np.arange(n)
    last = np.where(series.to_numpy(dtype=bool), idx, -10_000_000)
    last = np.maximum.accumulate(last)
    out = idx - last
    out[last < 0] = np.nan
    return pd.Series(out, index=series.index, dtype=float)


def _lookup_nearest_cells(ds: xr.Dataset, grid: pd.DataFrame) -> pd.DataFrame:
    lat_name = _resolve_coord_name(ds, ["latitude", "lat"])
    lon_name = _resolve_coord_name(ds, ["longitude", "lon"])
    time_name = _resolve_coord_name(ds, ["time"])

    lats = ds[lat_name].to_numpy()
    lons = ds[lon_name].to_numpy()
    times = pd.to_datetime(ds[time_name].to_numpy())

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    pix_coords = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    tree = _ckdtree(pix_coords)
    _, pix_idx = tree.query(grid[["lat", "lon"]].to_numpy(), k=1)

    n_time = len(times)
    n_lat = len(lats)
    n_lon = len(lons)

    lookup: dict[str, np.ndarray] = {}
    for src, out_name in SAT_VARIABLE_MAP.items():
        arr = ds[src].to_numpy().reshape(n_time, n_lat * n_lon)
        lookup[out_name] = arr[:, pix_idx]

    sst_3d = ds["CRW_SST"].to_numpy()
    grad_lat = sobel(sst_3d, axis=1, mode="nearest")
    grad_lon = sobel(sst_3d, axis=2, mode="nearest")
    grad = np.hypot(grad_lat, grad_lon).reshape(n_time, n_lat * n_lon)
    lookup["sat_sst_gradient"] = grad[:, pix_idx]

    cell_ids = grid["cell_id"].to_numpy()
    out = pd.DataFrame(
        {
            "date": np.repeat(times, len(cell_ids)),
            "cell_id": np.tile(cell_ids, n_time),
            "sat_sst": lookup["sat_sst"].reshape(-1),
            "sat_sst_anomaly": lookup["sat_sst_anomaly"].reshape(-1),
            "sat_dhw": lookup["sat_dhw"].reshape(-1),
            "sat_sst_gradient": lookup["sat_sst_gradient"].reshape(-1),
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

    ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4")
    try:
        sat = _lookup_nearest_cells(ds, grid)
    finally:
        ds.close()

    sat = sat[sat["date"].isin(date_index)].copy()
    sat = base.merge(sat, on=["cell_id", "date"], how="left")

    clim = compute_cell_climatology(sat, value_col="sat_sst", date_col="date", group_col="cell_id")
    sat = attach_climatology(sat, clim, date_col="date", group_col="cell_id", out_col="sat_climatology")
    cold = sat["sat_sst"] < (sat["sat_climatology"] - 0.5)
    sat["sat_days_since_cold"] = (
        sat.assign(_cold=cold)
        .sort_values(["cell_id", "date"])
        .groupby("cell_id", group_keys=False)["_cold"]
        .apply(_days_since_last_true)
        .reset_index(level=0, drop=True)
    )
    sat = sat.drop(columns=["sat_climatology", "_cold"])

    return sat[
        ["cell_id", "date", "sat_sst", "sat_sst_anomaly", "sat_dhw", "sat_sst_gradient", "sat_days_since_cold"]
    ]
