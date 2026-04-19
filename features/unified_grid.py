from __future__ import annotations

import numpy as np
import pandas as pd

LAT_MIN = 32.0
LAT_MAX = 35.0
LON_MIN = -121.0
LON_MAX = -117.0
GRID_RES_KM = 10.0


def _step_lat_deg(km: float) -> float:
    return km / 111.0


def _step_lon_deg(km: float, reference_lat: float) -> float:
    return km / (111.0 * np.cos(np.deg2rad(reference_lat)))


def get_grid() -> pd.DataFrame:
    """Build a ~10km target grid over the Southern California Bight."""
    lat_step = _step_lat_deg(GRID_RES_KM)
    lon_step = _step_lon_deg(GRID_RES_KM, reference_lat=(LAT_MIN + LAT_MAX) / 2)

    lat_values = np.arange(LAT_MIN, LAT_MAX + lat_step / 2, lat_step)
    lon_values = np.arange(LON_MIN, LON_MAX + lon_step / 2, lon_step)

    lon_mesh, lat_mesh = np.meshgrid(lon_values, lat_values)
    flat_lat = lat_mesh.ravel()
    flat_lon = lon_mesh.ravel()

    grid = pd.DataFrame({"lat": flat_lat, "lon": flat_lon})
    grid["cell_id"] = np.arange(len(grid), dtype=np.int64)
    return grid[["cell_id", "lat", "lon"]]
