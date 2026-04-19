from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.spatial as spspatial

from models.climatology import attach_climatology, compute_cell_climatology


def _ckdtree(coords: np.ndarray):
    tree_cls = getattr(spspatial, "cKDTree", spspatial.KDTree)
    return tree_cls(coords)


def _idw_from_nearest(
    grid: pd.DataFrame,
    buoy_daily: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    stations = (
        buoy_daily[["station_id", "lat", "lon"]]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )
    if stations.empty:
        return pd.DataFrame(columns=["cell_id", "date", "buoy_sst_idw"])

    station_coords = stations[["lat", "lon"]].to_numpy()
    cell_coords = grid[["lat", "lon"]].to_numpy()
    tree = _ckdtree(station_coords)
    dist, idx = tree.query(cell_coords, k=min(k, len(stations)))
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]

    station_ids = stations["station_id"].to_numpy()
    nearest_station_ids = station_ids[idx]

    pivot = buoy_daily.pivot_table(index="date", columns="station_id", values="sst", aggfunc="mean")
    pivot = pivot.sort_index()

    weighted_sum = pd.DataFrame(0.0, index=pivot.index, columns=grid["cell_id"].to_numpy())
    weights_sum = pd.DataFrame(0.0, index=pivot.index, columns=grid["cell_id"].to_numpy())

    for rank in range(nearest_station_ids.shape[1]):
        sid_for_cells = nearest_station_ids[:, rank]
        values = pivot.reindex(columns=sid_for_cells)
        values.columns = grid["cell_id"].to_numpy()

        w = 1.0 / np.maximum(dist[:, rank], 1e-6)
        w_row = pd.DataFrame(np.tile(w, (len(pivot.index), 1)), index=pivot.index, columns=values.columns)
        valid = values.notna()

        weighted_sum = weighted_sum + values.fillna(0.0) * w_row
        weights_sum = weights_sum + w_row.where(valid, 0.0)

    idw = (weighted_sum / weights_sum).replace([np.inf, -np.inf], np.nan)
    out = (
        idw.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="cell_id", value_name="buoy_sst_idw")
    )
    return out[["cell_id", "date", "buoy_sst_idw"]]


def compute_buoy_features(
    grid: pd.DataFrame,
    buoy_daily: pd.DataFrame,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute IDW buoy SST and rolling/anomaly features for each cell-date."""
    if buoy_daily.empty:
        base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
        base["buoy_sst_idw"] = np.nan
    else:
        work = buoy_daily.copy()
        work["date"] = pd.to_datetime(work["date"])
        base = _idw_from_nearest(grid, work, k=3)

    full_index = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
    df = full_index.merge(base, on=["cell_id", "date"], how="left")
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("cell_id", group_keys=False)
    df["buoy_sst_7d_mean"] = grouped["buoy_sst_idw"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["buoy_sst_30d_mean"] = grouped["buoy_sst_idw"].transform(lambda s: s.rolling(30, min_periods=10).mean())

    clim = compute_cell_climatology(df, value_col="buoy_sst_idw", date_col="date", group_col="cell_id")
    with_clim = attach_climatology(df, clim, date_col="date", group_col="cell_id", out_col="buoy_climatology")
    with_clim["buoy_anomaly"] = with_clim["buoy_sst_idw"] - with_clim["buoy_climatology"]

    return with_clim[
        ["cell_id", "date", "buoy_sst_idw", "buoy_sst_7d_mean", "buoy_sst_30d_mean", "buoy_anomaly"]
    ]
