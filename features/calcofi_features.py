from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.spatial as spspatial


def _ckdtree(coords: np.ndarray):
    tree_cls = getattr(spspatial, "cKDTree", spspatial.KDTree)
    return tree_cls(coords)


def _profile_metrics(profile: pd.DataFrame) -> pd.Series:
    prof = profile.sort_values("depth_m")
    depth = prof["depth_m"].to_numpy(dtype=float)
    temp = prof["temp_c"].to_numpy(dtype=float)
    sal = prof["salinity"].to_numpy(dtype=float)
    chla = prof["chlorophyll_a"].to_numpy(dtype=float)

    valid_t = np.isfinite(depth) & np.isfinite(temp)
    valid_s = np.isfinite(depth) & np.isfinite(sal)

    temp_50 = np.interp(50.0, depth[valid_t], temp[valid_t]) if valid_t.sum() >= 2 else np.nan
    temp_100 = np.interp(100.0, depth[valid_t], temp[valid_t]) if valid_t.sum() >= 2 else np.nan
    sal_50 = np.interp(50.0, depth[valid_s], sal[valid_s]) if valid_s.sum() >= 2 else np.nan

    surface_mask = np.isfinite(chla) & np.isfinite(depth) & (depth <= 20)
    chla_surface = float(np.nanmean(chla[surface_mask])) if surface_mask.any() else float(np.nanmean(chla))

    depth_0_200 = np.isfinite(depth) & np.isfinite(temp) & (depth >= 0) & (depth <= 200)
    if depth_0_200.sum() >= 3:
        d = depth[depth_0_200]
        t = temp[depth_0_200]
        grad = np.gradient(t, d)
        thermocline_depth = float(d[int(np.argmax(np.abs(grad)))])
    else:
        thermocline_depth = np.nan

    return pd.Series(
        {
            "calcofi_temp_50m": temp_50,
            "calcofi_temp_100m": temp_100,
            "calcofi_salinity_50m": sal_50,
            "calcofi_chla": chla_surface,
            "calcofi_thermocline_depth": thermocline_depth,
        }
    )


def _interpolate_daily(per_station_date: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    values = [
        "calcofi_temp_50m",
        "calcofi_temp_100m",
        "calcofi_salinity_50m",
        "calcofi_chla",
        "calcofi_thermocline_depth",
    ]
    out_frames: list[pd.DataFrame] = []

    for station, grp in per_station_date.groupby("station"):
        g = grp.sort_values("date").set_index("date")
        reindexed = g.reindex(date_index)
        reindexed[values] = reindexed[values].interpolate(method="time", limit_area="inside")
        reindexed["station"] = station
        reindexed = reindexed.reset_index().rename(columns={"index": "date"})
        out_frames.append(reindexed[["station", "date", *values]])

    if not out_frames:
        return pd.DataFrame(columns=["station", "date", *values])
    return pd.concat(out_frames, ignore_index=True)


def compute_calcofi_features(
    grid: pd.DataFrame,
    calcofi_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Nearest-station daily-interpolated CalCOFI features for each cell-date."""
    cols = ["cell_id", "date", "calcofi_temp_50m", "calcofi_temp_100m", "calcofi_salinity_50m", "calcofi_chla", "calcofi_thermocline_depth"]
    if calcofi_df.empty:
        base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
        for c in cols[2:]:
            base[c] = np.nan
        return base[cols]

    work = calcofi_df.copy()
    work["date"] = pd.to_datetime(work["date"])

    station_geo = work.groupby("station", as_index=False)[["lat", "lon"]].mean()
    tree = _ckdtree(station_geo[["lat", "lon"]].to_numpy())
    _, nearest_idx = tree.query(grid[["lat", "lon"]].to_numpy(), k=1)
    cell_station = grid[["cell_id"]].copy()
    cell_station["station"] = station_geo.iloc[nearest_idx]["station"].to_numpy()

    station_profiles = (
        work.groupby(["station", "date"], as_index=False)
        .apply(_profile_metrics)
        .reset_index(drop=True)
    )

    station_daily = _interpolate_daily(station_profiles, date_index)

    base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
    base = base.merge(cell_station, on="cell_id", how="left")
    out = base.merge(station_daily, on=["station", "date"], how="left").drop(columns=["station"])
    return out[cols]
