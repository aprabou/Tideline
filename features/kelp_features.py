"""
kelp_features.py — Join kelp canopy and density observations onto the unified grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial as spspatial

SCRIPPS_KELP_PATH = Path("data/raw/ingestion/scripps_kelp_canopy.parquet")
SDCITY_KELP_PATH = Path("data/raw/ingestion/sdcity_kelp_density.parquet")


def _ckdtree(coords: np.ndarray):
    tree_cls = getattr(spspatial, "cKDTree", spspatial.KDTree)
    return tree_cls(coords)


def _nearest_cell(obs_lat: float, obs_lon: float, grid: pd.DataFrame) -> int:
    tree = _ckdtree(grid[["lat", "lon"]].to_numpy())
    _, idx = tree.query([[obs_lat, obs_lon]], k=1)
    return int(grid.iloc[int(idx[0])]["cell_id"])


def compute_kelp_features(
    grid: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    scripps_path: Path = SCRIPPS_KELP_PATH,
    sdcity_path: Path = SDCITY_KELP_PATH,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        cell_id, date, kelp_canopy_extent, kelp_canopy_anomaly,
        kelp_density, kelp_recovery_index

    Observations are forward-filled within each cell to fill daily gaps.
    """
    base = (
        pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"])
        .to_frame(index=False)
    )
    nan_cols = ["kelp_canopy_extent", "kelp_canopy_anomaly", "kelp_density", "kelp_recovery_index"]

    # --- Scripps quarterly canopy ---
    if scripps_path.exists():
        kelp_q = pd.read_parquet(scripps_path)
        kelp_q["date"] = pd.to_datetime(kelp_q["date"])

        # Map each observation site to nearest grid cell
        kelp_q["cell_id"] = kelp_q.apply(
            lambda r: _nearest_cell(r["lat"], r["lon"], grid), axis=1
        )

        # Daily cell-level aggregate (mean of co-located sites per cell-date)
        kelp_q_agg = (
            kelp_q.groupby(["cell_id", "date"])["canopy_extent_km2"]
            .mean()
            .reset_index()
            .rename(columns={"canopy_extent_km2": "kelp_canopy_extent"})
        )

        # 10-year rolling anomaly
        kelp_q_agg = kelp_q_agg.sort_values(["cell_id", "date"])
        kelp_q_agg["_rolling_mean"] = (
            kelp_q_agg.groupby("cell_id")["kelp_canopy_extent"]
            .transform(lambda s: s.rolling(40, min_periods=1).mean())
        )
        kelp_q_agg["kelp_canopy_anomaly"] = (
            kelp_q_agg["kelp_canopy_extent"] - kelp_q_agg["_rolling_mean"]
        )
        kelp_q_agg = kelp_q_agg.drop(columns=["_rolling_mean"])

        base = base.merge(kelp_q_agg, on=["cell_id", "date"], how="left")
        # Forward-fill to fill daily gaps from quarterly observations
        base = base.sort_values(["cell_id", "date"])
        for col in ["kelp_canopy_extent", "kelp_canopy_anomaly"]:
            base[col] = base.groupby("cell_id")[col].transform(lambda s: s.ffill())
    else:
        base["kelp_canopy_extent"] = np.nan
        base["kelp_canopy_anomaly"] = np.nan

    # --- SD City monthly density ---
    if sdcity_path.exists():
        kelp_m = pd.read_parquet(sdcity_path)
        kelp_m["date"] = pd.to_datetime(kelp_m["date"])

        kelp_m["cell_id"] = kelp_m.apply(
            lambda r: _nearest_cell(r["lat"], r["lon"], grid), axis=1
        )

        kelp_m_agg = (
            kelp_m.groupby(["cell_id", "date"])
            .agg(
                kelp_density=("frond_density_per_m2", "mean"),
                _substrate=("substrate_coverage_pct", "mean"),
            )
            .reset_index()
        )

        # Recovery index: normalised frond density relative to pre-Blob baseline (2014-01)
        baselines = (
            kelp_m_agg[kelp_m_agg["date"] <= "2014-09-01"]
            .groupby("cell_id")["kelp_density"]
            .mean()
            .rename("_baseline")
        )
        kelp_m_agg = kelp_m_agg.merge(baselines, on="cell_id", how="left")
        kelp_m_agg["kelp_recovery_index"] = (
            kelp_m_agg["kelp_density"] / kelp_m_agg["_baseline"].clip(lower=0.01)
        ).clip(upper=1.5)
        kelp_m_agg = kelp_m_agg.drop(columns=["_baseline", "_substrate"])

        base = base.merge(kelp_m_agg, on=["cell_id", "date"], how="left")
        base = base.sort_values(["cell_id", "date"])
        for col in ["kelp_density", "kelp_recovery_index"]:
            base[col] = base.groupby("cell_id")[col].transform(lambda s: s.ffill())
    else:
        base["kelp_density"] = np.nan
        base["kelp_recovery_index"] = np.nan

    return base[["cell_id", "date"] + nan_cols]
