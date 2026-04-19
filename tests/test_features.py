from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.build_feature_table import build_feature_table
from features.unified_grid import get_grid


def test_unified_grid_shape_and_schema() -> None:
    grid = get_grid()
    assert list(grid.columns) == ["cell_id", "lat", "lon"]
    assert 1000 <= len(grid) <= 1500
    assert grid["cell_id"].is_unique


def test_feature_table_quality_and_lag_shift(monkeypatch, tmp_path: Path) -> None:
    import features.build_feature_table as bft

    def _fake_grid() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_id": [0, 1],
                "lat": [33.0, 33.2],
                "lon": [-119.5, -119.3],
            }
        )

    def _fake_buoy(grid: pd.DataFrame, buoy_daily: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
        n = len(base)
        base["buoy_sst_idw"] = np.linspace(15.0, 18.0, n)
        base["buoy_sst_7d_mean"] = base.groupby("cell_id")["buoy_sst_idw"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        base["buoy_sst_30d_mean"] = base.groupby("cell_id")["buoy_sst_idw"].transform(lambda s: s.rolling(30, min_periods=1).mean())
        base["buoy_anomaly"] = base["buoy_sst_idw"] - 16.0
        return base

    def _fake_calcofi(grid: pd.DataFrame, calcofi_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
        n = len(base)
        base["calcofi_temp_50m"] = 12.0
        base["calcofi_temp_100m"] = 10.5
        base["calcofi_salinity_50m"] = 33.1
        base["calcofi_chla"] = np.where(np.arange(n) % 20 == 0, np.nan, 0.4)
        base["calcofi_thermocline_depth"] = 45.0
        return base

    def _fake_sat(grid: pd.DataFrame, raster_dir: Path, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        base = pd.MultiIndex.from_product([grid["cell_id"], date_index], names=["cell_id", "date"]).to_frame(index=False)
        n = len(base)
        base["sat_sst"] = 18.0
        base["sat_sst_anomaly"] = np.where(np.arange(n) % 2 == 0, 1.2, 0.2)
        base["sat_dhw"] = 2.0
        base["sat_sst_gradient"] = 0.1
        base["sat_days_since_cold"] = np.where(np.arange(n) % 15 == 0, np.nan, 5.0)
        return base

    monkeypatch.setattr(bft, "get_grid", _fake_grid)
    monkeypatch.setattr(bft, "compute_buoy_features", _fake_buoy)
    monkeypatch.setattr(bft, "compute_calcofi_features", _fake_calcofi)
    monkeypatch.setattr(bft, "compute_satellite_features", _fake_sat)
    monkeypatch.setattr(bft, "START_DATE", "2015-01-01")
    monkeypatch.setattr(bft, "END_DATE", "2015-01-31")

    out = build_feature_table(out_path=tmp_path / "features.parquet")

    assert not out.duplicated(["cell_id", "date"]).any()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    miss = out[numeric_cols].isna().mean()
    assert (miss < 0.3).all()

    expected_lag1 = out.sort_values(["cell_id", "date"]).groupby("cell_id")["mhw_status"].shift(1)
    expected_lag3 = out.sort_values(["cell_id", "date"]).groupby("cell_id")["mhw_status"].shift(3)
    expected_lag7 = out.sort_values(["cell_id", "date"]).groupby("cell_id")["mhw_status"].shift(7)

    pd.testing.assert_series_equal(
        out.sort_values(["cell_id", "date"])["mhw_status_lag_1d"].reset_index(drop=True),
        expected_lag1.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out.sort_values(["cell_id", "date"])["mhw_status_lag_3d"].reset_index(drop=True),
        expected_lag3.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out.sort_values(["cell_id", "date"])["mhw_status_lag_7d"].reset_index(drop=True),
        expected_lag7.reset_index(drop=True),
        check_names=False,
    )
