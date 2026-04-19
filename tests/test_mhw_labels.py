from __future__ import annotations

import numpy as np
import pandas as pd

from models.climatology import compute_hobday_climatology
from models.mhw_labels import add_future_targets, build_training_labels


def _synthetic_socal_sst() -> pd.DataFrame:
    dates = pd.date_range("2014-01-01", "2016-12-31", freq="D")
    day = dates.dayofyear.to_numpy()

    seasonal = 16.0 + 1.5 * np.sin(2 * np.pi * day / 366.0)
    sst = seasonal.copy()

    blob_window = (dates >= pd.Timestamp("2015-07-01")) & (dates <= pd.Timestamp("2015-09-30"))
    sst[blob_window] = sst[blob_window] + 2.8

    return pd.DataFrame({"cell_id": 101, "date": dates, "sat_sst": sst})


def test_hobday_climatology_schema_and_coverage() -> None:
    sst = _synthetic_socal_sst()
    clim = compute_hobday_climatology(
        sst,
        date_col="date",
        sst_col="sat_sst",
        group_col="cell_id",
        baseline_start="1983-01-01",
        baseline_end="2012-12-31",
    )

    assert list(clim.columns) == ["cell_id", "day_of_year", "clim_mean", "clim_p90"]
    assert clim["day_of_year"].min() == 1
    assert clim["day_of_year"].max() == 366
    assert len(clim) == 366
    assert (clim["clim_p90"] >= clim["clim_mean"]).mean() > 0.95


def test_blob_window_labels_fire_and_future_targets() -> None:
    sst = _synthetic_socal_sst()
    clim = compute_hobday_climatology(
        sst,
        date_col="date",
        sst_col="sat_sst",
        group_col="cell_id",
        baseline_start="2014-01-01",
        baseline_end="2014-12-31",
    )

    labeled = build_training_labels(sst, clim, cell_col="cell_id", date_col="date", sst_col="sat_sst")

    active = labeled[(labeled["date"] >= "2015-07-10") & (labeled["date"] <= "2015-09-20")]
    quiet = labeled[(labeled["date"] >= "2015-03-01") & (labeled["date"] <= "2015-05-31")]

    assert active["mhw_status"].mean() > 0.8
    assert quiet["mhw_status"].mean() < 0.3
    assert (active["mhw_status"].mean() - quiet["mhw_status"].mean()) > 0.55
    assert (active["mhw_category"] != "None").mean() > 0.8

    shifted = add_future_targets(labeled, status_col="mhw_status", cell_col="cell_id", date_col="date")
    expected = shifted.sort_values(["cell_id", "date"]).groupby("cell_id")["mhw_status"].shift(-7)

    pd.testing.assert_series_equal(
        shifted.sort_values(["cell_id", "date"])["y_7d"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
