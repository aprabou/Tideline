from __future__ import annotations

import pandas as pd
import xarray as xr

from ingestion.calcofi import transform_bottle_dataframe
from ingestion.noaa_buoys import build_station_daily_sst, parse_stdmet_text
from ingestion.satellite_sst import subset_year_dataset


def test_noaa_parse_and_daily_schema(monkeypatch) -> None:
    sample_text = "\n".join(
        [
            "#YYYY MM DD hh mm WTMP",
            "#yr mo dy hr mn degC",
            "2014 01 01 00 00 15.0",
            "2014 01 01 12 00 17.0",
            "2014 01 02 00 00 99.0",
        ]
    )

    parsed = parse_stdmet_text(sample_text)
    assert ["date", "sst"] == list(parsed.columns)

    def _fake_fetch(station_id: str, year: int, timeout: int = 60) -> str | None:
        if station_id == "46025" and year == 2014:
            return sample_text
        return None

    monkeypatch.setattr("ingestion.noaa_buoys.fetch_historical_text", _fake_fetch)
    result = build_station_daily_sst("46025", range(2014, 2015))

    assert list(result.columns) == ["station_id", "lat", "lon", "date", "sst"]
    assert len(result) == 2
    assert result["station_id"].iloc[0] == "46025"
    assert result["sst"].iloc[0] == 16.0
    assert pd.isna(result["sst"].iloc[1])


def test_calcofi_schema_small_sample() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2015-07-01", "2012-01-01", "2016-01-01"],
            "St_Line": [90, 90, 90],
            "St_Station": [30, 31, 32],
            "Cst_Cnt": [1, 1, 1],
            "Depthm": [10.0, 10.0, 20.0],
            "T_degC": [16.0, 14.0, 15.5],
            "Salnty": [33.2, 33.0, 33.1],
            "ChlorA": [0.4, 0.3, 0.5],
            "Lat_Dec": [33.5, 33.5, 40.0],
            "Lon_Dec": [-119.0, -119.0, -119.0],
        }
    )

    out = transform_bottle_dataframe(df)
    assert list(out.columns) == ["date", "station", "lat", "lon", "depth_m", "temp_c", "salinity", "chlorophyll_a"]
    assert len(out) == 1
    assert out["station"].iloc[0] == "90-30"


def test_satellite_subset_schema_small_sample() -> None:
    ds = xr.Dataset(
        data_vars={
            "CRW_SST": (("time", "latitude", "longitude"), [[[20.0]]]),
            "CRW_SSTANOMALY": (("time", "latitude", "longitude"), [[[0.5]]]),
            "CRW_DHW": (("time", "latitude", "longitude"), [[[1.0]]]),
        },
        coords={
            "time": pd.to_datetime(["2014-06-01"]),
            "latitude": [33.0],
            "longitude": [-120.0],
        },
    )

    out = subset_year_dataset(ds, year=2014)
    assert set(out.data_vars) == {"CRW_SST", "CRW_SSTANOMALY", "CRW_DHW"}
    assert out.sizes["time"] == 1
