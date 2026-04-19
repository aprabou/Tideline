"""
tests/test_api.py — TDD tests for api/main.py.

All tests are written before implementation (RED phase).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEAD_TIMES = [1, 3, 5, 7]

_FAKE_LEAD_RESULTS = {
    lt: {
        "prob": 0.10 + lt * 0.05,
        "ci_low": max(0.0, 0.10 + lt * 0.05 - 0.08),
        "ci_high": min(1.0, 0.10 + lt * 0.05 + 0.08),
        "top_features": {"sst_anom": 0.8, "rolling_7d_mean": 0.6, "calcofi_temp_50m": 0.4},
    }
    for lt in LEAD_TIMES
}

_FAKE_BLOB = [
    {
        "date": f"2014-0{m}-01",
        "cell_id": i,
        "lat": 45.0 + i,
        "lon": -140.0 - i,
        "prob_mhw": 0.2 + m * 0.05,
        "category": "none",
        "lead_time": 7,
    }
    for m in range(1, 4)
    for i in range(3)
]


def _make_mock_ensemble() -> MagicMock:
    ens = MagicMock()
    ens.predict.return_value = _FAKE_LEAD_RESULTS
    return ens


@pytest.fixture
def client():
    from api.main import app, get_ensemble, get_blob_data

    app.dependency_overrides[get_ensemble] = _make_mock_ensemble
    app.dependency_overrides[get_blob_data] = lambda: _FAKE_BLOB
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_gemini(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from api.main import app, get_ensemble, get_blob_data

    app.dependency_overrides[get_ensemble] = _make_mock_ensemble
    app.dependency_overrides[get_blob_data] = lambda: _FAKE_BLOB
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------


def test_forecast_returns_200(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=2023-06-01")
    assert r.status_code == 200


def test_forecast_returns_four_lead_times(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=2023-06-01")
    data = r.json()
    assert len(data["forecasts"]) == 4
    assert [f["lead_days"] for f in data["forecasts"]] == LEAD_TIMES


def test_forecast_has_confidence_intervals(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=2023-06-01")
    for f in r.json()["forecasts"]:
        assert "ci_low" in f
        assert "ci_high" in f


def test_forecast_ci_bounds_are_valid(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=2023-06-01")
    for f in r.json()["forecasts"]:
        assert 0.0 <= f["ci_low"] <= f["prob_mhw"] <= f["ci_high"] <= 1.0


def test_forecast_has_valid_category(client: TestClient) -> None:
    valid = {"none", "moderate", "strong", "severe", "extreme"}
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=2023-06-01")
    for f in r.json()["forecasts"]:
        assert f["category"] in valid


def test_forecast_has_target_dates(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=2023-06-01")
    dates = [f["target_date"] for f in r.json()["forecasts"]]
    assert dates == ["2023-06-02", "2023-06-04", "2023-06-06", "2023-06-08"]


def test_forecast_invalid_date_422(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&lon=-120.0&date=not-a-date")
    assert r.status_code == 422


def test_forecast_lat_out_of_range_422(client: TestClient) -> None:
    r = client.get("/forecast?lat=999&lon=-120.0&date=2023-06-01")
    assert r.status_code == 422


def test_forecast_missing_param_422(client: TestClient) -> None:
    r = client.get("/forecast?lat=34.0&date=2023-06-01")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /forecast_grid
# ---------------------------------------------------------------------------


def test_forecast_grid_returns_200(client: TestClient) -> None:
    r = client.get("/forecast_grid?date=2023-06-01&lead_time=7")
    assert r.status_code == 200


def test_forecast_grid_returns_cell_list(client: TestClient) -> None:
    r = client.get("/forecast_grid?date=2023-06-01&lead_time=7")
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0
    cell = data[0]
    assert {"lat", "lon", "prob_mhw", "category"} <= set(cell.keys())


def test_forecast_grid_probs_in_range(client: TestClient) -> None:
    r = client.get("/forecast_grid?date=2023-06-01&lead_time=7")
    for cell in r.json():
        assert 0.0 <= cell["prob_mhw"] <= 1.0


def test_forecast_grid_invalid_lead_time_422(client: TestClient) -> None:
    r = client.get("/forecast_grid?date=2023-06-01&lead_time=99")
    assert r.status_code == 422


def test_forecast_grid_invalid_date_422(client: TestClient) -> None:
    r = client.get("/forecast_grid?date=bad-date&lead_time=7")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /backtest/blob
# ---------------------------------------------------------------------------


def test_backtest_blob_returns_200(client: TestClient) -> None:
    r = client.get("/backtest/blob")
    assert r.status_code == 200


def test_backtest_blob_structure(client: TestClient) -> None:
    data = client.get("/backtest/blob").json()
    assert "event_name" in data
    assert "description" in data
    assert "timeline" in data


def test_backtest_blob_timeline_non_empty(client: TestClient) -> None:
    data = client.get("/backtest/blob").json()
    assert len(data["timeline"]) > 0


def test_backtest_blob_timeline_entry_shape(client: TestClient) -> None:
    entry = client.get("/backtest/blob").json()["timeline"][0]
    assert "date" in entry
    assert "region_mean_prob" in entry
    assert "cells" in entry


# ---------------------------------------------------------------------------
# /summary
# ---------------------------------------------------------------------------


def test_summary_returns_200(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    import api.main as m

    monkeypatch.setattr(m, "_call_gemini", lambda *a, **kw: "Elevated SST anomalies drive risk.")
    r = client.get("/summary?lat=34.0&lon=-120.0&date=2023-06-01")
    assert r.status_code == 200


def test_summary_includes_forecasts(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    import api.main as m

    monkeypatch.setattr(m, "_call_gemini", lambda *a, **kw: "Elevated SST anomalies drive risk.")
    data = client.get("/summary?lat=34.0&lon=-120.0&date=2023-06-01").json()
    assert "forecasts" in data
    assert len(data["forecasts"]) == 4


def test_summary_includes_nl_text(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    import api.main as m

    monkeypatch.setattr(m, "_call_gemini", lambda *a, **kw: "Elevated SST anomalies drive risk.")
    data = client.get("/summary?lat=34.0&lon=-120.0&date=2023-06-01").json()
    assert "summary" in data
    assert len(data["summary"]) > 10


def test_summary_no_api_key_returns_503(client_no_gemini: TestClient) -> None:
    r = client_no_gemini.get("/summary?lat=34.0&lon=-120.0&date=2023-06-01")
    assert r.status_code == 503
