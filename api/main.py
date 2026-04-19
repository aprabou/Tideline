"""
api/main.py — FastAPI backend for the Tideline marine heatwave forecasting system.

Endpoints:
  GET /forecast?lat=&lon=&date=        → predictions for all 4 lead times with CIs
  GET /forecast_grid?date=&lead_time=  → predictions for all grid cells (heatmap)
  GET /backtest/blob                   → 2015 Pacific Blob replay data
  GET /summary?lat=&lon=&date=         → Gemini-generated NL forecast summary

Deploy target: GCP Cloud Run.
Set GEMINI_API_KEY env var before calling /summary.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.ensemble import EnsembleWrapper, LeadResult, _prob_to_category, build_pacific_blob_replay
from models.lightgbm_model import LEAD_TIMES
from api.sd_predictor import get_sd_predictor, SDPredictor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Tideline API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Grid definition used by /forecast_grid (NE Pacific, 2° resolution)
_GRID_CELLS: list[dict[str, float]] = [
    {"lat": lat, "lon": lon}
    for lat in [32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0]
    for lon in [-130.0, -128.0, -126.0, -124.0, -122.0, -120.0, -118.0, -116.0]
]


# ---------------------------------------------------------------------------
# Dependencies (overridable in tests)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_ensemble() -> EnsembleWrapper:
    return EnsembleWrapper.load()


@lru_cache(maxsize=1)
def get_blob_data() -> list[dict]:
    return build_pacific_blob_replay()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class LeadForecast(BaseModel):
    lead_days: int
    target_date: str
    prob_mhw: float
    ci_low: float
    ci_high: float
    category: str
    top_features: dict[str, float]


class ForecastResponse(BaseModel):
    lat: float
    lon: float
    date: str
    forecasts: list[LeadForecast]


class GridCell(BaseModel):
    lat: float
    lon: float
    prob_mhw: float
    category: str


class BlobTimestep(BaseModel):
    date: str
    region_mean_prob: float
    cells: list[dict[str, Any]]


class BlobResponse(BaseModel):
    event_name: str
    description: str
    timeline: list[BlobTimestep]


class SummaryResponse(BaseModel):
    lat: float
    lon: float
    date: str
    forecasts: list[LeadForecast]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date '{s}'. Expected YYYY-MM-DD.")


def _lead_result_to_schema(
    lead_time: int,
    base_date: date,
    result: LeadResult | dict,
) -> LeadForecast:
    if isinstance(result, dict):
        prob = float(result["prob"])
        ci_low = float(result["ci_low"])
        ci_high = float(result["ci_high"])
        top_features = dict(result.get("top_features", {}))
    else:
        prob = result.prob
        ci_low = result.ci_low
        ci_high = result.ci_high
        top_features = result.top_features

    return LeadForecast(
        lead_days=lead_time,
        target_date=(base_date + timedelta(days=lead_time)).isoformat(),
        prob_mhw=round(prob, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        category=_prob_to_category(prob),
        top_features={k: round(v, 4) for k, v in top_features.items()},
    )


def _call_gemini(location: str, forecast_date: str, prob: float, features: dict[str, float]) -> str:
    """Call an LLM via OpenRouter to generate a fisheries-manager-friendly summary."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY not set. Configure the environment variable to enable /summary.",
        )

    try:
        import httpx  # noqa: PLC0415

        feature_str = ", ".join(f"{k}={v:.2f}" for k, v in features.items())
        prompt = (
            f"Given this marine heatwave forecast for {location} on {forecast_date}: "
            f"MHW probability = {prob:.1%}, top contributing features = {feature_str}. "
            "Generate a 2-3 sentence summary for a fisheries manager. "
            "Cite specific oceanographic drivers."
        )
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "nvidia/nemotron-3-super-120b-a12b:free", "messages": [{"role": "user", "content": prompt}]},
            timeout=30.0,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
    except HTTPException:
        raise
    except Exception as exc:
        log.error("OpenRouter call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenRouter API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    date: Annotated[str, Query()],
    ensemble: EnsembleWrapper = Depends(get_ensemble),
) -> ForecastResponse:
    base_date = _parse_date(date)

    # Use the San Diego MVP model for SD coastal requests
    sd = get_sd_predictor()
    if sd.in_sd_box(lat, lon):
        sd_results = sd.predict(lat, lon, base_date)
        # Map SD leads (3, 7) to the full LEAD_TIMES response
        forecasts = []
        for lt in LEAD_TIMES:
            # nearest SD lead (3→3, 5→7, 7→7, 1→3)
            nearest = min(sd_results.keys(), key=lambda k: abs(k - lt))
            r = sd_results[nearest]
            forecasts.append(_lead_result_to_schema(lt, base_date, r))
        return ForecastResponse(lat=lat, lon=lon, date=date, forecasts=forecasts)

    # Global fallback: ensemble stubs
    results = ensemble.predict(lat, lon, base_date)
    return ForecastResponse(
        lat=lat,
        lon=lon,
        date=date,
        forecasts=[_lead_result_to_schema(lt, base_date, results[lt]) for lt in LEAD_TIMES],
    )


@app.get("/forecast_grid", response_model=list[GridCell])
def forecast_grid(
    date: Annotated[str, Query()],
    lead_time: Annotated[int, Query()],
    ensemble: EnsembleWrapper = Depends(get_ensemble),
) -> list[GridCell]:
    if lead_time not in LEAD_TIMES:
        raise HTTPException(
            status_code=422,
            detail=f"lead_time must be one of {LEAD_TIMES}, got {lead_time}.",
        )
    base_date = _parse_date(date)
    cells: list[GridCell] = []
    for cell in _GRID_CELLS:
        results = ensemble.predict(cell["lat"], cell["lon"], base_date)
        lr = results[lead_time]
        prob = lr["prob"] if isinstance(lr, dict) else lr.prob
        cells.append(GridCell(
            lat=cell["lat"],
            lon=cell["lon"],
            prob_mhw=round(float(prob), 4),
            category=_prob_to_category(float(prob)),
        ))
    return cells


@app.get("/backtest/blob", response_model=BlobResponse)
def backtest_blob(blob_rows: list[dict] = Depends(get_blob_data)) -> BlobResponse:
    # Aggregate rows by date → BlobTimestep
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in blob_rows:
        by_date[row["date"]].append(row)

    timeline: list[BlobTimestep] = []
    for d in sorted(by_date.keys()):
        cells = by_date[d]
        mean_prob = round(float(sum(c["prob_mhw"] for c in cells) / len(cells)), 4)
        timeline.append(BlobTimestep(date=d, region_mean_prob=mean_prob, cells=cells))

    return BlobResponse(
        event_name="2014-2015 Pacific Blob",
        description=(
            "The 2014-2015 Northeast Pacific marine heatwave, known as 'The Blob', "
            "was one of the most persistent warm-water anomalies on record. "
            "SST anomalies peaked at +2.5°C above the 1981-2010 climatological mean "
            "across much of the Gulf of Alaska and California Current System."
        ),
        timeline=timeline,
    )


@app.get("/summary", response_model=SummaryResponse)
def summary(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    date: Annotated[str, Query()],
    ensemble: EnsembleWrapper = Depends(get_ensemble),
) -> SummaryResponse:
    base_date = _parse_date(date)
    results = ensemble.predict(lat, lon, base_date)
    forecasts = [_lead_result_to_schema(lt, base_date, results[lt]) for lt in LEAD_TIMES]

    # Use 7-day lead for the summary (most actionable for fisheries managers)
    lead7 = forecasts[-1]
    nl_summary = _call_gemini(
        location=f"({lat:.2f}°N, {abs(lon):.2f}°W)",
        forecast_date=date,
        prob=lead7.prob_mhw,
        features=lead7.top_features,
    )

    return SummaryResponse(
        lat=lat,
        lon=lon,
        date=date,
        forecasts=forecasts,
        summary=nl_summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
