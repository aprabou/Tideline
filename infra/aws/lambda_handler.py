"""
lambda_handler.py — FastAPI inference API, deployable as an AWS Lambda container.

Endpoints:
  GET  /health
  POST /forecast   body: { lat: float, lon: float, date: str (YYYY-MM-DD) }
  GET  /forecast/grid?date=YYYY-MM-DD&lat_min=&lat_max=&lon_min=&lon_max=

The model artifact is loaded from S3 on cold start (or /tmp if already cached).
Set MODEL_S3_URI env var: s3://your-bucket/models/xgb_tideline.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Tideline Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_S3_URI = os.environ.get("MODEL_S3_URI", "")
LOCAL_MODEL_PATH = Path(tempfile.gettempdir()) / "xgb_tideline.json"
FEATURE_ORDER = [
    "sst", "sst_anom",
    "rolling_7d_mean", "rolling_14d_mean", "rolling_30d_mean",
    "sst_trend_14d",
    "month_sin", "month_cos",
    "lat", "lon",
]


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load XGBoost model from S3 (or local cache) once per Lambda instance."""
    import xgboost as xgb

    if LOCAL_MODEL_PATH.exists():
        log.info("loading model from local cache %s", LOCAL_MODEL_PATH)
    elif MODEL_S3_URI:
        import boto3
        log.info("downloading model from %s", MODEL_S3_URI)
        bucket, key = MODEL_S3_URI.replace("s3://", "").split("/", 1)
        boto3.client("s3").download_file(bucket, key, str(LOCAL_MODEL_PATH))
    else:
        raise RuntimeError("No model found. Set MODEL_S3_URI or provide a local model.")

    clf = xgb.XGBClassifier()
    clf.load_model(str(LOCAL_MODEL_PATH))
    return clf


def build_features(lat: float, lon: float, target_date: date) -> np.ndarray:
    """
    Build a single feature vector for (lat, lon, date).

    In production, real SST and precomputed rolling stats would be fetched
    from a feature store (e.g. DynamoDB or S3 parquet). This stub uses
    climatological approximations for demonstration.
    """
    doy = target_date.timetuple().tm_yday
    month = target_date.month

    # Placeholder: replace with real SST lookup
    sst = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 90) / 365) - 0.01 * abs(lat - 35)
    sst_anom = 0.0
    rolling_7 = sst
    rolling_14 = sst
    rolling_30 = sst
    trend = 0.0
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return np.array(
        [[sst, sst_anom, rolling_7, rolling_14, rolling_30, trend,
          month_sin, month_cos, lat, lon]],
        dtype=np.float32,
    )


# --- Request / Response models ---

class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class DayForecast(BaseModel):
    date: str
    prob_mhw: float
    category: str  # "none", "moderate", "strong", "severe", "extreme"


class ForecastResponse(BaseModel):
    lat: float
    lon: float
    horizon_days: int
    forecasts: list[DayForecast]


def prob_to_category(p: float) -> str:
    if p < 0.25:
        return "none"
    if p < 0.50:
        return "moderate"
    if p < 0.70:
        return "strong"
    if p < 0.85:
        return "severe"
    return "extreme"


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        clf = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    start = date.fromisoformat(req.date)
    forecasts = []
    for offset in range(14):
        target = start + timedelta(days=offset)
        X = build_features(req.lat, req.lon, target)
        prob = float(clf.predict_proba(X)[0, 1])
        forecasts.append(DayForecast(
            date=target.isoformat(),
            prob_mhw=round(prob, 4),
            category=prob_to_category(prob),
        ))

    return ForecastResponse(
        lat=req.lat,
        lon=req.lon,
        horizon_days=14,
        forecasts=forecasts,
    )


@app.get("/forecast/grid", response_model=list[dict])
def forecast_grid(
    date: str,
    lat_min: float = 32.0,
    lat_max: float = 48.0,
    lon_min: float = -130.0,
    lon_max: float = -115.0,
    step: float = 1.0,
) -> list[dict]:
    """Return day-1 MHW probability for a bounding-box grid (coarse, for map overlay)."""
    try:
        clf = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    target = date_from_str(date)
    results = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            X = build_features(lat, lon, target)
            prob = float(clf.predict_proba(X)[0, 1])
            results.append({"lat": lat, "lon": lon, "prob_mhw": round(prob, 4)})
            lon += step
        lat += step
    return results


def date_from_str(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {s}") from exc


# AWS Lambda entrypoint via Mangum
def handler(event: dict, context: object) -> dict:
    try:
        from mangum import Mangum
        return Mangum(app)(event, context)
    except ImportError:
        raise RuntimeError("mangum not installed — add it to requirements.txt for Lambda")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
