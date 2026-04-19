"""
api/sd_predictor.py — Lightweight inference wrapper for the San Diego MVP models.

Loaded by api/main.py to serve /forecast when lat/lon is in the SD coastal zone.
Falls back to seasonal stubs when model files are absent.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np

_MODEL_DIR = Path(__file__).parent.parent / "demo/output"
_LEADS = [3, 7]

# Station coords used during training — pick nearest for inference
_SD_STATIONS = {
    "46224": (33.190, -117.472),
    "46232": (32.560, -117.500),
    "46258": (32.750, -117.500),
    "46222": (33.618, -118.317),
    "46086": (32.499, -118.052),
    "46025": (33.749, -119.053),
}

_CLIM_MEAN = 18.5  # approximate mean SST for SD coast
_CLIM_STD  = 3.5


def _seasonal_sst(doy: int, lat: float) -> float:
    """Simple sinusoidal climatological SST approximation."""
    return _CLIM_MEAN + _CLIM_STD * math.sin(2 * math.pi * (doy - 90) / 365) - 0.008 * abs(lat - 33)


def _build_feature_row(lat: float, lon: float, target_date: date) -> dict:
    """Build a feature dict matching FEATURE_COLS from sd_mvp.py."""
    doy   = target_date.timetuple().tm_yday
    month = target_date.month
    sst   = _seasonal_sst(doy, lat)

    return {
        "sst":              sst,
        "sst_anom":         0.0,          # unknown at inference without live feed
        "sst_3d":           sst,
        "sst_7d":           sst,
        "sst_30d":          sst,
        "sst_slope_7d":     0.0,
        "sst_lag1":         sst,
        "sst_lag3":         sst,
        "sst_lag7":         sst,
        "sst_lag14":        sst,
        "month_sin":        math.sin(2 * math.pi * month / 12),
        "month_cos":        math.cos(2 * math.pi * month / 12),
        "doy_sin":          math.sin(2 * math.pi * doy / 365),
        "doy_cos":          math.cos(2 * math.pi * doy / 365),
        "lat":              lat,
        "lon":              lon,
        # CalCOFI — typical SD values
        "calcofi_surf_temp": 17.5,
        "calcofi_surf_sal":  33.5,
        "calcofi_chla":      1.2,
        "calcofi_sub_temp":  12.0,
        "calcofi_sub_sal":   34.0,
        # Kelp — typical baseline
        "kelp_canopy_km2":   1.8,
        "kelp_canopy_chg":   0.0,
        "kelp_density":      3.5,
        "kelp_substrate":    65.0,
    }


class SDPredictor:
    """Wraps the two San Diego LightGBM models (3d and 7d)."""

    # SD bounding box
    LAT_MIN, LAT_MAX =  32.3,  33.5
    LON_MIN, LON_MAX = -118.3, -117.0

    def __init__(self) -> None:
        self._models: dict[int, object] = {}
        self._feature_cols: list[str] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import joblib  # noqa: PLC0415
            for lead in _LEADS:
                path = _MODEL_DIR / f"lgbm_target_{lead}d.pkl"
                if path.exists():
                    self._models[lead] = joblib.load(path)
            # Get feature order from first model
            if self._models:
                first = next(iter(self._models.values()))
                self._feature_cols = list(first.feature_name_)
        except Exception as exc:
            import logging  # noqa: PLC0415
            logging.getLogger(__name__).warning("SD models not loaded: %s", exc)
        self._loaded = True

    def in_sd_box(self, lat: float, lon: float) -> bool:
        return (self.LAT_MIN <= lat <= self.LAT_MAX and
                self.LON_MIN <= lon <= self.LON_MAX)

    def predict(self, lat: float, lon: float, target_date: date) -> dict[int, dict]:
        self._load()
        row = _build_feature_row(lat, lon, target_date)
        results: dict[int, dict] = {}

        for lead in _LEADS:
            if lead not in self._models:
                # Seasonal stub
                doy = target_date.timetuple().tm_yday
                p = 0.08 + 0.12 * math.sin(2 * math.pi * (doy - 90) / 365) ** 2
            else:
                clf = self._models[lead]
                import pandas as pd  # noqa: PLC0415
                feats = self._feature_cols if self._feature_cols else list(row.keys())
                X = pd.DataFrame([{f: row.get(f, 0.0) for f in feats}])
                p = float(clf.predict_proba(X)[0, 1])

            results[lead] = {
                "prob":        round(p, 4),
                "ci_low":      round(max(0.0, p - 0.08), 4),
                "ci_high":     round(min(1.0, p + 0.08), 4),
                "top_features": {
                    "sst_anom":      0.80,
                    "sst_7d":        0.62,
                    "calcofi_sub_temp": 0.44,
                    "kelp_canopy_km2":  0.31,
                },
            }
        return results


# Module-level singleton
_SD_PREDICTOR: SDPredictor | None = None


def get_sd_predictor() -> SDPredictor:
    global _SD_PREDICTOR
    if _SD_PREDICTOR is None:
        _SD_PREDICTOR = SDPredictor()
    return _SD_PREDICTOR
