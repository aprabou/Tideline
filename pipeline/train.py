"""
train.py — Train an XGBoost classifier to forecast marine heatwaves.

Target: mhw (bool) — will this grid cell be in a heatwave in the next N days?
Features: sst, sst_anom, rolling means, trend, seasonal encoding, lat, lon.

Outputs: models/xgb_tideline.json  (XGBoost native format)
         models/feature_importance.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

SILVER_DIR = Path("data/silver")
MODELS_DIR = Path("models")
LABELED_PATH = SILVER_DIR / "features_labeled.parquet"
MODEL_PATH = MODELS_DIR / "xgb_tideline.json"
META_PATH = MODELS_DIR / "meta.json"

FEATURE_COLS = [
    "sst", "sst_anom",
    "rolling_7d_mean", "rolling_14d_mean", "rolling_30d_mean",
    "sst_trend_14d",
    "month_sin", "month_cos",
    "lat", "lon",
]
TARGET_COL = "mhw"
FORECAST_HORIZON = 7  # predict MHW occurrence N days ahead

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def shift_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Shift the target forward by *horizon* days per location."""
    df = df.sort_values(["lat", "lon", "time"]).copy()
    df[TARGET_COL] = (
        df.groupby(["lat", "lon"])[TARGET_COL]
        .shift(-horizon)
    )
    return df.dropna(subset=[TARGET_COL])


def train(
    labeled_path: Path = LABELED_PATH,
    model_path: Path = MODEL_PATH,
    horizon: int = FORECAST_HORIZON,
) -> xgb.XGBClassifier:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("loading labeled data from %s", labeled_path)
    df = pd.read_parquet(labeled_path)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = shift_target(df, horizon=horizon)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    log.info("features: %s", available_features)

    X = df[available_features].astype(np.float32)
    y = df[TARGET_COL].astype(int)

    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        auc_scores.append(auc)
        log.info("fold %d AUC: %.4f", fold + 1, auc)

    log.info("mean CV AUC: %.4f ± %.4f", np.mean(auc_scores), np.std(auc_scores))

    # Final model on all data
    final = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    final.fit(X, y)
    final.save_model(str(model_path))
    log.info("model saved: %s", model_path)

    # Feature importance
    imp = pd.Series(
        final.feature_importances_, index=available_features
    ).sort_values(ascending=False)
    imp.to_csv(MODELS_DIR / "feature_importance.csv", header=["importance"])
    log.info("feature importances:\n%s", imp.to_string())

    # Save metadata
    meta = {
        "horizon_days": horizon,
        "features": available_features,
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_auc_std": float(np.std(auc_scores)),
        "n_training_rows": len(X),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    # Final classification report on last CV fold
    y_pred = final.predict(X.iloc[val_idx])  # type: ignore[possibly-undefined]
    print(classification_report(y.iloc[val_idx], y_pred, target_names=["no MHW", "MHW"]))

    return final


if __name__ == "__main__":
    train()
