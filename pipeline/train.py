"""
pipeline/train.py — Train LightGBM and XGBoost MHW forecast models.

Reads unified feature table from data/silver/feature_table.parquet.
Trains one model per algorithm per lead time (1 / 3 / 5 / 7 days ahead).

Usage:
    python3 -m pipeline.train --lead 1
    python3 -m pipeline.train --lead 7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

FEATURE_TABLE_PATH = Path("data/silver/feature_table.parquet")
MODELS_DIR = Path("models")
LEAD_TIMES = [1, 3, 5, 7]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "buoy_sst_idw", "buoy_sst_7d_mean", "buoy_sst_30d_mean", "buoy_anomaly",
    "sat_sst", "sat_sst_anomaly", "sat_dhw", "sat_sst_gradient", "sat_days_since_cold",
    "calcofi_temp_50m", "calcofi_temp_100m", "calcofi_salinity_50m",
    "calcofi_thermocline_depth", "calcofi_chla",
    "kelp_canopy_extent", "kelp_canopy_anomaly", "kelp_density", "kelp_recovery_index",
    "month_sin", "month_cos", "lat", "lon",
]


def _add_temporal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def _make_target(df: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Shift mhw_status forward by `lead` days per cell to create forecast target."""
    df = df.sort_values(["cell_id", "date"]).copy()
    df["target"] = df.groupby("cell_id")["mhw_status"].shift(-lead)
    return df.dropna(subset=["target"])


def _train_lgbm(X: pd.DataFrame, y: pd.Series, lead: int) -> None:
    import lightgbm as lgb

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=pos_w,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        auc_scores.append(auc)
        log.info("LightGBM lead=%dd fold %d AUC: %.4f", lead, fold + 1, auc)

    log.info("LightGBM lead=%dd mean CV AUC: %.4f ± %.4f", lead, np.mean(auc_scores), np.std(auc_scores))

    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    final = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pos_w,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    final.fit(X, y)
    path = MODELS_DIR / f"lgbm_lead_{lead}d.txt"
    final.booster_.save_model(str(path))
    log.info("LightGBM model saved: %s", path)

    meta = {
        "lead_days": lead, "algorithm": "lightgbm",
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_auc_std": float(np.std(auc_scores)),
        "n_rows": len(X),
        "features": list(X.columns),
    }
    (MODELS_DIR / f"lgbm_lead_{lead}d_meta.json").write_text(json.dumps(meta, indent=2))


def _train_xgb(X: pd.DataFrame, y: pd.Series, lead: int) -> None:
    import xgboost as xgb

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, early_stopping_rounds=30,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        auc_scores.append(auc)
        log.info("XGBoost   lead=%dd fold %d AUC: %.4f", lead, fold + 1, auc)

    log.info("XGBoost   lead=%dd mean CV AUC: %.4f ± %.4f", lead, np.mean(auc_scores), np.std(auc_scores))

    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    final = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    final.fit(X, y)
    path = MODELS_DIR / f"xgb_lead_{lead}d.json"
    final.save_model(str(path))
    log.info("XGBoost   model saved: %s", path)

    meta = {
        "lead_days": lead, "algorithm": "xgboost",
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_auc_std": float(np.std(auc_scores)),
        "n_rows": len(X),
        "features": list(X.columns),
    }
    (MODELS_DIR / f"xgb_lead_{lead}d_meta.json").write_text(json.dumps(meta, indent=2))


def train(lead: int) -> None:
    log.info("=== Training lead=%dd models ===", lead)
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = _add_temporal_encoding(df)
    df = _make_target(df, lead)

    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    log.info("using %d / %d feature columns", len(available_feats), len(FEATURE_COLS))

    X = df[available_feats].astype(np.float32)
    y = df["target"].astype(int)

    mhw_rate = y.mean()
    log.info("MHW prevalence at lead=%dd: %.2f%%", lead, mhw_rate * 100)

    _train_lgbm(X, y, lead)
    _train_xgb(X, y, lead)
    log.info("=== lead=%dd complete ===", lead)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=int, choices=LEAD_TIMES, required=True)
    args = parser.parse_args()
    train(args.lead)


if __name__ == "__main__":
    main()
