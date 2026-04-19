from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from models.ensemble import ENSEMBLE_2023_PATH, build_ensemble_predictions, build_pacific_blob_replay
from models.lightgbm_model import FEATURE_TABLE_PATH, LEAD_TIMES

BACKTEST_DIR = Path("backtest")
FIG_DIR = BACKTEST_DIR / "figures"
METRICS_PATH = BACKTEST_DIR / "metrics.json"


def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_pr_auc(y: pd.Series, p: pd.Series) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _precision_at_recall(y: pd.Series, p: pd.Series, target_recall: float = 0.9) -> float:
    if y.nunique() < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, p)
    valid = np.where(recall >= target_recall)[0]
    if len(valid) == 0:
        return float("nan")
    return float(np.max(precision[valid]))


def _confusion(y: pd.Series, p: pd.Series, threshold: float = 0.5) -> dict[str, int]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _lead_metrics(df: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lead in LEAD_TIMES:
        d = df[df["lead_time"] == lead].dropna(subset=["y_true", prob_col])
        y = d["y_true"].astype(int)
        p = d[prob_col].astype(float)
        out[f"lead_{lead}d"] = {
            "auc": _safe_auc(y, p),
            "pr_auc": _safe_pr_auc(y, p),
            "brier": float(brier_score_loss(y, p)) if len(d) else float("nan"),
            "confusion_p50": _confusion(y, p) if len(d) else {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "precision_at_90_recall": _precision_at_recall(y, p),
            "n": int(len(d)),
        }
    return out


def _plot_roc(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for lead in LEAD_TIMES:
        d = df[df["lead_time"] == lead].dropna(subset=["y_true", "ensemble_prob", "lgbm_prob", "cnn_prob"])
        if d.empty or d["y_true"].nunique() < 2:
            continue

        plt.figure(figsize=(7, 6))
        for name, col in [("Ensemble", "ensemble_prob"), ("LightGBM", "lgbm_prob"), ("RasterCNN", "cnn_prob")]:
            fpr, tpr, _ = roc_curve(d["y_true"].astype(int), d[col].astype(float))
            sns.lineplot(x=fpr, y=tpr, label=name)

        sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Random")
        plt.title(f"ROC Curves - Lead {lead}d")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"roc_lead_{lead}d.png", dpi=150)
        plt.close()


def _plot_calibration(df: pd.DataFrame) -> None:
    for lead in LEAD_TIMES:
        d = df[df["lead_time"] == lead].dropna(subset=["y_true", "ensemble_prob", "lgbm_prob", "cnn_prob"])
        if d.empty or d["y_true"].nunique() < 2:
            continue

        plt.figure(figsize=(7, 6))
        for name, col in [("Ensemble", "ensemble_prob"), ("LightGBM", "lgbm_prob"), ("RasterCNN", "cnn_prob")]:
            frac_pos, mean_pred = calibration_curve(d["y_true"].astype(int), d[col].astype(float), n_bins=10, strategy="quantile")
            sns.lineplot(x=mean_pred, y=frac_pos, marker="o", label=name)

        sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Perfect")
        plt.title(f"Calibration - Lead {lead}d")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"calibration_lead_{lead}d.png", dpi=150)
        plt.close()


def _plot_timeseries(df: pd.DataFrame) -> None:
    lead7 = df[df["lead_time"] == 7].copy()
    cell_ids = lead7["cell_id"].dropna().unique()[:3]

    for cell in cell_ids:
        d = lead7[lead7["cell_id"] == cell].sort_values("date")
        plt.figure(figsize=(11, 4))
        sns.lineplot(data=d, x="date", y="ensemble_prob", label="Ensemble p(7d)")
        sns.lineplot(data=d, x="date", y="y_true", label="Actual (y_7d)")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Predicted vs Actual MHW - Cell {cell}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"timeseries_cell_{cell}.png", dpi=150)
        plt.close()


def _plot_auc_map(df: pd.DataFrame, feature_table: pd.DataFrame) -> None:
    lead7 = df[df["lead_time"] == 7].dropna(subset=["y_true", "ensemble_prob"]).copy()

    rows: list[dict[str, float]] = []
    for cell_id, g in lead7.groupby("cell_id"):
        if g["y_true"].nunique() < 2:
            continue
        rows.append({"cell_id": float(np.asarray(cell_id).item()), "auc": float(roc_auc_score(g["y_true"], g["ensemble_prob"]))})

    if not rows:
        return

    perf = pd.DataFrame(rows)
    geo = feature_table[["cell_id", "lat", "lon"]].drop_duplicates("cell_id")
    plot_df = geo.merge(perf, on="cell_id", how="inner")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(plot_df["lon"], plot_df["lat"], c=plot_df["auc"], cmap="viridis", s=18)
    plt.colorbar(sc, label="AUC (Lead 7d)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Per-cell Ensemble Performance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "map_auc_per_cell.png", dpi=150)
    plt.close()


def _extract_events(feature_table: pd.DataFrame) -> pd.DataFrame:
    ft = feature_table.copy()
    ft["date"] = pd.to_datetime(ft["date"])
    ft = ft[(ft["date"] >= "2023-01-01") & (ft["date"] <= "2023-12-31")].sort_values(["cell_id", "date"])

    events: list[dict[str, Any]] = []
    for cell_id, g in ft.groupby("cell_id"):
        status = g["mhw_status"].fillna(0).astype(int).to_numpy()
        dates = pd.to_datetime(g["date"]).to_numpy()
        i = 0
        while i < len(status):
            if status[i] == 0:
                i += 1
                continue
            start = i
            while i < len(status) and status[i] == 1:
                i += 1
            end = i - 1
            events.append(
                {
                    "cell_id": int(np.asarray(cell_id).item()),
                    "event_start": pd.Timestamp(dates[start]),
                    "event_end": pd.Timestamp(dates[end]),
                    "duration_days": int(end - start + 1),
                }
            )
    return pd.DataFrame(events)


def _event_metrics(ensemble_df: pd.DataFrame, events: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    if events.empty:
        return {"n_events": 0, "detected_events": 0, "detection_rate": float("nan"), "events": []}

    pred = ensemble_df.copy()
    pred["date"] = pd.to_datetime(pred["date"])

    event_rows: list[dict[str, Any]] = []
    detected = 0
    for ev in events.to_dict(orient="records"):
        cell = int(ev["cell_id"])
        start = pd.Timestamp(ev["event_start"])

        first_alert_lead = None
        first_alert_date = None
        for lead in sorted(LEAD_TIMES, reverse=True):
            ref_date = start - pd.Timedelta(days=lead)
            m = pred[(pred["cell_id"] == cell) & (pred["lead_time"] == lead) & (pred["date"] == ref_date)]
            if m.empty:
                continue
            if float(m["ensemble_prob"].iloc[0]) >= threshold:
                first_alert_lead = lead
                first_alert_date = ref_date
                break

        did_predict = first_alert_lead is not None
        if did_predict:
            detected += 1

        event_rows.append(
            {
                    "cell_id": int(cell),
                "event_start": str(start.date()),
                    "event_end": str(pd.Timestamp(ev["event_end"]).date()),
                    "duration_days": int(ev["duration_days"]),
                "did_predict": bool(did_predict),
                "first_alert_lead_days": int(first_alert_lead) if first_alert_lead is not None else None,
                "first_alert_date": str(first_alert_date.date()) if first_alert_date is not None else None,
            }
        )

    return {
        "n_events": int(len(events)),
        "detected_events": int(detected),
        "detection_rate": float(detected / max(len(events), 1)),
        "events": event_rows,
    }


def evaluate_backtest() -> dict[str, Any]:
    ensemble = build_ensemble_predictions() if not ENSEMBLE_2023_PATH.exists() else pd.read_parquet(ENSEMBLE_2023_PATH)
    ensemble["date"] = pd.to_datetime(ensemble["date"])

    feature_table = pd.read_parquet(FEATURE_TABLE_PATH)
    feature_table["date"] = pd.to_datetime(feature_table["date"])

    metrics = {
        "ensemble": _lead_metrics(ensemble, "ensemble_prob"),
        "lightgbm": _lead_metrics(ensemble, "lgbm_prob"),
        "raster_cnn": _lead_metrics(ensemble, "cnn_prob"),
    }

    events = _extract_events(feature_table)
    metrics["event_metrics"] = _event_metrics(ensemble, events)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_roc(ensemble)
    _plot_calibration(ensemble)
    _plot_timeseries(ensemble)
    _plot_auc_map(ensemble, feature_table)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    build_pacific_blob_replay()
    return metrics


def main() -> None:
    metrics = evaluate_backtest()
    print(json.dumps({"saved_metrics": str(METRICS_PATH), "keys": list(metrics.keys())}, indent=2))


if __name__ == "__main__":
    main()
