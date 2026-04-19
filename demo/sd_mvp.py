"""
sd_mvp.py — San Diego Coast Marine Heatwave Classifier

Trains two LightGBM models:
  - mhw_3d: will there be an MHW at this site within 3 days?
  - mhw_7d: will there be an MHW at this site within 7 days?

Geography: San Diego coastal zone (32.3–33.5°N, 117.0–118.3°W)
Time split: train 2014–2019 | val 2020–2021 | test 2022–2023
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, classification_report,
)
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
BUOY_PATH    = ROOT / "data/raw/ingestion/noaa_buoys_sst.parquet"
CALCOFI_PATH = ROOT / "data/raw/ingestion/calcofi_bottle.parquet"
KELP_CANOPY  = ROOT / "data/raw/ingestion/scripps_kelp_canopy.parquet"
KELP_DENSITY = ROOT / "data/raw/ingestion/sdcity_kelp_density.parquet"
OUT_DIR      = ROOT / "demo/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── San Diego bounding box ────────────────────────────────────────────────────
SD_LAT_MIN, SD_LAT_MAX =  32.3,  33.5
SD_LON_MIN, SD_LON_MAX = -118.3, -117.0

# ── time splits ───────────────────────────────────────────────────────────────
TRAIN_END = "2019-12-31"
VAL_END   = "2021-12-31"
# Test: 2022-01-01 onward (includes 2022-23 warm anomaly events)

LEAD_DAYS = [3, 7]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load & filter buoy data
# ═════════════════════════════════════════════════════════════════════════════

def load_buoys() -> pd.DataFrame:
    df = pd.read_parquet(BUOY_PATH)
    df = df[(df["sst"] > 5) & (df["sst"] < 40)]          # drop fill values
    df = df[
        (df["lat"] >= SD_LAT_MIN) & (df["lat"] <= SD_LAT_MAX) &
        (df["lon"] >= SD_LON_MIN) & (df["lon"] <= SD_LON_MAX)
    ]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering per station
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Build all temperature features. Climatology computed from training data only."""
    frames = []

    for sid, grp in df.groupby("station_id"):
        g = grp.set_index("date").sort_index().copy()

        # Rolling means
        g["sst_3d"]  = g["sst"].rolling(3,  min_periods=2).mean()
        g["sst_7d"]  = g["sst"].rolling(7,  min_periods=4).mean()
        g["sst_30d"] = g["sst"].rolling(30, min_periods=15).mean()

        # 7-day linear slope (°C / day)
        def _slope(x: pd.Series) -> float:
            if x.isna().sum() > len(x) // 2:
                return np.nan
            t = np.arange(len(x))
            m = ~np.isnan(x.values)
            if m.sum() < 4:
                return np.nan
            return float(np.polyfit(t[m], x.values[m], 1)[0])

        g["sst_slope_7d"] = g["sst"].rolling(7, min_periods=4).apply(_slope, raw=False)

        # Lag features
        for lag in [1, 3, 7, 14]:
            g[f"sst_lag{lag}"] = g["sst"].shift(lag)

        # Seasonal encoding
        g["month_sin"] = np.sin(2 * np.pi * g.index.month / 12)
        g["month_cos"] = np.cos(2 * np.pi * g.index.month / 12)
        g["doy_sin"]   = np.sin(2 * np.pi * g.index.dayofyear / 365)
        g["doy_cos"]   = np.cos(2 * np.pi * g.index.dayofyear / 365)

        # Day-of-year climatology (from training period only — no leakage)
        train_mask = g.index <= train_end
        doy_clim = (
            g.loc[train_mask, "sst"]
            .groupby(g.loc[train_mask].index.dayofyear)
            .mean()
        )
        g["sst_clim"]  = g.index.dayofyear.map(doy_clim)
        g["sst_anom"]  = g["sst"] - g["sst_clim"]

        # Spatial
        g["lat"] = grp["lat"].iloc[0]
        g["lon"] = grp["lon"].iloc[0]
        g["station_id"] = sid

        g = g.reset_index()
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. MHW labeling (Hobday criterion — training data threshold only)
# ═════════════════════════════════════════════════════════════════════════════

def label_mhw(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """
    Per station: MHW = SST > 90th-pct climatological threshold for 5+ consecutive days.
    Threshold is computed from training data only.
    """
    results = []
    for sid, grp in df.groupby("station_id"):
        g = grp.sort_values("date").copy()

        # Compute 90th percentile from training data
        train = g[g["date"] <= train_end]["sst"].dropna()
        thresh = float(train.quantile(0.90)) if len(train) > 30 else g["sst"].quantile(0.90)

        # Mark days exceeding threshold
        above = g["sst"] > thresh

        # Run-length encode: label runs of consecutive True days
        run_id = (above != above.shift()).cumsum()
        run_len = above.groupby(run_id).transform("sum")
        g["mhw"] = (above & (run_len >= 5)).astype(int)
        g["sst_thresh"] = thresh
        results.append(g)

    return pd.concat(results, ignore_index=True)


def make_targets(df: pd.DataFrame, leads: list[int]) -> pd.DataFrame:
    """target_Nd[t] = 1 if any of mhw[t+1]…mhw[t+N] is 1."""
    frames = []
    for sid, grp in df.groupby("station_id"):
        g = grp.sort_values("date").copy()
        for lead in leads:
            # Stack mhw shifted 1..lead days forward, take row-wise max
            shifted = pd.concat(
                [g["mhw"].shift(-i) for i in range(1, lead + 1)], axis=1
            )
            g[f"target_{lead}d"] = shifted.max(axis=1).fillna(0).astype(int)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# 4. CalCOFI subsurface features (nearest station, forward-filled to daily)
# ═════════════════════════════════════════════════════════════════════════════

def load_calcofi_features(buoy_df: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_parquet(CALCOFI_PATH)
    cal = cal[
        (cal["lat"] >= SD_LAT_MIN) & (cal["lat"] <= SD_LAT_MAX) &
        (cal["lon"] >= SD_LON_MIN) & (cal["lon"] <= SD_LON_MAX)
    ]

    # Surface (0–20m) and subsurface (40–60m) aggregates per cruise
    surface = cal[cal["depth_m"] <= 20].groupby("date").agg(
        calcofi_surf_temp=("temp_c", "mean"),
        calcofi_surf_sal=("salinity", "mean"),
        calcofi_chla=("chlorophyll_a", "mean"),
    ).reset_index()

    sub = cal[(cal["depth_m"] >= 40) & (cal["depth_m"] <= 60)].groupby("date").agg(
        calcofi_sub_temp=("temp_c", "mean"),
        calcofi_sub_sal=("salinity", "mean"),
    ).reset_index()

    cal_daily = surface.merge(sub, on="date", how="outer").sort_values("date")
    cal_daily["date"] = pd.to_datetime(cal_daily["date"])

    # Merge onto buoy dates, forward-fill (CalCOFI is quarterly)
    date_range = pd.DataFrame(
        {"date": pd.date_range(buoy_df["date"].min(), buoy_df["date"].max(), freq="D")}
    )
    cal_daily = date_range.merge(cal_daily, on="date", how="left")
    cal_daily = cal_daily.sort_values("date").ffill().bfill()
    return cal_daily


# ═════════════════════════════════════════════════════════════════════════════
# 5. Kelp features (forward-filled to daily, nearest SD sites)
# ═════════════════════════════════════════════════════════════════════════════

def load_kelp_features(buoy_df: pd.DataFrame) -> pd.DataFrame:
    canopy = pd.read_parquet(KELP_CANOPY)
    density = pd.read_parquet(KELP_DENSITY)

    # Filter SD sites
    sd_sites = ["Point Loma", "La Jolla"]
    canopy = canopy[canopy["site"].isin(sd_sites)]
    sd_den_sites = ["Ocean Beach", "Mission Beach", "Pacific Beach", "La Jolla Cove", "La Jolla Shores"]
    density = density[density["site"].isin(sd_den_sites)]

    canopy_daily = canopy.groupby("date").agg(
        kelp_canopy_km2=("canopy_extent_km2", "mean")
    ).reset_index()

    density_daily = density.groupby("date").agg(
        kelp_density=("frond_density_per_m2", "mean"),
        kelp_substrate=("substrate_coverage_pct", "mean"),
    ).reset_index()

    date_range = pd.DataFrame(
        {"date": pd.date_range(buoy_df["date"].min(), buoy_df["date"].max(), freq="D")}
    )
    kelp = date_range.merge(canopy_daily, on="date", how="left")
    kelp = kelp.merge(density_daily, on="date", how="left")
    kelp["date"] = pd.to_datetime(kelp["date"])
    kelp = kelp.sort_values("date").ffill().bfill()

    # Canopy change rate (quarter-over-quarter)
    kelp["kelp_canopy_chg"] = kelp["kelp_canopy_km2"].pct_change(90).fillna(0)

    return kelp


# ═════════════════════════════════════════════════════════════════════════════
# 6. Train + evaluate
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "sst", "sst_anom", "sst_3d", "sst_7d", "sst_30d", "sst_slope_7d",
    "sst_lag1", "sst_lag3", "sst_lag7", "sst_lag14",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "lat", "lon",
    "calcofi_surf_temp", "calcofi_surf_sal", "calcofi_chla",
    "calcofi_sub_temp", "calcofi_sub_sal",
    "kelp_canopy_km2", "kelp_canopy_chg", "kelp_density", "kelp_substrate",
]


def _metrics(y_true, y_prob, label: str) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Find threshold maximising F1 on this split
    prec_all, rec_all, thresholds = precision_recall_curve(y_true, y_prob)
    f1_all = np.where((prec_all + rec_all) > 0,
                      2 * prec_all * rec_all / (prec_all + rec_all), 0)
    best_idx = int(np.argmax(f1_all[:-1]))  # last element has no threshold
    best_thresh = float(thresholds[best_idx])

    y_pred = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # Precision at 80% recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    idx = np.searchsorted(-rec, -0.80)
    p_at_80r = float(prec[min(idx, len(prec) - 1)])

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  AUC-ROC              : {auc:.4f}")
    print(f"  Brier Score          : {brier:.4f}")
    print(f"  Precision@80%Recall  : {p_at_80r:.4f}")
    print(f"  MHW prevalence       : {y_true.mean():.2%}")
    print(f"  Optimal threshold        : {best_thresh:.3f}")
    print(f"\n  Confusion Matrix (optimal threshold):")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['no MHW','MHW'])}")

    return {"auc": auc, "brier": brier, "p_at_80r": p_at_80r}


def train_and_eval(df: pd.DataFrame, target_col: str, label: str) -> tuple[lgb.LGBMClassifier, dict]:
    feats = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=[target_col])

    train = df[df["date"] <= TRAIN_END]
    val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test  = df[df["date"] > VAL_END]

    print(f"\n  Split sizes — train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")

    X_tr, y_tr = train[feats], train[target_col].astype(int)
    X_val, y_val = val[feats], val[target_col].astype(int)
    X_te, y_te = test[feats], test[target_col].astype(int)

    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=pos_w,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Val metrics
    print("\n  [VALIDATION 2020–2021]")
    val_m = _metrics(y_val, clf.predict_proba(X_val)[:, 1], f"{label} — Val")

    # Test metrics
    print("\n  [TEST 2022–2023]")
    test_m = _metrics(y_te, clf.predict_proba(X_te)[:, 1], f"{label} — Test")

    # Feature importance plot
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Top features — {label}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    slug = target_col.replace("target_", "")
    fig.savefig(OUT_DIR / f"feature_importance_{slug}.png", dpi=120)
    plt.close()

    # Calibration plot
    prob_pred = clf.predict_proba(X_te)[:, 1]
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(prob_pred, bins) - 1
    cal_x, cal_y = [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() > 5:
            cal_x.append(prob_pred[mask].mean())
            cal_y.append(y_te.values[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.scatter(cal_x, cal_y, color="steelblue", s=60, label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed MHW fraction")
    ax.set_title(f"Calibration — {label}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"calibration_{slug}.png", dpi=120)
    plt.close()

    return clf, {**test_m, "val_auc": val_m["auc"], "n_features": len(feats)}


# ═════════════════════════════════════════════════════════════════════════════
# 7. Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading buoy data...")
    buoy = load_buoys()
    print(f"  {len(buoy):,} rows, {buoy['station_id'].nunique()} stations")

    print("Engineering features...")
    buoy = engineer_features(buoy, TRAIN_END)

    print("Labeling MHW events...")
    buoy = label_mhw(buoy, TRAIN_END)
    buoy = make_targets(buoy, LEAD_DAYS)
    print(f"  Overall MHW prevalence : {buoy['mhw'].mean():.2%}")

    print("Loading CalCOFI features...")
    cal_feats = load_calcofi_features(buoy)
    buoy = buoy.merge(cal_feats, on="date", how="left")

    print("Loading kelp features...")
    kelp_feats = load_kelp_features(buoy)
    buoy = buoy.merge(kelp_feats, on="date", how="left")

    results = {}
    models = {}

    for lead in LEAD_DAYS:
        target = f"target_{lead}d"
        label  = f"San Diego {lead}-day MHW forecast"
        clf, m = train_and_eval(buoy, target, label)
        results[target] = m
        models[target]  = clf

    # Save models
    import joblib
    for target, clf in models.items():
        path = OUT_DIR / f"lgbm_{target}.pkl"
        joblib.dump(clf, path)
        print(f"\nModel saved: {path}")

    # Save metrics
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # ── Backtest time-series plot (2022-2023 test period) ──────────────────
    feats = [c for c in FEATURE_COLS if c in buoy.columns]
    test_df = buoy[buoy["date"] > VAL_END].copy().dropna(subset=feats)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, lead in zip(axes, LEAD_DAYS):
        target = f"target_{lead}d"
        clf = models[target]
        sub = test_df.dropna(subset=[target])
        probs = clf.predict_proba(sub[feats])[:, 1]

        # Aggregate to daily mean across stations
        daily = (
            pd.DataFrame({"date": sub["date"], "prob": probs, "actual": sub[target]})
            .groupby("date")
            .mean()
        )

        ax.fill_between(daily.index, daily["actual"], alpha=0.25,
                        color="red", label="Actual MHW (any station)")
        ax.plot(daily.index, daily["prob"], color="steelblue",
                lw=1.5, label=f"Model P(MHW in {lead}d)")
        ax.axhline(0.3, color="orange", lw=1, ls="--", label="Alert threshold (0.30)")
        ax.set_ylabel("Probability")
        ax.set_title(f"San Diego {lead}-day MHW forecast — 2022–2023 backtest  (AUC={results[target]['auc']:.3f})")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 1)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "backtest_2022_2023.png", dpi=150)
    plt.close()
    print(f"\nBacktest plot saved: {OUT_DIR / 'backtest_2022_2023.png'}")

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    for target, m in results.items():
        print(f"  {target}: AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  P@80R={m['p_at_80r']:.4f}")
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
