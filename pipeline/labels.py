"""
labels.py — Generate marine heatwave labels using Hobday et al. (2016) criteria.

Definition (Hobday et al. 2016, Prog. Oceanogr.):
  A marine heatwave is a period when SST exceeds the 90th-percentile of the
  climatological baseline for that calendar day for at least 5 consecutive days.
  Two events separated by ≤ 2 days are merged into one.

Reference: Hobday, A.J. et al. (2016). A hierarchical approach to defining
  marine heatwaves. Progress in Oceanography, 141, 227–238.
  https://doi.org/10.1016/j.pocean.2015.12.014
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SILVER_DIR = Path("data/silver")
CLIM_PERCENTILE = 90  # threshold percentile
MIN_DURATION = 5      # minimum consecutive days
GAP_TOLERANCE = 2     # days gap allowed between events to merge


def compute_clim_threshold(
    df: pd.DataFrame,
    sst_col: str = "sst",
    time_col: str = "time",
    pct: int = CLIM_PERCENTILE,
    window: int = 11,  # ±5 day window for smoothed percentile
) -> pd.DataFrame:
    """
    Compute smoothed 90th-percentile climatology per day-of-year.

    Returns df with a new column `thresh` for each row's calendar day.
    """
    df = df.copy()
    df["doy"] = pd.to_datetime(df[time_col]).dt.dayofyear

    # Build a DOY → percentile lookup using a sliding window across all years
    doy_thresholds: dict[int, float] = {}
    for doy in range(1, 367):
        # Collect all SST values within ±window/2 days of this DOY
        doys = [(doy - window // 2 + i - 1) % 365 + 1 for i in range(window)]
        mask = df["doy"].isin(doys)
        vals = df.loc[mask, sst_col].dropna()
        doy_thresholds[doy] = float(np.nanpercentile(vals, pct)) if len(vals) > 0 else np.nan

    df["thresh"] = df["doy"].map(doy_thresholds)
    return df


def label_mhw(
    df: pd.DataFrame,
    sst_col: str = "sst",
    thresh_col: str = "thresh",
    time_col: str = "time",
    min_duration: int = MIN_DURATION,
    gap_tol: int = GAP_TOLERANCE,
) -> pd.DataFrame:
    """
    Add a boolean `mhw` column — True if the day is part of a marine heatwave.

    Operates on a single location series (expects df sorted by time).
    """
    df = df.copy().sort_values(time_col).reset_index(drop=True)
    df["_above"] = df[sst_col] > df[thresh_col]

    # Mark runs of consecutive exceedance days
    df["_run_id"] = (df["_above"] != df["_above"].shift()).cumsum()
    run_lengths = df.groupby("_run_id")["_above"].transform("sum")
    df["mhw"] = (df["_above"]) & (run_lengths >= min_duration)

    # Merge events separated by ≤ gap_tol days
    if gap_tol > 0:
        in_event = df["mhw"].to_numpy().copy()
        for i in range(1, len(in_event)):
            if not in_event[i] and i + gap_tol < len(in_event):
                # Check if gap is surrounded by event days
                gap_end = min(i + gap_tol + 1, len(in_event))
                if in_event[i - 1] and any(in_event[i + 1 : gap_end]):
                    in_event[i] = True
        df["mhw"] = in_event

    df.drop(columns=["_above", "_run_id"], inplace=True)
    return df


def build_labels(
    features_path: Path = SILVER_DIR / "features.parquet",
    out_path: Path = SILVER_DIR / "features_labeled.parquet",
) -> pd.DataFrame:
    """Load silver features, compute thresholds, attach MHW labels."""
    df = pd.read_parquet(features_path)
    df = compute_clim_threshold(df)
    df = (
        df.groupby(["lat", "lon"], group_keys=False)
        .apply(label_mhw)
        .reset_index(drop=True)
    )
    mhw_rate = df["mhw"].mean()
    print(f"MHW prevalence: {mhw_rate:.2%} of grid-cell-days")
    df.to_parquet(out_path, index=False)
    print(f"Labeled dataset saved: {out_path} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    df = build_labels()
    print(df[["time", "lat", "lon", "sst", "thresh", "mhw"]].head(20))
