"""
tideline_analysis.py — Marimo interactive notebook for Tideline EDA and model exploration.

Run with:  marimo edit notebooks/tideline_analysis.py
"""

import marimo as mo

__generated_with = "0.6.0"
app = mo.App(width="full")


@app.cell
def _():
    import marimo as mo
    mo.md("# 🌊 Tideline — Marine Heatwave Analysis")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np

    SILVER = Path("data/silver")
    labeled_path = SILVER / "features_labeled.parquet"

    if labeled_path.exists():
        df = pd.read_parquet(labeled_path)
        mo.md(f"Loaded **{len(df):,}** rows · {df['mhw'].mean():.2%} MHW prevalence")
    else:
        df = pd.DataFrame()
        mo.callout(
            mo.md("Run the pipeline first: `python pipeline/features.py && python pipeline/labels.py`"),
            kind="warn",
        )
    return df, pd, np, Path


@app.cell
def _(df, mo):
    if df.empty:
        mo.stop(True, mo.md("No data available."))

    # Summary stats table
    summary = df[["sst", "sst_anom", "rolling_14d_mean", "sst_trend_14d"]].describe().round(3)
    mo.ui.table(summary.reset_index(), label="Feature summary statistics")


@app.cell
def _(df, mo):
    import datetime

    date_slider = mo.ui.slider(
        start=0,
        stop=13,
        value=0,
        label="Forecast day offset",
    )
    date_slider


@app.cell
def _(df, mo, date_slider):
    if df.empty or "time" not in df.columns:
        mo.stop(True)

    import matplotlib.pyplot as plt

    # SST anomaly distribution: MHW vs non-MHW days
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mhw_anom = df.loc[df["mhw"] == 1, "sst_anom"].dropna()
    non_anom = df.loc[df["mhw"] == 0, "sst_anom"].dropna()

    axes[0].hist(non_anom.sample(min(5000, len(non_anom))), bins=50,
                 alpha=0.6, label="No MHW", color="#4c9edd")
    axes[0].hist(mhw_anom.sample(min(5000, len(mhw_anom))), bins=50,
                 alpha=0.7, label="MHW", color="#e04a2f")
    axes[0].set_xlabel("SST anomaly (°C)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SST anomaly distribution by MHW label")
    axes[0].legend()

    # MHW prevalence by month
    df["month"] = pd.to_datetime(df["time"]).dt.month
    monthly = df.groupby("month")["mhw"].mean()
    axes[1].bar(monthly.index, monthly.values, color="#e04a2f", alpha=0.8)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("MHW prevalence")
    axes[1].set_title("Seasonal MHW prevalence")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

    fig.tight_layout()
    mo.mpl.interactive(fig)


@app.cell
def _(mo):
    mo.md("""
    ## Model explanation

    SHAP values show which features drive each forecast.
    Run `python pipeline/train.py` to generate the model, then reload this notebook.
    """)


@app.cell
def _(mo):
    import json
    from pathlib import Path

    meta_path = Path("models/meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        mo.md(f"""
        | Metric | Value |
        |---|---|
        | CV AUC (mean) | {meta['cv_auc_mean']:.4f} |
        | CV AUC (std)  | {meta['cv_auc_std']:.4f} |
        | Horizon       | {meta['horizon_days']} days |
        | Training rows | {meta['n_training_rows']:,} |
        """)
    else:
        mo.callout(mo.md("Train the model first: `python pipeline/train.py`"), kind="info")


if __name__ == "__main__":
    app.run()
