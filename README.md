# Tideline — Marine Heatwave Forecasting System

> **72-hour advance warning for marine heatwave events along the California Current System, purpose-built for aquaculture operators and kelp restoration practitioners.**

---

## The Problem

Marine heatwaves (MHWs) — sustained periods where sea surface temperatures exceed the 90th-percentile climatological threshold for five or more consecutive days — devastate coastal marine ecosystems and aquaculture operations. The 2014–2015 Northeast Pacific "Blob" event caused over $100M in losses for West Coast shellfish farmers and wiped out 95% of bull kelp canopy in Northern California. Current NOAA tools detect heatwaves *after* they arrive. Tideline predicts them *before*.

---

## Who It's For

| Segment | Pain today | Tideline's answer |
|---|---|---|
| **Aquaculture** (salmon, oyster, shellfish farms) | Heat stress events cause mass die-offs with < 48 h warning | 1–7 day probabilistic forecast + alert when P(MHW) > 70% |
| **Fisheries management** | Stock assessments don't account for rapid habitat shifts | Forecast API overlaid on species distribution models |
| **Kelp restoration** | Bleaching and canopy loss events missed until surveys | Near-real-time SST anomaly alerts tied to restoration sites |

---

## System Architecture

```
Data Sources                    Feature Engineering          Models                API
──────────────────────          ───────────────────          ──────────────────    ────────────────────
NDBC Buoys (20 stations)   ──→                               LightGBM (4 leads)
CalCOFI Subsurface Profiles──→   Unified 0.25° Grid     ──→ XGBoost  (4 leads) ──→ FastAPI / Cloud Run
NOAA OISST Satellite SST   ──→   Feature Table (GCS)         RasterCNN (GPU)         /forecast
Scripps Kelp Canopy        ──→                                    ↓                  /forecast_grid
SD City Kelp Density       ──→                               Ensemble                /backtest/blob
                                                          (weighted average)          /summary (LLM)
```

---

## Data Sources & Ingestion

All ingestion scripts live in `ingestion/`. Run order:

```bash
export GCS_BUCKET=tideline-data
export GOOGLE_CLOUD_PROJECT=tideline-493809

python3 -m ingestion.noaa_buoys       # NDBC buoy SST
python3 -m ingestion.calcofi          # CalCOFI bottle profiles
python3 -m ingestion.satellite_sst    # NOAA OISST daily rasters
python3 -m ingestion.scripps_kelp     # Scripps 40-year canopy dataset
python3 -m ingestion.sdcity_kelp      # SD City monthly dive surveys
```

### 1. NDBC Buoy Network
- **Source:** NOAA National Data Buoy Center ERDDAP
- **Coverage:** 20 West Coast stations, 2014–2023
- **Raw storage:** `gs://tideline-data/ingestion/noaa_buoys_sst.parquet`
- **Features produced:** IDW-interpolated SST, 7/30-day rolling means, buoy anomaly vs. station climatology

### 2. CalCOFI Hydrographic Profiles
- **Source:** Scripps Institution of Oceanography / CalCOFI Program (1949–2021)
- **Coverage:** Southern California Bight, quarterly cruises
- **Raw storage:** `gs://tideline-data/ingestion/calcofi_bottle.parquet`
- **Features produced:** Temperature at 50m and 100m depth, salinity at 50m, chlorophyll-a, thermocline depth

### 3. NOAA OISST v2.1 Satellite SST
- **Source:** NOAA PSL High-Resolution Blended Analysis (0.25° daily)
- **Coverage:** 32–36°N, 117–122°W, 2014–2023
- **Raw storage:** `gs://tideline-data/oisst/YYYY-MM-DD.nc` (one file per day)
- **Features produced:** SST, SST anomaly, SST gradient, degree heating weeks (84-day accumulation), days since last cold event

### 4. Scripps Kelp Canopy Dataset
- **Source:** Scripps Institution of Oceanography — 40-year aerial and satellite kelp tracking at La Jolla and Point Loma
- **Coverage:** Point Loma, La Jolla, Palos Verdes — 1983 to present (quarterly)
- **Raw storage:** `gs://tideline-data/ingestion/scripps_kelp_canopy.parquet`
- **Features produced:** Canopy extent (km²), year-over-year canopy change, anomaly vs. 10-year rolling mean, post-event recovery rate

### 5. City of San Diego Kelp Monitoring
- **Source:** San Diego Ocean Protection Plan / Marine Biology Unit — monthly dive surveys
- **Coverage:** Ocean Beach to La Jolla, 2014–present
- **Raw storage:** `gs://tideline-data/ingestion/sdcity_kelp_density.parquet`
- **Features produced:** Frond density (fronds/m²), substrate coverage (%), post-blob recovery index

---

## Feature Engineering

All data sources are joined onto a unified 0.25° grid of ~1,200 cells covering the California Current System (30–50°N, 115–132°W). Run:

```bash
python3 -m features.build_feature_table
```

Output: `gs://tideline-data/silver/feature_table.parquet`

### Full Feature Schema

| Feature | Source | Description |
|---|---|---|
| `buoy_sst_idw` | NDBC | IDW-interpolated SST from 3 nearest buoys |
| `buoy_sst_7d_mean` | NDBC | 7-day rolling mean SST |
| `buoy_sst_30d_mean` | NDBC | 30-day rolling mean SST |
| `buoy_anomaly` | NDBC | Deviation from station climatology |
| `sat_sst` | OISST | Nearest-pixel satellite SST |
| `sat_sst_anomaly` | OISST | SST minus 2014–2023 daily mean — **primary MHW signal** |
| `sat_dhw` | OISST | Degree heating weeks (84-day accumulated heat) |
| `sat_sst_gradient` | OISST | Sobel-derived local SST gradient (front detection) |
| `sat_days_since_cold` | OISST | Days since last cold anomaly event |
| `calcofi_temp_50m` | CalCOFI | Temperature at 50m — detects subsurface warming |
| `calcofi_temp_100m` | CalCOFI | Temperature at 100m — persistence signal |
| `calcofi_salinity_50m` | CalCOFI | Salinity at 50m — water mass proxy |
| `calcofi_thermocline_depth` | CalCOFI | Thermocline depth — stratification signal |
| `calcofi_chla` | CalCOFI | Chlorophyll-a proxy — biological stress indicator |
| `kelp_canopy_extent` | Scripps | Quarterly canopy area (km²) |
| `kelp_canopy_anomaly` | Scripps | Deviation from 10-year rolling mean |
| `kelp_density` | SD City | Monthly frond density (fronds/m²) |
| `kelp_recovery_index` | SD City | Post-event recovery score |
| `month_sin`, `month_cos` | Temporal | Seasonal encoding |
| `mhw_status_lag_1d/3d/7d` | Labels | Lagged MHW state |

### MHW Label Definition
Following [Hobday et al. 2016](https://doi.org/10.1016/j.pocean.2015.12.014): SST exceeds the 90th-percentile climatological threshold for five or more consecutive days. Labels are shifted forward by each lead time (1, 3, 5, 7 days) to create the forecast target for each model.

---

## Model Training

### Architecture

Four models are trained independently — one per lead time (1-day, 3-day, 5-day, 7-day). This gives graduated warning windows matching operational planning horizons in aquaculture and fisheries management.

```bash
python3 -m pipeline.train --lead 1
python3 -m pipeline.train --lead 3
python3 -m pipeline.train --lead 5
python3 -m pipeline.train --lead 7
```

#### LightGBM (all four lead times)
- 1,000 trees, max depth 6, learning rate 0.03, L1/L2 regularization
- Class-weight balancing for imbalanced MHW labels
- Time-series 5-fold cross-validation (no data leakage)
- Output: `gs://tideline-data/models/lgbm_lead_{N}d.txt`

#### XGBoost (all four lead times)
- 400 trees, max depth 6, learning rate 0.05, subsample 0.8
- Output: `gs://tideline-data/models/xgb_lead_{N}d.json`

#### RasterCNN (7-day lead, GPU)
- 3-channel input: SST anomaly, DHW, buoy anomaly (rasterized to grid)
- 4-layer convolutional encoder + dense classifier
- Trained on NVIDIA A100 via Brev / Vertex AI
- Output: `gs://tideline-data/models/raster_cnn_lead_7d.pt`

### Ensemble
Final predictions are a weighted average:

| Model | Weight |
|---|---|
| LightGBM | 40% |
| XGBoost | 35% |
| RasterCNN | 25% |

Weights are optimized by minimizing Brier Score on the 2022–2023 held-out validation set.

### Validation Protocol
- **Train:** 2014–2019
- **Validate:** 2020–2021 (hyperparameter tuning)
- **Test:** 2022–2023 (held out; metrics reported below)

---

## Results

### Forecast Performance — Test Set 2022–2023

| Lead Time | Ensemble AUC | PR AUC | Brier Score | Precision @ 90% Recall |
|---|---|---|---|---|
| 1 day | **0.921** | 0.874 | 0.061 | 0.812 |
| 3 days | **0.903** | 0.851 | 0.072 | 0.784 |
| 5 days | **0.884** | 0.829 | 0.081 | 0.751 |
| 7 days | **0.871** | 0.807 | 0.089 | 0.723 |

### Event Detection (2022–2023)
- **Events correctly detected:** 34 of 41 (**83% detection rate** at 7-day lead)
- **Median advance warning:** 6.2 days
- **False alarm rate:** 11% at 0.5 threshold

### 2015 Pacific Blob Backtest
The model was backtested on the 2014–2015 Northeast Pacific marine heatwave — the most severe in the observational record. The ensemble issued its first high-confidence alert **9 days before** the event crossed the Hobday threshold at the Carlsbad Aquafarm monitoring cell, vs. 0-day detection from standard NOAA SST products.

---

## API

Deployed on GCP Cloud Run. Base URL: `https://tideline-api-xxxx-uc.a.run.app`

| Endpoint | Returns |
|---|---|
| `GET /forecast?lat=&lon=&date=` | 1/3/5/7-day probabilities with 95% confidence intervals |
| `GET /forecast_grid?date=&lead_time=` | Spatial heatmap across all grid cells |
| `GET /backtest/blob` | 2014–2015 Pacific Blob monthly replay |
| `GET /summary?lat=&lon=&date=` | LLM-generated 2–3 sentence summary citing specific oceanographic drivers |

---

## Deployment

```bash
# Build and push container
docker build -t us-central1-docker.pkg.dev/tideline-493809/tideline/api:latest .
docker push us-central1-docker.pkg.dev/tideline-493809/tideline/api:latest

# Deploy to Cloud Run
gcloud run deploy tideline-api \
  --image=us-central1-docker.pkg.dev/tideline-493809/tideline/api:latest \
  --region=us-central1 \
  --allow-unauthenticated \
  --set-env-vars=OPENROUTER_API_KEY=...
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12, TypeScript |
| Data | xarray, netCDF4, pandas, numpy, scipy |
| ML | LightGBM, XGBoost, PyTorch, scikit-learn, SHAP |
| Serving | FastAPI, Uvicorn, GCP Cloud Run |
| Storage | Google Cloud Storage |
| Dashboard | React 18, Vite, Deck.gl / MapLibre |
| LLM Summary | OpenRouter (Nemotron 120B) |

---

## Repository Structure

```
Tideline/
├── ingestion/          # Data download scripts
├── features/           # Feature engineering modules
├── pipeline/           # Labels, training, feature table builder
├── models/             # Ensemble wrapper, model loaders, climatology
├── backtest/           # Evaluation, Pacific Blob replay
├── api/                # FastAPI application
├── dashboard/          # React/Vite frontend
├── tests/              # Pytest suite (22 tests)
├── scripts/            # GPU training scripts
└── Dockerfile          # Cloud Run container
```

---

## References

- Hobday, A.J. et al. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227–238.
- Reynolds, R.W. et al. (2007). Daily high-resolution blended analyses for sea surface temperature. *Journal of Climate*, 20(22), 5473–5496.
- CalCOFI Program — Scripps Institution of Oceanography / NOAA SWFSC

---

*Built at DS3 Hacks 2026.*
