# Tideline — System Architecture

## High-level overview

```mermaid
flowchart TB
    subgraph Data Sources
        OISST["NOAA OISST\n0.25°/day NetCDF"]
        NDBC["NDBC Buoys\nhourly CSV"]
        CalCOFI["CalCOFI CTD\nquarterly CSV"]
    end

    subgraph Pipeline ["Batch Pipeline (daily cron / EventBridge)"]
        direction LR
        I1["ingest_oisst.py"]
        I2["ingest_buoys.py"]
        I3["ingest_calcofi.py"]
        F["features.py"]
        L["labels.py"]
        T["train.py"]

        OISST --> I1
        NDBC  --> I2
        CalCOFI --> I3
        I1 & I2 & I3 --> F --> L --> T
    end

    subgraph Storage ["AWS Storage"]
        S3_raw["S3\ndata/raw (Bronze)"]
        S3_silver["S3\ndata/silver (Features)"]
        S3_model["S3\nmodels/xgb_tideline.json"]
    end

    subgraph Serving ["Inference Layer"]
        LM["Lambda Container\n(FastAPI + Mangum)"]
        AG["API Gateway\nHTTPS REST"]
    end

    subgraph Frontend ["Dashboard (React / Vite)"]
        FM["ForecastMap\n(Deck.gl)"]
        TS["TimeSlider"]
        RD["RegionDetail"]
        API_CLIENT["tideline.ts\nAPI client"]
    end

    Pipeline --> S3_raw --> S3_silver
    T --> S3_model --> LM
    LM <--> AG
    AG <--> API_CLIENT
    API_CLIENT --> FM & TS & RD
```

## Data flow

| Stage | Format | Location | Updated |
|---|---|---|---|
| Raw OISST | NetCDF (.nc) | S3 `raw/oisst/` | Daily |
| Raw buoys | Parquet | S3 `raw/buoys/` | Hourly |
| Raw CalCOFI | Parquet | S3 `raw/calcofi/` | Quarterly |
| Silver features | Parquet | S3 `silver/features.parquet` | Daily |
| Labeled features | Parquet | S3 `silver/features_labeled.parquet` | Daily |
| Model artifact | JSON (XGBoost native) | S3 `models/` | Weekly retrain |

## Inference latency budget

| Step | Target |
|---|---|
| API Gateway → Lambda cold start | < 3 s |
| Feature lookup | < 50 ms |
| XGBoost predict (14-day grid) | < 100 ms |
| Total P95 | < 500 ms |

## Security

- Lambda IAM role: read-only S3 access to `models/` prefix
- API Gateway: API key auth for production; open for hackathon demo
- No SST data persisted on Lambda (stateless)
