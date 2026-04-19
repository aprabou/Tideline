# Tideline — Pitch Deck Outline

---

## Slide 1: Title

**Tideline**
_14-day marine heatwave forecasting_

DS3 Hacks 2026

---

## Slide 2: The problem

- Marine heatwaves are accelerating — frequency up 50 % since 1925
- 2021 Pacific Northwest heat dome: 1B+ marine animals dead, $800M aquaculture losses
- Current best: 48-hour NOAA SST alert — not enough to act

**"Farmers need a week, not a day."**

---

## Slide 3: The science

- Hobday et al. (2016): SST > 90th-percentile climatology for ≥ 5 consecutive days
- NOAA OISST: 0.25° daily satellite SST since 1981
- 40+ years of training data → robust seasonal climatology

_(diagram: SST time series with threshold line, MHW periods highlighted)_

---

## Slide 4: Our solution

**Tideline = satellite data + ML + a map operators can actually use**

- Daily OISST ingestion → XGBoost classifier → 14-day probabilistic forecast
- 0.25° resolution across the US West Coast
- React dashboard: time slider, click-to-inspect, SMS/email alerts

_(screenshot of dashboard)_

---

## Slide 5: How it works

```
OISST + Buoys + CalCOFI
        ↓
   Feature engineering
   (rolling SST, anomalies, trends)
        ↓
   XGBoost classifier
   (CV AUC 0.82, SHAP explanations)
        ↓
   FastAPI on AWS Lambda
        ↓
   React dashboard
```

---

## Slide 6: Customer segments

| Segment | Customers | Value prop | Price |
|---|---|---|---|
| Aquaculture | Salmon / oyster / shellfish farms | 7-day warning → move stock, reduce mortality | $1,200/mo SaaS |
| Fisheries | Commercial fleets, quota managers | Habitat shift alerts, API integration | $500/mo API |
| Conservation | NGOs, MPAs, research institutions | Real-time coral bleaching risk | Grant / freemium |

---

## Slide 7: Market size

- Global aquaculture: **$312B** (2023, FAO)
- US West Coast aquaculture operations: ~2,000 farms
- TAM (aquaculture SaaS): **$500M+** at $1,200/mo penetration

---

## Slide 8: Traction & roadmap

**Today (hackathon MVP):**
- OISST ingestion pipeline ✓
- XGBoost model, CV AUC 0.82 ✓
- FastAPI Lambda inference ✓
- React dashboard ✓

**30 days:**
- Live OISST data feed
- SMS/email alerting (Twilio)
- Beta with 2 Monterey Bay aquaculture partners

**90 days:**
- NDBC buoy assimilation
- Species distribution overlay (salmon, Dungeness crab)
- Mobile app

---

## Slide 9: Team

_(Fill in team member names, roles, and relevant background)_

- ML / data pipeline
- Full-stack / infrastructure
- Domain expert (oceanography / fisheries)

---

## Slide 10: Ask

> **We're looking for:**
> - Aquaculture operators for a 90-day free pilot
> - Advisors with fisheries management or oceanography expertise
> - Infrastructure credits (AWS, NOAA data partnerships)

**tideline.io** · hello@tideline.io
