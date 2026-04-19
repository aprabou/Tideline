# Tideline — Demo Script

**Total time: 3 minutes**

---

## 0:00 — Hook (20 s)

> "In 2021, the Pacific Northwest heat dome killed over a billion marine animals in 48 hours.
> Aquaculture farms had zero warning.
> Tideline changes that."

---

## 0:20 — What is a marine heatwave? (30 s)

Show the Marimo notebook open to the SST anomaly histogram.

> "A marine heatwave is when sea surface temperature exceeds the 90th-percentile
> climatological baseline for at least five consecutive days.
> This is the Hobday 2016 definition — the scientific standard.
> Tideline ingests NOAA satellite data daily, labels every 0.25° grid cell,
> and trains an XGBoost classifier to predict the next 14 days."

Point at the seasonal prevalence bar chart.

> "MHWs peak in late summer, but the 2021 event hit in June — exactly when a model matters."

---

## 0:50 — Live dashboard (60 s)

Switch to the React dashboard (`localhost:5173`).

> "This is the Tideline dashboard.
> Each cell is coloured by the probability of a marine heatwave today."

Drag the time slider forward to day +7.

> "Scrub forward — probabilities shift as the forecast horizon extends.
> Click a cell off the Monterey Bay coast."

RegionDetail panel appears.

> "You see the full 14-day probability curve.
> This farm operator can see that by next Friday, there's an 80 % chance of a severe event —
> enough time to move stock to deeper, cooler water."

---

## 1:50 — Architecture (30 s)

Show `infra/architecture.md` in rendered Mermaid.

> "Under the hood: OISST NetCDF lands in S3 bronze.
> Features and labels land in silver.
> The XGBoost model is a 40 MB artifact served by FastAPI on Lambda.
> Cold start under 3 seconds, inference under 100 ms."

---

## 2:20 — Business case (30 s)

> "Three customer segments:
> Aquaculture — $1,200/month SaaS per farm, alert integrations.
> Fisheries — API licensing to quota management platforms.
> Conservation — grant-funded MPA monitoring dashboards.
>
> Global aquaculture is a $300B industry.
> A 1-week warning on a heatwave event can save a farm season."

---

## 2:50 — Ask (10 s)

> "We're looking for a partner in the aquaculture space to run a 90-day pilot.
> If you work with salmon or shellfish operators on the West Coast, let's talk."

---

**Questions to anticipate:**

- *How accurate is the model?* Cross-validated AUC ~0.82 on held-out time periods; see `models/meta.json`.
- *What's the latency?* Grid forecast for West Coast (≈200 cells) in < 500 ms end-to-end.
- *Real-time or batch?* Daily batch retrain; real-time SST lookup from feature store is on the roadmap.
- *Why XGBoost and not a deep learning model?* Interpretability (SHAP) matters to regulators; XGBoost trains in minutes on CPU, ships in 40 MB.
