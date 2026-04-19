import { useEffect, useState } from "react";
import { getPointForecast, ForecastResponse, CATEGORY_COLORS } from "../api/tideline";

interface RegionDetailProps {
  lat: number;
  lon: number;
  date: string;
}

export default function RegionDetail({ lat, lon, date }: RegionDetailProps) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setForecast(null);
    setError(null);
    getPointForecast(lat, lon, date)
      .then(setForecast)
      .catch((err: Error) => setError(err.message));
  }, [lat, lon, date]);

  if (error) return <div style={{ color: "#f66", padding: 8, fontSize: 12 }}>Error: {error}</div>;
  if (!forecast) return <div style={{ color: "#aaa", padding: 8, fontSize: 12 }}>Loading…</div>;

  const maxProb = Math.max(...forecast.forecasts.map((f) => f.prob_mhw));

  return (
    <div style={{ padding: "4px 0" }}>
      <h3 style={{ margin: "0 0 12px", fontSize: 13, color: "#ccc" }}>
        {lat.toFixed(2)}°N, {Math.abs(lon).toFixed(2)}°W
      </h3>

      {forecast.forecasts.map((f) => {
        const pct = Math.round(f.prob_mhw * 100);
        const color = CATEGORY_COLORS[f.category];
        return (
          <div key={f.lead_days} style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4, color: "#ccc" }}>
              <span>{f.lead_days}-day forecast</span>
              <span style={{ color, fontWeight: 600 }}>{pct}% · {f.category}</span>
            </div>
            <div style={{ background: "#21262d", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.4s" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#666", marginTop: 2 }}>
              <span>CI: {Math.round(f.ci_low * 100)}%–{Math.round(f.ci_high * 100)}%</span>
              <span>{f.target_date}</span>
            </div>
          </div>
        );
      })}

      <hr style={{ border: "none", borderTop: "1px solid #21262d", margin: "12px 0" }} />
      <div style={{ fontSize: 11, color: "#666" }}>
        <div style={{ marginBottom: 4, fontWeight: 600, color: "#888" }}>Top drivers</div>
        {Object.entries(forecast.forecasts.find(f => f.lead_days === 7)?.top_features ?? {}).map(([k, v]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
            <span>{k}</span>
            <span style={{ color: "#aaa" }}>{(v as number).toFixed(2)}</span>
          </div>
        ))}
      </div>

      <p style={{ fontSize: 11, color: "#555", marginTop: 10 }}>
        Peak: <strong style={{ color: "#fff" }}>{Math.round(maxProb * 100)}%</strong>
      </p>
    </div>
  );
}
