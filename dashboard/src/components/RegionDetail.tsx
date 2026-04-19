/**
 * RegionDetail.tsx — 14-day probability chart for a selected lat/lon point.
 */

import { useEffect, useState } from "react";
import { format, parseISO } from "date-fns";
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

  if (error) return <div style={{ color: "#f66", padding: 8 }}>Error: {error}</div>;
  if (!forecast) return <div style={{ color: "#aaa", padding: 8 }}>Loading…</div>;

  const maxProb = Math.max(...forecast.forecasts.map((f) => f.prob_mhw));

  return (
    <div style={{ padding: "12px 0" }}>
      <h3 style={{ margin: "0 0 8px", fontSize: 14, color: "#fff" }}>
        {lat.toFixed(2)}°N, {Math.abs(lon).toFixed(2)}°W — 14-day forecast
      </h3>
      <div style={{ overflowX: "auto" }}>
        <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 100 }}>
          {forecast.forecasts.map((day) => (
            <div
              key={day.date}
              title={`${format(parseISO(day.date), "MMM d")}: ${(day.prob_mhw * 100).toFixed(0)}% (${day.category})`}
              style={{
                flex: "0 0 20px",
                height: `${Math.max(4, day.prob_mhw * 100)}px`,
                background: CATEGORY_COLORS[day.category],
                borderRadius: 2,
                cursor: "pointer",
                opacity: day.prob_mhw === maxProb ? 1 : 0.75,
              }}
            />
          ))}
        </div>
        <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
          {forecast.forecasts
            .filter((_, i) => i % 7 === 0)
            .map((day) => (
              <div key={day.date} style={{ flex: "0 0 20px", fontSize: 10, color: "#888" }}>
                {format(parseISO(day.date), "d")}
              </div>
            ))}
        </div>
      </div>
      <p style={{ fontSize: 12, color: "#aaa", marginTop: 8 }}>
        Peak probability: <strong style={{ color: "#fff" }}>{(maxProb * 100).toFixed(0)}%</strong>
      </p>
    </div>
  );
}
