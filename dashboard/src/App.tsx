import { useState } from "react";
import { format, addMonths } from "date-fns";
import ForecastMap from "./components/ForecastMap";
import BlobMap from "./components/BlobMap";
import TimeSlider from "./components/TimeSlider";
import RegionDetail from "./components/RegionDetail";
import { GridCell } from "./api/tideline";

const FORECAST_BASE = new Date("2023-08-01");
const BLOB_START = new Date("2014-01-01");

type Mode = "forecast" | "blob";

export default function App() {
  const [mode, setMode] = useState<Mode>("blob");
  const [offsetDays, setOffsetDays] = useState(0);
  const [blobMonth, setBlobMonth] = useState(11); // Jan 2015 peak
  const [selected, setSelected] = useState<GridCell | null>(null);

  const forecastDate = new Date(FORECAST_BASE);
  forecastDate.setDate(forecastDate.getDate() + offsetDays);
  const dateStr = format(forecastDate, "yyyy-MM-dd");

  const blobDate = addMonths(BLOB_START, blobMonth);
  const blobDateStr = format(blobDate, "yyyy-MM-01");

  const btnBase: React.CSSProperties = {
    padding: "6px 14px",
    borderRadius: 6,
    border: "1px solid #30363d",
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 600,
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0d1117", color: "#e6edf3", fontFamily: "'Inter', system-ui, sans-serif", display: "flex", flexDirection: "column" }}>
      <header style={{ padding: "14px 24px", borderBottom: "1px solid #21262d", display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.5px" }}>🌊 Tideline</span>
        <span style={{ fontSize: 13, color: "#8b949e" }}>Marine Heatwave Forecasting · West Coast USA</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <button
            style={{ ...btnBase, background: mode === "forecast" ? "#238636" : "#161b22", color: mode === "forecast" ? "#fff" : "#8b949e" }}
            onClick={() => setMode("forecast")}
          >
            Live Forecast
          </button>
          <button
            style={{ ...btnBase, background: mode === "blob" ? "#d73027" : "#161b22", color: mode === "blob" ? "#fff" : "#8b949e" }}
            onClick={() => setMode("blob")}
          >
            2014–15 Pacific Blob
          </button>
        </div>
      </header>

      <main style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 320px" }}>
        <div>
          {mode === "forecast" ? (
            <>
              <ForecastMap date={dateStr} onCellClick={setSelected} />
              <div style={{ padding: "0 16px 16px", position: "relative", zIndex: 10, background: "#0d1117" }}>
                <TimeSlider baseDate={FORECAST_BASE} offsetDays={offsetDays} onChange={setOffsetDays} />
              </div>
            </>
          ) : (
            <>
              <BlobMap monthOffset={blobMonth} />
              <div style={{ padding: "12px 24px 16px", display: "flex", flexDirection: "column", gap: 8, position: "relative", zIndex: 10, background: "#0d1117" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#ccc" }}>
                  <span>Jan 2014</span>
                  <span style={{ fontWeight: 700, color: "#f46d43", fontSize: 15 }}>
                    {format(blobDate, "MMM yyyy")}
                    {blobMonth >= 10 && blobMonth <= 14 ? " — Peak Blob" : ""}
                  </span>
                  <span>Dec 2015</span>
                </div>
                <input
                  type="range" min={0} max={23} value={blobMonth}
                  onChange={(e) => setBlobMonth(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#d73027" }}
                />
                <div style={{ textAlign: "center", fontSize: 12, color: "#aaa" }}>
                  Drag to replay the 2014–2015 NE Pacific marine heatwave
                </div>
              </div>
            </>
          )}
        </div>

        <aside style={{ borderLeft: "1px solid #21262d", padding: 20, background: "#161b22", overflowY: "auto" }}>
          {mode === "forecast" ? (
            <>
              <h2 style={{ margin: "0 0 16px", fontSize: 15, fontWeight: 600 }}>Forecast detail</h2>
              {selected ? (
                <RegionDetail lat={selected.lat} lon={selected.lon} date={dateStr} />
              ) : (
                <p style={{ fontSize: 13, color: "#8b949e" }}>Click a cell on the map to see the 4-lead-time forecast.</p>
              )}
            </>
          ) : (
            <>
              <h2 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600 }}>2014–15 Pacific Blob</h2>
              <p style={{ fontSize: 12, color: "#8b949e", lineHeight: 1.7, marginBottom: 12 }}>
                "The Blob" was the largest marine heatwave on record in the NE Pacific.
                SST anomalies reached <strong style={{ color: "#f46d43" }}>+2.5°C</strong> above the 1981–2010 climatological mean,
                devastating kelp forests and causing mass fish die-offs.
              </p>
              <p style={{ fontSize: 12, color: "#8b949e", lineHeight: 1.7, marginBottom: 12 }}>
                Drag the slider to replay how Tideline's model would have tracked
                the event from its early onset (mid-2014) through peak intensity (Jan 2015).
              </p>
              <div style={{ background: "#0d1117", borderRadius: 8, padding: 12, marginTop: 8 }}>
                <div style={{ fontSize: 11, color: "#666", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>Model performance</div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                  <span style={{ color: "#8b949e" }}>AUC (3-day)</span>
                  <span style={{ color: "#3fb950", fontWeight: 700 }}>0.8946</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                  <span style={{ color: "#8b949e" }}>AUC (7-day)</span>
                  <span style={{ color: "#3fb950", fontWeight: 700 }}>0.8928</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                  <span style={{ color: "#8b949e" }}>Test period</span>
                  <span style={{ color: "#aaa" }}>2022–2023</span>
                </div>
              </div>
            </>
          )}

          <hr style={{ border: "none", borderTop: "1px solid #21262d", margin: "24px 0" }} />
          <h2 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600 }}>About</h2>
          <p style={{ fontSize: 12, color: "#8b949e", lineHeight: 1.6 }}>
            Trained on NDBC buoy telemetry, CalCOFI oceanographic surveys, and kelp canopy data.
            Uses Hobday et al. (2016) marine heatwave thresholds.
            San Diego coastal model: LightGBM, AUC 0.89.
          </p>
        </aside>
      </main>
    </div>
  );
}
