import { useState } from "react";
import { format } from "date-fns";
import ForecastMap from "./components/ForecastMap";
import TimeSlider from "./components/TimeSlider";
import RegionDetail from "./components/RegionDetail";
import { GridCell } from "./api/tideline";

const BASE_DATE = new Date();

export default function App() {
  const [offsetDays, setOffsetDays] = useState(0);
  const [selected, setSelected] = useState<GridCell | null>(null);

  const forecastDate = new Date(BASE_DATE);
  forecastDate.setDate(forecastDate.getDate() + offsetDays);
  const dateStr = format(forecastDate, "yyyy-MM-dd");

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0d1117",
        color: "#e6edf3",
        fontFamily: "'Inter', system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: "14px 24px",
          borderBottom: "1px solid #21262d",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.5px" }}>
          🌊 Tideline
        </span>
        <span style={{ fontSize: 13, color: "#8b949e" }}>
          Marine Heatwave Forecasting · West Coast USA
        </span>
      </header>

      {/* Main */}
      <main style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 320px" }}>
        {/* Map */}
        <div>
          <ForecastMap date={dateStr} onCellClick={setSelected} />
          <div style={{ padding: "0 16px 16px" }}>
            <TimeSlider
              baseDate={BASE_DATE}
              offsetDays={offsetDays}
              onChange={setOffsetDays}
            />
          </div>
        </div>

        {/* Sidebar */}
        <aside
          style={{
            borderLeft: "1px solid #21262d",
            padding: 20,
            background: "#161b22",
            overflowY: "auto",
          }}
        >
          <h2 style={{ margin: "0 0 16px", fontSize: 15, fontWeight: 600 }}>
            Forecast detail
          </h2>
          {selected ? (
            <RegionDetail lat={selected.lat} lon={selected.lon} date={dateStr} />
          ) : (
            <p style={{ fontSize: 13, color: "#8b949e" }}>
              Click a cell on the map to see the 14-day probability curve.
            </p>
          )}

          <hr style={{ border: "none", borderTop: "1px solid #21262d", margin: "24px 0" }} />

          <h2 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600 }}>About</h2>
          <p style={{ fontSize: 12, color: "#8b949e", lineHeight: 1.6 }}>
            Tideline uses NOAA OISST satellite SST, NDBC buoy telemetry, and
            CalCOFI oceanographic data to train an XGBoost classifier on
            Hobday et al. (2016) marine heatwave thresholds.
          </p>
          <p style={{ fontSize: 12, color: "#8b949e", lineHeight: 1.6 }}>
            Forecasts are probabilistic. Cells coloured red indicate &gt; 70 %
            likelihood of a heatwave event within the selected time horizon.
          </p>
        </aside>
      </main>
    </div>
  );
}
