import { useEffect, useState } from "react";
import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { Map } from "@vis.gl/react-maplibre";
import { getGridForecast, GridCell, CATEGORY_COLORS } from "../api/tideline";

interface ForecastMapProps {
  date: string;
  onCellClick?: (cell: GridCell) => void;
}

const INITIAL_VIEW = {
  longitude: -122.0,
  latitude: 38.0,
  zoom: 5,
  pitch: 0,
  bearing: 0,
};

function probToColor(prob: number): [number, number, number, number] {
  const r = Math.round(255 * prob);
  const g = Math.round(255 * (1 - prob));
  return [r, g, 50, 210];
}

export default function ForecastMap({ date, onCellClick }: ForecastMapProps) {
  const [cells, setCells] = useState<GridCell[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getGridForecast(date, 7)
      .then(setCells)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [date]);

  const layer = new ScatterplotLayer<GridCell>({
    id: "mhw-grid",
    data: cells,
    getPosition: (d) => [d.lon, d.lat],
    getRadius: 80000,
    getFillColor: (d) => probToColor(d.prob_mhw),
    pickable: true,
    onClick: ({ object }) => object && onCellClick?.(object),
  });

  return (
    <div style={{ position: "relative", width: "100%", height: "600px" }}>
      {loading && (
        <div style={{ position: "absolute", top: 8, left: 8, zIndex: 10, background: "rgba(0,0,0,0.6)", color: "#fff", padding: "4px 10px", borderRadius: 4, fontSize: 12 }}>
          Loading forecast…
        </div>
      )}
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={true}
        layers={[layer]}
        getTooltip={({ object }: { object?: GridCell }) =>
          object
            ? { html: `<b>${(object.prob_mhw * 100).toFixed(0)}% MHW risk</b><br/>${object.lat.toFixed(1)}°N ${Math.abs(object.lon).toFixed(1)}°W<br/><i>${object.category}</i>` }
            : null
        }
      >
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
          attributionControl={false}
        />
      </DeckGL>
      <div style={{ position: "absolute", bottom: 24, right: 16, background: "rgba(0,0,0,0.75)", color: "#fff", padding: "10px 14px", borderRadius: 6, fontSize: 12 }}>
        <div style={{ marginBottom: 6, fontWeight: 600 }}>7-day MHW risk</div>
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
          <div key={cat} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{ width: 12, height: 12, background: color, borderRadius: 2, display: "inline-block" }} />
            <span style={{ textTransform: "capitalize" }}>{cat}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
