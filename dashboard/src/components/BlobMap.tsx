import { useEffect, useState } from "react";
import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { Map } from "@vis.gl/react-maplibre";

interface BlobCell {
  lat: number;
  lon: number;
  prob_mhw: number;
  category: string;
}

interface BlobTimestep {
  date: string;
  region_mean_prob: number;
  cells: BlobCell[];
}

const INITIAL_VIEW = {
  longitude: -138.0,
  latitude: 44.0,
  zoom: 4,
  pitch: 0,
  bearing: 0,
};

function probToColor(prob: number): [number, number, number, number] {
  const r = Math.round(255 * Math.min(1, prob * 1.5));
  const g = Math.round(255 * Math.max(0, 1 - prob * 1.5));
  return [r, g, 30, 220];
}

const BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

export default function BlobMap({ monthOffset }: { monthOffset: number }) {
  const [timeline, setTimeline] = useState<BlobTimestep[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`${BASE_URL}/backtest/blob`)
      .then((r) => r.json())
      .then((d) => setTimeline(d.timeline ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const step = timeline[monthOffset];
  const cells: BlobCell[] = step?.cells ?? [];

  const layer = new ScatterplotLayer<BlobCell>({
    id: "blob-grid",
    data: cells,
    getPosition: (d) => [d.lon, d.lat],
    getRadius: 120000,
    getFillColor: (d) => probToColor(d.prob_mhw),
    pickable: true,
  });

  return (
    <div style={{ position: "relative", width: "100%", height: "600px" }}>
      {loading && (
        <div style={{ position: "absolute", top: 8, left: 8, zIndex: 10, background: "rgba(0,0,0,0.6)", color: "#fff", padding: "4px 10px", borderRadius: 4, fontSize: 12 }}>
          Loading blob replay…
        </div>
      )}
      {step && (
        <div style={{ position: "absolute", top: 8, left: "50%", transform: "translateX(-50%)", zIndex: 10, background: "rgba(0,0,0,0.75)", color: "#fff", padding: "6px 14px", borderRadius: 6, fontSize: 13, textAlign: "center" }}>
          Region mean MHW probability: <strong style={{ color: step.region_mean_prob > 0.5 ? "#f46d43" : step.region_mean_prob > 0.25 ? "#fdae61" : "#3fb950" }}>
            {(step.region_mean_prob * 100).toFixed(0)}%
          </strong>
        </div>
      )}
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={true}
        layers={[layer]}
        getTooltip={({ object }: { object?: BlobCell }) =>
          object
            ? { html: `<b>${(object.prob_mhw * 100).toFixed(0)}% MHW risk</b><br/>${object.lat}°N ${Math.abs(object.lon)}°W<br/><i>${object.category}</i>` }
            : null
        }
      >
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
          attributionControl={false}
        />
      </DeckGL>
      <div style={{ position: "absolute", bottom: 24, right: 16, background: "rgba(0,0,0,0.75)", color: "#fff", padding: "10px 14px", borderRadius: 6, fontSize: 12 }}>
        <div style={{ marginBottom: 6, fontWeight: 600 }}>MHW Probability</div>
        {[["< 25%", "#1a9641"], ["25–50%", "#fdae61"], ["50–70%", "#f46d43"], ["70–85%", "#d73027"], ["> 85%", "#7f0000"]].map(([label, color]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{ width: 12, height: 12, background: color, borderRadius: 2, display: "inline-block" }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
