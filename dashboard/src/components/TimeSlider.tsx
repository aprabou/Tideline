/**
 * TimeSlider.tsx — Scrub through 14 forecast days.
 */

import { addDays, format } from "date-fns";

interface TimeSliderProps {
  baseDate: Date;
  offsetDays: number;
  onChange: (offsetDays: number) => void;
  maxDays?: number;
}

export default function TimeSlider({
  baseDate,
  offsetDays,
  onChange,
  maxDays = 13,
}: TimeSliderProps) {
  const displayDate = addDays(baseDate, offsetDays);

  return (
    <div style={{ padding: "12px 0", display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#ccc" }}>
        <span>{format(baseDate, "MMM d")}</span>
        <span style={{ fontWeight: 600, color: "#fff" }}>
          {format(displayDate, "EEE, MMM d")}
        </span>
        <span>{format(addDays(baseDate, maxDays), "MMM d")}</span>
      </div>
      <input
        type="range"
        min={0}
        max={maxDays}
        value={offsetDays}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#e04a2f" }}
      />
      <div style={{ textAlign: "center", fontSize: 12, color: "#aaa" }}>
        Day +{offsetDays} — drag to explore forecast
      </div>
    </div>
  );
}
