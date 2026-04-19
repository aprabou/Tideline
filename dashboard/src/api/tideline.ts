/**
 * tideline.ts — Typed API client for the Tideline inference API.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

export interface DayForecast {
  date: string;
  prob_mhw: number;
  category: "none" | "moderate" | "strong" | "severe" | "extreme";
}

export interface ForecastResponse {
  lat: number;
  lon: number;
  horizon_days: number;
  forecasts: DayForecast[];
}

export interface GridCell {
  lat: number;
  lon: number;
  prob_mhw: number;
}

export interface GridForecastParams {
  date: string;
  lat_min?: number;
  lat_max?: number;
  lon_min?: number;
  lon_max?: number;
  step?: number;
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { ...init, headers: { "Content-Type": "application/json" } });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/** 14-day probabilistic forecast for a single lat/lon point. */
export async function getPointForecast(
  lat: number,
  lon: number,
  date: string
): Promise<ForecastResponse> {
  return fetchJSON<ForecastResponse>(`${BASE_URL}/forecast`, {
    method: "POST",
    body: JSON.stringify({ lat, lon, date }),
  });
}

/** Day-1 MHW probability for a bounding-box grid (map overlay). */
export async function getGridForecast(params: GridForecastParams): Promise<GridCell[]> {
  const qs = new URLSearchParams(
    Object.fromEntries(
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    )
  ).toString();
  return fetchJSON<GridCell[]>(`${BASE_URL}/forecast/grid?${qs}`);
}

/** Map MHW category to a display colour (hex). */
export const CATEGORY_COLORS: Record<DayForecast["category"], string> = {
  none: "#1a9641",
  moderate: "#fdae61",
  strong: "#f46d43",
  severe: "#d73027",
  extreme: "#7f0000",
};
