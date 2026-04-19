const BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

export interface LeadForecast {
  lead_days: number;
  target_date: string;
  prob_mhw: number;
  ci_low: number;
  ci_high: number;
  category: "none" | "moderate" | "strong" | "severe" | "extreme";
  top_features: Record<string, number>;
}

export interface ForecastResponse {
  lat: number;
  lon: number;
  date: string;
  forecasts: LeadForecast[];
}

export interface GridCell {
  lat: number;
  lon: number;
  prob_mhw: number;
  category: "none" | "moderate" | "strong" | "severe" | "extreme";
}

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function getPointForecast(lat: number, lon: number, date: string): Promise<ForecastResponse> {
  return fetchJSON<ForecastResponse>(`${BASE_URL}/forecast?lat=${lat}&lon=${lon}&date=${date}`);
}

export async function getGridForecast(date: string, leadTime = 7): Promise<GridCell[]> {
  return fetchJSON<GridCell[]>(`${BASE_URL}/forecast_grid?date=${date}&lead_time=${leadTime}`);
}

export const CATEGORY_COLORS: Record<LeadForecast["category"], string> = {
  none: "#1a9641",
  moderate: "#fdae61",
  strong: "#f46d43",
  severe: "#d73027",
  extreme: "#7f0000",
};
