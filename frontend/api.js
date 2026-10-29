const API_ORIGIN = (
  import.meta.env.VITE_API_BASE ||
  "https://automated-precision-supply-chain-for.onrender.com"
).replace(/\/$/, "");

const BASE = `${API_ORIGIN}/api`;
const REQUEST_TIMEOUT_MS = 20000;

async function get(path) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${BASE}${path}`, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });

    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = body?.detail ? ` — ${body.detail}` : "";
      } catch {
        // The response body was not JSON; keep the HTTP status message.
      }
      throw new Error(`${path} failed: ${res.status}${detail}`);
    }

    return await res.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
        `${path} timed out after ${REQUEST_TIMEOUT_MS / 1000} seconds. The backend may be starting or processing slowly.`
      );
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export const api = {
  products: () => get("/products"),
  reorderAll: (days = 7) => get(`/reorder?days=${days}`),
  reorder: (id, days = 7) => get(`/reorder/${encodeURIComponent(id)}?days=${days}`),
  forecast: (id, days = 7) =>
    get(`/forecast/${encodeURIComponent(id)}?days=${days}`),
  alerts: (days = 7) => get(`/alerts?days=${days}`),
  suppliers: () => get("/suppliers"),
  salesHistory: (id, lastNDays = 60) =>
    get(`/sales-history/${encodeURIComponent(id)}?last_n_days=${lastNDays}`),
  dashboardSummary: (days = 7) =>
    get(`/dashboard-summary?days=${days}`),
  modelInfo: () => get("/model-info"),
};
