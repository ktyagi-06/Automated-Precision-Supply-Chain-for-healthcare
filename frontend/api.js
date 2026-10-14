
const API_ORIGIN = (
  import.meta.env.VITE_API_BASE ||
  "https://automated-precision-supply-chain-for.onrender.com"
).replace(/\/$/, "");

const BASE = `${API_ORIGIN}/api`;

async function get(path) {
  const res = await fetch(`${BASE}${path}`);

  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }

  return res.json();
}

export const api = {
  products: () => get("/products"),
  reorderAll: () => get("/reorder"),
  reorder: (id) => get(`/reorder/${id}`),
  forecast: (id, days = 7) => get(`/forecast/${id}?days=${days}`),
  alerts: (days = 7) => get(`/alerts?days=${days}`),
  suppliers: () => get("/suppliers"),
  salesHistory: (id, lastNDays = 60) =>
    get(`/sales-history/${id}?last_n_days=${lastNDays}`),
  dashboardSummary: (days = 7) =>
    get(`/dashboard-summary?days=${days}`),
  modelInfo: () => get("/model-info"),
};
