// In dev, Vite proxies /api to localhost:8000, so an empty base works.
// In production set VITE_API_BASE to the backend URL (no trailing slash).
const API_ORIGIN = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
export const BASE = `${API_ORIGIN}/api`;

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new Error(`${path} failed: ${res.status}${detail ? ` — ${detail}` : ""}`);
  }
  return res.json();
}

const post = (path, body) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });

export const api = {
  snapshot: (days = 7) => request(`/snapshot?days=${days}`),
  products: () => request("/products"),
  forecast: (id, days = 7) => request(`/forecast/${id}?days=${days}`),
  suppliers: () => request("/suppliers"),
  salesHistory: (id, lastNDays = 60) => request(`/sales-history/${id}?last_n_days=${lastNDays}`),
  modelInfo: () => request("/model-info"),
  placeOrder: (product_id, qty) => post("/orders", { product_id, qty }),
  retrain: () => post("/model/retrain"),
};
