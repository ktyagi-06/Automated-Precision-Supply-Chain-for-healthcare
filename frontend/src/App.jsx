import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { api } from "./api.js";
import { useLiveStream } from "./useLive.js";

const HORIZONS = [7, 14, 30];

export default function App() {
  const [horizon, setHorizon] = useState(7);
  const [products, setProducts] = useState(null);
  const [suppliers, setSuppliers] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [snap, setSnap] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [chartTick, setChartTick] = useState(0);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  // ---- initial reference data (retries until the backend answers)
  useEffect(() => {
    let cancelled = false;
    let timer;
    const load = () =>
      Promise.all([api.products(), api.suppliers(), api.modelInfo()])
        .then(([p, s, m]) => {
          if (cancelled) return;
          setProducts(p);
          setSuppliers(s);
          setModelInfo(m);
          setSelectedProduct((cur) => cur ?? p[0]?.product_id ?? null);
          setError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setError(e.message);
          timer = setTimeout(load, 4000);
        });
    load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  // ---- live snapshot: coalesced so a burst of events causes one fetch
  const horizonRef = useRef(horizon);
  horizonRef.current = horizon;
  const inFlight = useRef(false);
  const queued = useRef(false);
  const refreshSnapshot = useCallback(() => {
    if (inFlight.current) { queued.current = true; return; }
    inFlight.current = true;
    api.snapshot(horizonRef.current)
      .then((d) => { setSnap(d); setError(null); })
      .catch((e) => setError(e.message))
      .finally(() => {
        inFlight.current = false;
        if (queued.current) { queued.current = false; refreshSnapshot(); }
      });
  }, []);

  useEffect(() => { refreshSnapshot(); }, [horizon, refreshSnapshot]);

  const refreshModel = useCallback(
    () => api.modelInfo().then(setModelInfo).catch(() => {}), []);

  const connected = useLiveStream((type, data) => {
    switch (type) {
      case "hello":
      case "tick":
      case "order_placed":
      case "order_received":
        refreshSnapshot();
        break;
      case "day_closed":
        refreshSnapshot();
        setChartTick((n) => n + 1);
        break;
      case "retrain_started":
        setNotice("Retraining the model on the data collected so far…");
        refreshSnapshot();
        refreshModel();
        break;
      case "model_retrained":
        setNotice(`New model ${data.model_version} is live (${data.best_model}).`);
        refreshModel();
        refreshSnapshot();
        setChartTick((n) => n + 1);
        break;
      case "retrain_failed":
        setNotice(`Retrain failed: ${data.error}`);
        refreshModel();
        break;
      default:
    }
  });

  // refresh periodically as a safety net if the stream is down
  useEffect(() => {
    if (connected) return undefined;
    const t = setInterval(refreshSnapshot, 5000);
    return () => clearInterval(t);
  }, [connected, refreshSnapshot]);

  useEffect(() => {
    if (!notice) return undefined;
    const t = setTimeout(() => setNotice(null), 8000);
    return () => clearTimeout(t);
  }, [notice]);

  // ---- chart data for the selected product
  useEffect(() => {
    if (!selectedProduct) return;
    let cancelled = false;
    Promise.all([api.salesHistory(selectedProduct, 45), api.forecast(selectedProduct, horizon)])
      .then(([history, forecast]) => {
        if (cancelled) return;
        const histPoints = history.map((h) => ({ date: h.date.slice(5), actual: h.units_sold, promo: h.promo_flag }));
        const forecastPoints = forecast.daily_forecast.map((f) => ({ date: f.date.slice(5), predicted: f.predicted_units }));
        setChartData({ histPoints, forecastPoints, forecast });
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [selectedProduct, horizon, chartTick]);

  const combinedSeries = useMemo(() => {
    if (!chartData) return [];
    const { histPoints, forecastPoints } = chartData;
    const bridge = histPoints.length
      ? [{ ...histPoints[histPoints.length - 1], predicted: histPoints[histPoints.length - 1].actual }]
      : [];
    return [...histPoints, ...bridge, ...forecastPoints];
  }, [chartData]);

  const placeOrder = useCallback(async (productId, qty) => {
    try {
      const o = await api.placeOrder(productId, qty);
      setNotice(`Ordered ${o.qty} × ${o.product_name} from ${o.supplier_name}, due ${o.eta.slice(0, 10)}.`);
      refreshSnapshot();
    } catch (e) {
      setNotice(`Order failed: ${e.message}`);
    }
  }, [refreshSnapshot]);

  const retrain = useCallback(async () => {
    try {
      const r = await api.retrain();
      if (!r.started) setNotice(`Retrain not started: ${r.reason}.`);
    } catch (e) {
      setNotice(`Retrain failed: ${e.message}`);
    }
  }, []);

  if (!products || !snap || !suppliers) {
    return (
      <div className="app-shell">
        <div className="container">
          {error ? (
            <div className="error-box">
              Connection error — {error}. Retrying… Make sure the backend is running
              (see README).
            </div>
          ) : (
            <div className="loading">LOADING SUPPLY CHAIN STATE…</div>
          )}
        </div>
      </div>
    );
  }

  const { status, summary, alerts, reorders, orders, events } = snap;
  const selectedRec = reorders.find((r) => r.product_id === selectedProduct);

  return (
    <div className="app-shell">
      <div className="container">
        <LiveBar status={status} connected={connected} />
        {error && <div className="error-box" style={{ marginBottom: 16 }}>Connection problem — {error}. Retrying…</div>}
        {notice && <div className="notice-box">{notice}</div>}
        <Header horizon={horizon} setHorizon={setHorizon} modelInfo={modelInfo} />
        <KpiStrip summary={summary} />

        <div className="grid-main">
          <ForecastPanel
            products={products}
            selectedProduct={selectedProduct}
            setSelectedProduct={setSelectedProduct}
            combinedSeries={combinedSeries}
            chartData={chartData}
            selectedRec={selectedRec}
            horizon={horizon}
          />
          <AlertsPanel alerts={alerts} />
        </div>

        <InventoryPanel reorders={reorders} horizon={horizon} onOrder={placeOrder} />

        <div className="footer-grid">
          <OrdersPanel orders={orders} />
          <FeedPanel events={events} />
        </div>

        <div className="footer-grid">
          <SupplierPanel suppliers={suppliers} />
          <PipelinePanel modelInfo={modelInfo} status={status} onRetrain={retrain} />
        </div>
      </div>
    </div>
  );
}

function fmtSim(iso) {
  if (!iso) return "";
  return iso.replace("T", " ").slice(0, 16);
}

function LiveBar({ status, connected }) {
  const mode = { simulate: "SIMULATED FEED", ingest: "INGEST API", off: "STATIC DATA" }[status.live_mode] || status.live_mode;
  const live = connected && status.live_mode !== "off";
  return (
    <div className="live-bar">
      <span className={`live-dot ${live ? "on" : "off"}`} />
      <strong>{live ? "LIVE" : connected ? "PAUSED" : "OFFLINE"}</strong>
      <span>{mode}</span>
      <span>system time {fmtSim(status.sim_time)}</span>
      <span>data through {status.last_closed}</span>
      <span>{status.open_orders} open order{status.open_orders === 1 ? "" : "s"}</span>
      {status.auto_reorder && <span>auto-reorder on</span>}
    </div>
  );
}

function Header({ horizon, setHorizon, modelInfo }) {
  return (
    <div className="header">
      <div className="header__title-block">
        <div className="header__eyebrow">SUPPLY CHAIN CONTROL PANEL</div>
        <h1 className="header__title">Automated Precision Supply Chain</h1>
        <p className="header__subtitle">
          Live demand forecasts, reorder points, and supplier risk — computed
          from the trained model, not a spreadsheet guess.
        </p>
      </div>
      <div className="header__meta">
        <div className="horizon-select">
          {HORIZONS.map((h) => (
            <button
              key={h}
              className={h === horizon ? "active" : ""}
              onClick={() => setHorizon(h)}
            >
              {h}D HORIZON
            </button>
          ))}
        </div>
        {modelInfo && (
          <div className="model-tag">
            Forecast engine: <strong>{modelInfo.best_model}</strong>
          </div>
        )}
      </div>
    </div>
  );
}

function KpiStrip({ summary }) {
  const kpis = [
    {
      value: summary.total_inventory_units.toLocaleString(),
      label: "Units in inventory, all products",
      cls: "neutral",
    },
    {
      value: summary.low_stock_products,
      label: "Products at or below reorder point",
      cls: summary.low_stock_products > 0 ? "amber" : "healthy",
    },
    {
      value: summary.critical_products,
      label: "Products at critical stockout risk",
      cls: summary.critical_products > 0 ? "danger" : "healthy",
    },
    {
      value: summary.units_on_order.toLocaleString(),
      label: `Units on order (${summary.open_orders} open orders)`,
      cls: "neutral",
    },
    {
      value: summary.total_predicted_demand_next_period.toLocaleString(),
      label: `Predicted demand, next ${summary.forecast_horizon_days} days`,
      cls: "neutral",
    },
  ];
  return (
    <div className="kpi-strip">
      {kpis.map((k, i) => (
        <div className="kpi" key={i}>
          <div className={`kpi__index kpi__index--${k.cls}`} />
          <div className="kpi__value">{k.value}</div>
          <div className="kpi__label">{k.label}</div>
        </div>
      ))}
    </div>
  );
}

function ForecastPanel({ products, selectedProduct, setSelectedProduct, combinedSeries, chartData, selectedRec, horizon }) {
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Demand forecast</h2>
        <span className="panel__hint">
          {chartData ? `${chartData.forecast.model_used} · ${horizon}d horizon` : ""}
        </span>
      </div>
      <div className="panel__body">
        <div className="chip-row">
          {products.map((p) => (
            <button
              key={p.product_id}
              className={`chip ${p.product_id === selectedProduct ? "active" : ""}`}
              onClick={() => setSelectedProduct(p.product_id)}
            >
              {p.product_id} · {p.product_name}
            </button>
          ))}
        </div>

        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <ComposedChart data={combinedSeries} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="#2a5386" strokeOpacity={0.35} vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#b9c4d4", fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: "#2a5386" }} tickLine={false} />
              <YAxis tick={{ fill: "#b9c4d4", fontSize: 11, fontFamily: "IBM Plex Mono" }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "#0e2238", border: "1px solid #2a5386", fontFamily: "IBM Plex Mono", fontSize: 12 }}
                labelStyle={{ color: "#f2eee2" }}
              />
              <Area type="monotone" dataKey="actual" stroke="#f2eee2" fill="#f2eee2" fillOpacity={0.06} strokeWidth={2} name="Actual units sold" connectNulls />
              <Line type="monotone" dataKey="predicted" stroke="#f2a93b" strokeWidth={2} strokeDasharray="5 3" dot={false} name="Predicted units" connectNulls />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-caption">
          <span><span className="dot" style={{ background: "#f2eee2" }} />actual sales</span>
          <span><span className="dot" style={{ background: "#f2a93b" }} />ML forecast</span>
        </div>

        {selectedRec && (
          <div style={{ marginTop: 18, borderTop: "1px solid #2a5386", paddingTop: 14 }}>
            <table className="data-table">
              <tbody>
                <tr>
                  <td>Current stock</td>
                  <td className="num">{selectedRec.current_stock}</td>
                  <td>Reorder point</td>
                  <td className="num">{selectedRec.reorder_point}</td>
                </tr>
                <tr>
                  <td>Predicted demand ({horizon}d)</td>
                  <td className="num">{selectedRec.predicted_demand_horizon}</td>
                  <td>Safety stock</td>
                  <td className="num">{selectedRec.safety_stock}</td>
                </tr>
                <tr>
                  <td>Recommended order</td>
                  <td className="num">{selectedRec.recommended_order_qty}</td>
                  <td>On order</td>
                  <td className="num">{selectedRec.on_order_qty}{selectedRec.next_eta ? ` · due ${selectedRec.next_eta.slice(5, 10)}` : ""}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function AlertsPanel({ alerts }) {
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Supply-chain alerts</h2>
        <span className="panel__hint">{alerts.length} active</span>
      </div>
      <div className="alert-list">
        {alerts.length === 0 && <div className="alert-empty">No active alerts. All products within safe range.</div>}
        {alerts.map((a, i) => (
          <div className={`alert-item ${a.severity}`} key={i}>
            <div className="alert-icon">{a.icon}</div>
            <div className="alert-body">
              <div className="alert-product">{a.product_name}</div>
              <div className="alert-message">{a.message}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function InventoryPanel({ reorders, horizon, onOrder }) {
  const [busy, setBusy] = useState(null);
  const sorted = [...reorders].sort((a, b) => {
    const order = { CRITICAL: 0, HIGH: 1, WATCH: 2, HEALTHY: 3, OVERSTOCK: 4 };
    return order[a.risk_level] - order[b.risk_level];
  });
  const order = async (r) => {
    setBusy(r.product_id);
    await onOrder(r.product_id, r.recommended_order_qty);
    setBusy(null);
  };
  return (
    <div className="panel" style={{ marginBottom: 20 }}>
      <div className="panel__header">
        <h2 className="panel__title">Inventory status &amp; reorder recommendations</h2>
        <span className="panel__hint">{horizon}d horizon · updates live</span>
      </div>
      <div className="panel__body" style={{ overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Product</th>
              <th className="num">Stock</th>
              <th className="num">On order</th>
              <th className="num">Reorder pt.</th>
              <th className="num">Predicted demand</th>
              <th className="num">Days left</th>
              <th>Status</th>
              <th className="num">Order qty.</th>
              <th>Supplier</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.product_id}>
                <td>
                  <div className="product-cell">
                    <strong>{r.product_name}</strong>
                    <span>{r.product_id}</span>
                  </div>
                </td>
                <td className="num">{r.current_stock}</td>
                <td className="num">{r.on_order_qty || "—"}</td>
                <td className="num">{r.reorder_point}</td>
                <td className="num">{r.predicted_demand_horizon}</td>
                <td className="num">{r.days_of_stock_remaining >= 0 ? r.days_of_stock_remaining : "—"}</td>
                <td><span className={`status-tag ${r.risk_level}`}>{r.risk_level}</span></td>
                <td className="num">{r.recommended_order_qty || "—"}</td>
                <td className="num">{r.preferred_supplier_id}</td>
                <td>
                  {r.recommended_order_qty > 0 && (
                    <button className="btn" disabled={busy === r.product_id} onClick={() => order(r)}>
                      {busy === r.product_id ? "…" : "Order"}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OrdersPanel({ orders }) {
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Purchase orders</h2>
        <span className="panel__hint">{orders.filter((o) => o.status === "in_transit").length} in transit</span>
      </div>
      <div className="panel__body" style={{ overflowX: "auto", maxHeight: 320, overflowY: "auto" }}>
        {orders.length === 0 ? (
          <div className="alert-empty">No orders yet. Orders appear here when stock hits the reorder point.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr><th>#</th><th>Product</th><th className="num">Qty</th><th>Supplier</th><th>Due</th><th>Status</th></tr>
            </thead>
            <tbody>
              {orders.map((o) => (
                <tr key={o.id}>
                  <td className="num">{o.id}</td>
                  <td>{o.product_name}<span className="muted"> · {o.source}</span></td>
                  <td className="num">{o.qty}</td>
                  <td className="num">{o.supplier_id}</td>
                  <td className="num">{(o.received_at || o.eta).slice(5, 10)}{o.delay_days ? ` (+${o.delay_days}d late)` : ""}</td>
                  <td><span className={`status-tag ${o.status === "received" ? "HEALTHY" : "WATCH"}`}>{o.status === "received" ? "RECEIVED" : "IN TRANSIT"}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

const EVENT_LABEL = { sale: "SALE", order_placed: "ORDER", order_received: "RECEIVED" };

function FeedPanel({ events }) {
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Live activity</h2>
        <span className="panel__hint">sales &amp; deliveries as they happen</span>
      </div>
      <div className="feed">
        {events.length === 0 && <div className="alert-empty">Waiting for activity…</div>}
        {events.map((e, i) => (
          <div className={`feed-row ${e.kind}`} key={`${e.t}-${e.product_id}-${e.kind}-${i}`}>
            <span className="feed-time">{e.t.slice(5, 16).replace("T", " ")}</span>
            <span className="feed-kind">{EVENT_LABEL[e.kind] || e.kind}</span>
            <span className="feed-text">
              {e.kind === "sale" && `${e.units} × ${e.product_name} → ${e.stock_after} left${e.lost ? ` (${e.lost} lost, out of stock)` : ""}`}
              {e.kind === "order_placed" && `${e.units} × ${e.product_name} (${e.source})`}
              {e.kind === "order_received" && `${e.units} × ${e.product_name} arrived`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SupplierPanel({ suppliers }) {
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Supplier performance</h2>
      </div>
      <div className="panel__body">
        <table className="data-table">
          <thead>
            <tr>
              <th>Supplier</th>
              <th className="num">Avg. lead time</th>
              <th className="num">On-time rate</th>
              <th>Rating</th>
            </tr>
          </thead>
          <tbody>
            {suppliers.map((s) => (
              <tr key={s.supplier_id}>
                <td>{s.supplier_name}</td>
                <td className="num">{s.avg_lead_time_days}d</td>
                <td className="num">{Math.round(s.on_time_rate * 100)}%</td>
                <td><span className={`status-tag ${s.reliability_rating === "Poor" ? "CRITICAL" : s.reliability_rating === "Good" ? "HIGH" : "HEALTHY"}`}>{s.reliability_rating}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PipelinePanel({ modelInfo, status, onRetrain }) {
  if (!modelInfo) return null;
  const rt = status.retrain;
  const rows = [...modelInfo.model_comparison].sort((a, b) => (a.MAE_7d ?? a.MAE) - (b.MAE_7d ?? b.MAE));
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Model</h2>
        <span className="panel__hint">7-day recursive backtest, lower is better</span>
      </div>
      <div className="panel__body">
        <div className="model-meta">
          <span>version <strong>{modelInfo.model_version}</strong></span>
          <span>trained on data through <strong>{modelInfo.data_through}</strong></span>
          <span>{status.days_since_retrain}/{status.retrain_every_days || "—"} days since retrain</span>
        </div>
        {rows.map((m) => (
          <div className={`model-row ${m.model === modelInfo.best_model ? "selected" : ""}`} key={m.model}>
            <span>{m.model}{m.model === modelInfo.best_model ? " — selected" : ""}</span>
            <span className="model-metrics">
              <span>MAE {m.MAE_7d ?? m.MAE}</span>
              <span>RMSE {m.RMSE_7d ?? m.RMSE}</span>
              <span>MAPE {m.MAPE_7d ?? m.MAPE}%</span>
            </span>
          </div>
        ))}
        {modelInfo.baseline && (
          <div className="model-row baseline">
            <span>{modelInfo.baseline.model} (to beat)</span>
            <span className="model-metrics">
              <span>MAE {modelInfo.baseline.MAE_7d}</span>
              <span>RMSE {modelInfo.baseline.RMSE_7d}</span>
              <span>MAPE {modelInfo.baseline.MAPE_7d}%</span>
            </span>
          </div>
        )}
        <div className="retrain-row">
          <button className="btn" disabled={rt.running} onClick={onRetrain}>
            {rt.running ? "Retraining…" : "Retrain now"}
          </button>
          <span className="muted">
            Retrains automatically as new sales arrive.
            {rt.last_error ? ` Last error: ${rt.last_error}` : ""}
          </span>
        </div>
        <div style={{ marginTop: 16 }}>
          <div className="pipeline-steps">
            {["Live sales", "Cleaning", "Feature eng.", "Retraining", "Backtest", "Optimization", "Reorder"].map((s, i, arr) => (
              <React.Fragment key={s}>
                <span className="pipeline-step">{s}</span>
                {i < arr.length - 1 && <span className="pipeline-arrow">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
