import React, { useEffect, useMemo, useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { api } from "./api.js";

const HORIZONS = [7, 14, 30];

export default function App() {
  const [horizon, setHorizon] = useState(7);
  const [products, setProducts] = useState(null);
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [reorders, setReorders] = useState(null);
  const [suppliers, setSuppliers] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [error, setError] = useState(null);

  // initial load
  useEffect(() => {
    Promise.all([api.products(), api.suppliers(), api.modelInfo()])
      .then(([p, s, m]) => {
        setProducts(p);
        setSuppliers(s);
        setModelInfo(m);
        setSelectedProduct(p[0]?.product_id ?? null);
      })
      .catch((e) => setError(e.message));
  }, []);

  // horizon-dependent data
  useEffect(() => {
    Promise.all([
      api.dashboardSummary(horizon),
      api.alerts(horizon),
      api.reorderAll(horizon),
    ])
      .then(([sum, al, ro]) => {
        setSummary(sum);
        setAlerts(al);
        setReorders(ro);
      })
      .catch((e) => setError(e.message));
  }, [horizon]);

  // chart data for selected product
  useEffect(() => {
    if (!selectedProduct) return;
    Promise.all([
      api.salesHistory(selectedProduct, 45),
      api.forecast(selectedProduct, horizon),
    ])
      .then(([history, forecast]) => {
        const histPoints = history.map((h) => ({
          date: h.date.slice(5),
          actual: h.units_sold,
          promo: h.promo_flag,
        }));
        const forecastPoints = forecast.daily_forecast.map((f) => ({
          date: f.date.slice(5),
          predicted: f.predicted_units,
        }));
        setChartData({ histPoints, forecastPoints, forecast });
      })
      .catch((e) => setError(e.message));
  }, [selectedProduct, horizon]);

  const combinedSeries = useMemo(() => {
    if (!chartData) return [];
    const { histPoints, forecastPoints } = chartData;
    const bridge = histPoints.length
      ? [{ ...histPoints[histPoints.length - 1], predicted: histPoints[histPoints.length - 1].actual }]
      : [];
    return [...histPoints, ...bridge, ...forecastPoints];
  }, [chartData]);

  if (error) {
    return (
      <div className="app-shell">
        <div className="container">
          <div className="error-box">
            Connection error — {error}. Make sure the backend is running at
            localhost:8000 (see README).
          </div>
        </div>
      </div>
    );
  }

  if (!products || !summary || !alerts || !reorders || !suppliers) {
    return (
      <div className="app-shell">
        <div className="container">
          <div className="loading">LOADING SUPPLY CHAIN STATE…</div>
        </div>
      </div>
    );
  }

  const selectedRec = reorders.find((r) => r.product_id === selectedProduct);

  return (
    <div className="app-shell">
      <div className="container">
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

        <InventoryPanel reorders={reorders} horizon={horizon} />

        <div className="footer-grid">
          <SupplierPanel suppliers={suppliers} />
          <PipelinePanel modelInfo={modelInfo} />
        </div>
      </div>
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
                  <td>Preferred supplier</td>
                  <td className="num">{selectedRec.preferred_supplier_id}</td>
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

function InventoryPanel({ reorders, horizon }) {
  const sorted = [...reorders].sort((a, b) => {
    const order = { CRITICAL: 0, HIGH: 1, WATCH: 2, HEALTHY: 3, OVERSTOCK: 4 };
    return order[a.risk_level] - order[b.risk_level];
  });
  return (
    <div className="panel" style={{ marginBottom: 20 }}>
      <div className="panel__header">
        <h2 className="panel__title">Inventory status &amp; reorder recommendations</h2>
        <span className="panel__hint">{horizon}d horizon</span>
      </div>
      <div className="panel__body" style={{ overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Product</th>
              <th className="num">Stock</th>
              <th className="num">Reorder pt.</th>
              <th className="num">Predicted demand</th>
              <th className="num">Days left</th>
              <th>Status</th>
              <th className="num">Order qty.</th>
              <th>Supplier</th>
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
                <td className="num">{r.reorder_point}</td>
                <td className="num">{r.predicted_demand_horizon}</td>
                <td className="num">{r.days_of_stock_remaining >= 0 ? r.days_of_stock_remaining : "—"}</td>
                <td><span className={`status-tag ${r.risk_level}`}>{r.risk_level}</span></td>
                <td className="num">{r.recommended_order_qty || "—"}</td>
                <td className="num">{r.preferred_supplier_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
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

function PipelinePanel({ modelInfo }) {
  if (!modelInfo) return null;
  return (
    <div className="panel">
      <div className="panel__header">
        <h2 className="panel__title">Model comparison</h2>
        <span className="panel__hint">validation MAE, lower is better</span>
      </div>
      <div className="panel__body">
        {modelInfo.model_comparison
          .slice()
          .sort((a, b) => a.MAE - b.MAE)
          .map((m) => (
            <div className={`model-row ${m.model === modelInfo.best_model ? "selected" : ""}`} key={m.model}>
              <span>{m.model}{m.model === modelInfo.best_model ? " — selected" : ""}</span>
              <span className="model-metrics">
                <span>MAE {m.MAE}</span>
                <span>RMSE {m.RMSE}</span>
                <span>MAPE {m.MAPE}%</span>
              </span>
            </div>
          ))}
        <div style={{ marginTop: 16 }}>
          <div className="pipeline-steps">
            {["Raw data", "Cleaning", "Feature eng.", "Model training", "Evaluation", "Optimization", "Recommendation"].map((s, i, arr) => (
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
