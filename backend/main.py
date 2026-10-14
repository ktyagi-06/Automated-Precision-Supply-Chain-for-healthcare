"""
Automated Precision Supply Chain — Backend API
================================================
FastAPI service exposing:
  GET  /api/products                 -> product master list + current stock
  GET  /api/products/{id}            -> single product detail
  GET  /api/forecast/{id}?days=7     -> ML demand forecast for a product
  GET  /api/reorder/{id}?days=7      -> reorder recommendation for a product
  GET  /api/reorder                  -> reorder recommendations for ALL products
  GET  /api/alerts                   -> active supply-chain alerts
  GET  /api/suppliers                -> supplier performance table
  GET  /api/dashboard-summary        -> aggregated KPIs for the dashboard
  GET  /api/sales-history/{id}       -> recent sales history (for charts)
  GET  /api/model-info               -> model comparison metrics & metadata

Run: uvicorn main:app --reload --port 8000
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import forecasting
import optimization

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(
    title="Automated Precision Supply Chain API",
    description="ML-powered demand forecasting and inventory optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load reference data once at startup
# ---------------------------------------------------------------------------
products_df = pd.read_csv(DATA_DIR / "products.csv")
suppliers_df = pd.read_csv(DATA_DIR / "suppliers.csv")
inventory_df = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
sales_df = pd.read_csv(DATA_DIR / "sales_history.csv", parse_dates=["date"])

merged_df = products_df.merge(inventory_df, on="product_id").merge(
    suppliers_df, on="supplier_id"
)

DEFAULT_HORIZON = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_product_row(product_id: str) -> pd.Series:
    row = merged_df[merged_df["product_id"] == product_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    return row.iloc[0]


def _build_recommendation_for(product_id: str, horizon_days: int) -> optimization.ReorderRecommendation:
    row = _get_product_row(product_id)
    fc = forecasting.forecast_product(product_id, horizon_days=horizon_days)
    return optimization.build_recommendation(
        product_id=product_id,
        product_name=row.product_name,
        current_stock=row.current_stock,
        avg_daily_demand=fc["avg_daily_demand_recent"],
        demand_std=fc["demand_std_recent"],
        lead_time_days=row.lead_time_days,
        predicted_demand_horizon=fc["total_predicted_demand"],
        horizon_days=horizon_days,
        max_stock=row.max_stock,
        preferred_supplier_id=row.supplier_id,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/products")
def list_products():
    out = []
    for _, r in merged_df.iterrows():
        out.append({
            "product_id": r.product_id,
            "product_name": r.product_name,
            "category": r.category,
            "current_stock": int(r.current_stock),
            "min_stock": int(r.min_stock),
            "max_stock": int(r.max_stock),
            "unit_cost": float(r.unit_cost),
            "warehouse": r.warehouse,
            "supplier_id": r.supplier_id,
            "supplier_name": r.supplier_name,
            "lead_time_days": int(r.lead_time_days),
        })
    return out


@app.get("/api/products/{product_id}")
def get_product(product_id: str):
    r = _get_product_row(product_id)
    return {
        "product_id": r.product_id,
        "product_name": r.product_name,
        "category": r.category,
        "current_stock": int(r.current_stock),
        "min_stock": int(r.min_stock),
        "max_stock": int(r.max_stock),
        "unit_cost": float(r.unit_cost),
        "warehouse": r.warehouse,
        "supplier_id": r.supplier_id,
        "supplier_name": r.supplier_name,
        "lead_time_days": int(r.lead_time_days),
    }


@app.get("/api/forecast/{product_id}")
def get_forecast(product_id: str, days: int = Query(DEFAULT_HORIZON, ge=1, le=60)):
    _get_product_row(product_id)  # validates existence
    try:
        return forecasting.forecast_product(product_id, horizon_days=days)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/reorder/{product_id}")
def get_reorder(product_id: str, days: int = Query(DEFAULT_HORIZON, ge=1, le=60)):
    rec = _build_recommendation_for(product_id, days)
    return rec.to_dict()


@app.get("/api/reorder")
def get_all_reorders(days: int = Query(DEFAULT_HORIZON, ge=1, le=60)):
    return [
        _build_recommendation_for(pid, days).to_dict()
        for pid in merged_df["product_id"]
    ]


@app.get("/api/alerts")
def get_alerts(days: int = Query(DEFAULT_HORIZON, ge=1, le=60)):
    recs = [_build_recommendation_for(pid, days) for pid in merged_df["product_id"]]
    alerts = []
    for r in recs:
        if r.risk_level == "CRITICAL":
            alerts.append({
                "severity": "critical",
                "icon": "🚨",
                "product_id": r.product_id,
                "product_name": r.product_name,
                "message": (
                    f"Predicted stockout in {r.predicted_stockout_in_days} days"
                    if r.predicted_stockout_in_days is not None
                    else "Inventory critically low"
                ),
            })
        elif r.risk_level == "HIGH":
            alerts.append({
                "severity": "high",
                "icon": "⚠",
                "product_id": r.product_id,
                "product_name": r.product_name,
                "message": "Inventory below reorder point — order recommended",
            })
        elif r.risk_level == "OVERSTOCK":
            alerts.append({
                "severity": "info",
                "icon": "📦",
                "product_id": r.product_id,
                "product_name": r.product_name,
                "message": "Inventory significantly exceeds expected demand",
            })

    # demand spike detection: compare last 7 days vs prior 7 days per product
    for pid in merged_df["product_id"]:
        s = sales_df[sales_df["product_id"] == pid].sort_values("date")
        last7 = s.tail(7)["units_sold"].sum()
        prev7 = s.tail(14).head(7)["units_sold"].sum()
        if prev7 > 0:
            change = (last7 - prev7) / prev7 * 100
            if change >= 30:
                name = merged_df.loc[merged_df.product_id == pid, "product_name"].iloc[0]
                alerts.append({
                    "severity": "info",
                    "icon": "📈",
                    "product_id": pid,
                    "product_name": name,
                    "message": f"Demand increased {change:.0f}% vs prior week",
                })
    return alerts


@app.get("/api/suppliers")
def get_suppliers():
    return suppliers_df.to_dict(orient="records")


@app.get("/api/sales-history/{product_id}")
def get_sales_history(product_id: str, last_n_days: int = Query(90, ge=7, le=730)):
    _get_product_row(product_id)
    s = sales_df[sales_df["product_id"] == product_id].sort_values("date").tail(last_n_days)
    return [
        {"date": d.strftime("%Y-%m-%d"), "units_sold": int(u), "promo_flag": int(p)}
        for d, u, p in zip(s["date"], s["units_sold"], s["promo_flag"])
    ]


@app.get("/api/model-info")
def model_info():
    return forecasting.get_model_metadata()


@app.get("/api/dashboard-summary")
def dashboard_summary(days: int = Query(DEFAULT_HORIZON, ge=1, le=60)):
    recs = [_build_recommendation_for(pid, days) for pid in merged_df["product_id"]]
    total_inventory = int(merged_df["current_stock"].sum())
    low_stock_count = sum(1 for r in recs if r.risk_level in ("HIGH", "CRITICAL"))
    total_predicted_demand = round(sum(r.predicted_demand_horizon for r in recs), 1)
    overstock_count = sum(1 for r in recs if r.risk_level == "OVERSTOCK")
    critical_count = sum(1 for r in recs if r.risk_level == "CRITICAL")

    status_breakdown = {}
    for r in recs:
        status_breakdown[r.risk_level] = status_breakdown.get(r.risk_level, 0) + 1

    return {
        "total_inventory_units": total_inventory,
        "low_stock_products": low_stock_count,
        "critical_products": critical_count,
        "overstock_products": overstock_count,
        "total_predicted_demand_next_period": total_predicted_demand,
        "forecast_horizon_days": days,
        "status_breakdown": status_breakdown,
        "product_statuses": [
            {"product_id": r.product_id, "product_name": r.product_name, "risk_level": r.risk_level}
            for r in recs
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
