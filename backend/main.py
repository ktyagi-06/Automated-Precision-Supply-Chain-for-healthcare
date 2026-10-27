"""
Automated Precision Supply Chain — backend API
===============================================
FastAPI service: ML demand forecasts, inventory optimisation and a live
event stream.

Read
  GET  /api/health                    liveness + live-engine summary
  GET  /api/status                    live clock, mode, model version, retrain state
  GET  /api/snapshot?days=7           everything the dashboard needs in one call
  GET  /api/products[/{id}]           product master data + live stock
  GET  /api/forecast/{id}?days=7      ML demand forecast
  GET  /api/reorder[/{id}]?days=7     reorder recommendations
  GET  /api/alerts?days=7             stockout / overstock / demand-spike alerts
  GET  /api/dashboard-summary?days=7  aggregated KPIs
  GET  /api/sales-history/{id}        daily sales history
  GET  /api/suppliers                 supplier table
  GET  /api/orders                    purchase orders (in transit + recent)
  GET  /api/events                    recent sales / order events
  GET  /api/model-info                model comparison and metadata
  GET  /api/stream                    Server-Sent Events: tick, day_closed,
                                      order_placed, order_received, model_retrained

Write
  POST /api/orders                    place a purchase order
  POST /api/model/retrain             retrain on all data collected so far
  POST /api/ingest/sale               push a real sale (X-API-Key if API_KEY is set)
  POST /api/ingest/stock              sync an absolute stock count

Run (from backend/):  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Settings
from forecasting import ModelIncompatible, ModelStore
from live import MAX_FORECAST_DAYS, LiveEngine

log = logging.getLogger("scm.api")
DEFAULT_HORIZON = 7

STATE = SimpleNamespace(engine=None, store=None, cfg=None, task=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Settings.from_env()
    STATE.cfg = cfg
    STATE.store = ModelStore(cfg.model_dir)
    STATE.engine = await asyncio.to_thread(LiveEngine, cfg, STATE.store)
    await asyncio.to_thread(STATE.engine.ensure_model)
    await asyncio.to_thread(STATE.engine.get_forecasts, 45)  # warm the cache
    STATE.engine.attach_loop(asyncio.get_running_loop())
    if cfg.autostart and cfg.live_mode != "off":
        STATE.task = asyncio.create_task(STATE.engine.run())
        log.info("live engine running: mode=%s tick=%ss step=%smin",
                 cfg.live_mode, cfg.tick_seconds, cfg.sim_minutes_per_tick)
    try:
        yield
    finally:
        if STATE.task:
            STATE.task.cancel()
        STATE.engine.close()


app = FastAPI(
    title="Automated Precision Supply Chain API",
    description="ML-powered demand forecasting, inventory optimisation and live supply-chain events",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.from_env().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def engine() -> LiveEngine:
    if STATE.engine is None:
        raise HTTPException(status_code=503, detail="service starting")
    return STATE.engine


def require_api_key(x_api_key: str | None = Header(default=None)):
    key = STATE.cfg.api_key if STATE.cfg else None
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _horizon(days: int = Query(DEFAULT_HORIZON, ge=1, le=MAX_FORECAST_DAYS)) -> int:
    return days


def _known(pid: str) -> str:
    if pid not in engine().products:
        raise HTTPException(status_code=404, detail=f"Product '{pid}' not found")
    return pid


# ----------------------------------------------------------------------------
# Read endpoints
# ----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    e = STATE.engine
    return {"status": "ok" if e else "starting",
            "live_mode": STATE.cfg.live_mode if STATE.cfg else None,
            "model_version": STATE.store.version if STATE.store else None}


@app.get("/api/status")
def status():
    return engine().status()


@app.get("/api/snapshot")
def snapshot(days: int = Depends(_horizon)):
    return engine().snapshot(days)


@app.get("/api/products")
def list_products():
    return engine().products_with_stock()


@app.get("/api/products/{product_id}")
def get_product(product_id: str):
    _known(product_id)
    return next(p for p in engine().products_with_stock() if p["product_id"] == product_id)


@app.get("/api/forecast/{product_id}")
def get_forecast(product_id: str, days: int = Depends(_horizon)):
    _known(product_id)
    try:
        return engine().forecast_product(product_id, days)
    except ModelIncompatible as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/reorder/{product_id}")
def get_reorder(product_id: str, days: int = Depends(_horizon)):
    _known(product_id)
    return next(r for r in engine().recommendations(days) if r["product_id"] == product_id)


@app.get("/api/reorder")
def get_all_reorders(days: int = Depends(_horizon)):
    return engine().recommendations(days)


@app.get("/api/alerts")
def get_alerts(days: int = Depends(_horizon)):
    return engine().alerts(days)


@app.get("/api/dashboard-summary")
def dashboard_summary(days: int = Depends(_horizon)):
    return engine().dashboard_summary(days)


@app.get("/api/suppliers")
def get_suppliers():
    return engine().suppliers_df.to_dict(orient="records")


@app.get("/api/sales-history/{product_id}")
def get_sales_history(product_id: str, last_n_days: int = Query(90, ge=7, le=730)):
    _known(product_id)
    return engine().sales_history(product_id, last_n_days)


@app.get("/api/orders")
def get_orders():
    return engine().orders_view()


@app.get("/api/events")
def get_events(n: int = Query(40, ge=1, le=80)):
    return engine().recent_events(n)


@app.get("/api/model-info")
def model_info():
    if not STATE.store or not STATE.store.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {**STATE.store.metadata, "retrain": engine().retrain_state,
            "days_since_retrain": engine().days_since_retrain}


# ----------------------------------------------------------------------------
# Live stream (Server-Sent Events)
# ----------------------------------------------------------------------------
@app.get("/api/stream")
async def stream(request: Request):
    eng = engine()
    q = eng.subscribe()

    async def gen():
        try:
            yield "retry: 3000\n\n"
            yield f"event: hello\ndata: {json.dumps({'type': 'hello', **eng.status()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    etype, msg = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # keep proxies from closing an idle stream
                    continue
                yield f"event: {etype}\ndata: {msg}\n\n"
        finally:
            eng.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------------
# Write endpoints
# ----------------------------------------------------------------------------
class OrderIn(BaseModel):
    product_id: str
    qty: int = Field(gt=0)


class SaleIn(BaseModel):
    product_id: str
    units: int = Field(gt=0, le=100000)


class StockIn(BaseModel):
    product_id: str
    stock: int = Field(ge=0)


@app.post("/api/orders", status_code=201)
def create_order(body: OrderIn):
    _known(body.product_id)
    try:
        return engine().place_order(body.product_id, body.qty, source="manual")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/model/retrain", status_code=202)
def retrain(fast: bool | None = None):
    return engine().retrain(fast=fast, reason="manual")


@app.post("/api/ingest/sale", status_code=202, dependencies=[Depends(require_api_key)])
def ingest_sale(body: SaleIn):
    _known(body.product_id)
    return engine().ingest_sale(body.product_id, body.units)


@app.post("/api/ingest/stock", dependencies=[Depends(require_api_key)])
def ingest_stock(body: StockIn):
    _known(body.product_id)
    return engine().ingest_stock(body.product_id, body.stock)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
