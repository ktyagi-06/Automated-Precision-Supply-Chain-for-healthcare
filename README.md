# Automated Precision Supply Chain

A live, ML-driven supply-chain decision-support system. Sales stream in
continuously, stock is depleted in real time, an ML model forecasts demand and
is **retrained automatically as new data arrives**, and the system raises
purchase orders at statistically-derived reorder points — all visible on a
React dashboard that updates without a refresh.

```
live sales ──► stock & history ──► daily close ──► forecast (ML) ──► safety stock / reorder point
     ▲                                  │                                      │
     │                                  └──► retrain every N days ◄────────────┤
  simulator or                                                                 ▼
  POST /api/ingest/sale                                  auto purchase orders ──► supplier lead time ──► stock
                                                                               │
                                              Server-Sent Events ──► React dashboard
```

## What is real, and what is simulated

| Part | Status |
|---|---|
| ML model (Random Forest / XGBoost / Linear, compared and selected) | Real: trained and backtested by `ml/train_model.py` |
| Retraining on newly collected data, hot-swapped with no downtime | Real |
| Forecasting, safety stock, reorder points, order sizing | Real |
| Live updates to the dashboard (Server-Sent Events) | Real |
| Sales feed | **Simulated** by default, from the same demand process the model was trained on |
| Products, suppliers, 2 years of history | **Synthetic** demo data |

The synthetic data means the reported accuracy describes this demo dataset, not
real procurement performance. To use real data, set `LIVE_MODE=ingest` and push
sales to `POST /api/ingest/sale` (see below), and replace the CSVs in `data/`
with real history in the same schema.

## Quick start

```bash
pip install -r requirements-dev.txt
cd backend && uvicorn main:app --port 8000        # terminal 1
cd frontend && npm install && npm run dev          # terminal 2 → http://localhost:5173
```

The backend starts the live engine automatically. By default one simulated hour
passes every 5 seconds (a full day in 2 minutes) so you can watch the system work;
see [Configuration](#configuration) for real-time speed.

Run the tests: `python -m pytest` (24 tests: feature parity between training and
serving, stock conservation, restart recovery, API contract, order policy).

## Project layout

```
data/        products, suppliers, 2 years of daily sales, starting inventory (CSV)
ml/          demand_process.py   demand model shared by the generator and the live simulator
             features.py         feature engineering, identical for training and serving
             forecast_core.py    recursive multi-day forecasting
             train_model.py      train, backtest, compare, save
             generate_data.py    regenerate the synthetic CSVs
Models/      saved model, encoder, metadata, comparison table
backend/     main.py             FastAPI routes + SSE stream
             live.py             live engine: clock, sales, stock, orders, retraining
             forecasting.py      model store with hot-swap and version checks
             optimization.py     safety stock, reorder point, order sizing, risk levels
             config.py           environment-variable settings
frontend/    React + Vite dashboard
tests/       pytest suite
```

## The ML model

`ml/train_model.py` cleans the sales series, builds lag / rolling / calendar / promo
features and trains one global model (product and category one-hot encoded) that
predicts next-day units sold. Three candidates are compared on a time-based split
(last 60 days held out, no leakage).

The API forecasts several days ahead by feeding predictions back in, so models are
ranked by their **7-day recursive backtest** error, not just one-step error, and
compared against a naive "28-day average" baseline:

| Model | 1-step MAE | 7-day MAE | 7-day RMSE | 7-day MAPE |
|---|---|---|---|---|
| **Random Forest** (selected) | 7.22 | **7.87** | 10.54 | 21.5% |
| XGBoost | 7.29 | 7.94 | 10.65 | 20.9% |
| Linear Regression | 7.51 | 8.11 | 11.03 | 21.1% |
| *Baseline: 28-day average* | – | 9.45 | 12.61 | 25.4% |

The winner beats the baseline by about 17% on 7-day MAE. Training takes about 20
seconds: `python ml/train_model.py` (`--fast` for fewer trees).

The model file records the library versions it was built with. If the server
starts with different scikit-learn / XGBoost versions it retrains rather than
loading a file that may not be compatible.

## How the live system works

- **Clock and sales.** Each tick advances a simulated clock and sells the units
  that fall in that interval, following an hourly retail profile. Daily totals come
  from the same demand process as the training data, so the model is forecasting
  something it can learn.
- **Stock and orders.** Sales reduce stock (demand is still counted when stock runs
  out, and shown as lost sales). Purchase orders arrive after the supplier's lead
  time, plus a random delay according to that supplier's on-time rate.
- **Auto-reorder.** When inventory position (stock plus what is already on order)
  falls to the reorder point, an order is placed for the recommended quantity.
- **Daily close.** At midnight the day's sales become a new history row and
  forecasts are recomputed.
- **Retraining.** Every `RETRAIN_EVERY_DAYS` closed days (or on demand) the model
  is retrained in the background on all data so far, the best model is selected
  again, and the new one is swapped in without interrupting requests.
- **Persistence.** State is stored in SQLite (`backend/state/live.db`) and resumes
  after a restart.

In a 120-day simulation with 17 automatic retrains, this policy filled about 96% of
demand from stock (stable from month to month); the demo starts with several products
deliberately low, so the first month is the weakest.

### Reorder logic

1. **Forecast** daily demand for the horizon plus the supplier lead time.
2. **Safety stock** = `z × σ(demand) × √(lead time)`, with z = 1.645 (95% service level).
3. **Reorder point** = `avg daily demand × lead time + safety stock`.
4. **Inventory position** = stock on hand + units already on order.
5. **Order** nothing while the position is above the reorder point; otherwise
   `forecast demand over (lead time + horizon) + safety stock − position`, capped
   so the position never exceeds the warehouse maximum.
6. **Risk level**: CRITICAL / HIGH / WATCH / HEALTHY / OVERSTOCK. Under two days of stock is
   CRITICAL, unless an order will arrive before the stock runs out (then HIGH), since stock
   normally bottoms out just before a delivery.

## API

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | |
|---|---|
| `GET /api/snapshot?days=7` | everything the dashboard needs in one call |
| `GET /api/stream` | Server-Sent Events: `tick`, `day_closed`, `order_placed`, `order_received`, `model_retrained` |
| `GET /api/forecast/{id}?days=7` | ML demand forecast |
| `GET /api/reorder[/{id}]`, `/api/alerts`, `/api/dashboard-summary` | recommendations, alerts, KPIs |
| `GET /api/products`, `/suppliers`, `/orders`, `/events`, `/sales-history/{id}`, `/model-info`, `/status` | reference and live data |
| `POST /api/orders` | place a purchase order `{"product_id": "P001", "qty": 100}` |
| `POST /api/model/retrain` | retrain now |
| `POST /api/ingest/sale` | push a real sale `{"product_id": "P001", "units": 3}` |
| `POST /api/ingest/stock` | sync an absolute count `{"product_id": "P001", "stock": 250}` |

### Feeding in real data

```bash
LIVE_MODE=ingest API_KEY=my-secret uvicorn main:app --port 8000

curl -X POST localhost:8000/api/ingest/sale \
  -H "X-API-Key: my-secret" -H "Content-Type: application/json" \
  -d '{"product_id": "P001", "units": 3}'
```

In `ingest` mode the clock follows real time (days close at UTC midnight), and
deliveries and orders work as before.

## Configuration

Environment variables for the backend:

| Variable | Default | |
|---|---|---|
| `LIVE_MODE` | `simulate` | `simulate`, `ingest` (real sales via API) or `off` (static) |
| `TICK_SECONDS` | `5` | real seconds between ticks |
| `SIM_MINUTES_PER_TICK` | `60` | simulated minutes per tick. For true real time use `TICK_SECONDS=5 SIM_MINUTES_PER_TICK=0.0833` |
| `AUTO_REORDER` | `true` | place orders automatically at the reorder point |
| `RETRAIN_EVERY_DAYS` | `7` | `0` disables scheduled retraining |
| `RETRAIN_FAST` | `true` | fewer trees for background retrains |
| `API_KEY` | unset | if set, `/api/ingest/*` requires header `X-API-Key` |
| `LIVE_DB_PATH` | `backend/state/live.db` | where live state is stored |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `SIM_SEED`, `SIM_START_DATE` | `7`, today (UTC) | simulator seed and first live day |

Frontend: `VITE_API_BASE` is the backend URL in production (empty in development,
where Vite proxies `/api`).

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md).
