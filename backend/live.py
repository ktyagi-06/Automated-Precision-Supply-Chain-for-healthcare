"""
Live engine: keeps the supply chain moving in real time.

* Sales stream in continuously (simulated from the same demand process the
  model was trained on, or pushed in through /api/ingest/sale).
* Stock is depleted by each sale; purchase orders are placed (automatically at
  the reorder point, or by hand) and arrive after the supplier lead time.
* At the end of each day the day's sales become a new history row, forecasts
  are recomputed, and every N days the model is retrained on the grown data.
* Every change is pushed to connected dashboards over Server-Sent Events.

State is persisted to SQLite so a restart resumes where it stopped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import optimization
from config import DATA_DIR, Settings
from demand_process import PROFILE_BY_ID, cumulative_units, simulate_day
from forecasting import ModelIncompatible, ModelStore

log = logging.getLogger("scm.live")

MAX_FORECAST_DAYS = 60       # longest horizon the API accepts
MAX_COMPUTE_DAYS = 90        # horizon + longest lead time must fit in this
WARM_FORECAST_DAYS = 45      # computed up front: covers a 30-day horizon + 11-day lead time
RETRAIN_COOLDOWN_S = 30
MAX_ORDER_QTY_FACTOR = 10  # sanity cap: no single order above 10x warehouse max


class LiveEngine:
    def __init__(self, cfg: Settings, store: ModelStore):
        self.cfg = cfg
        self.store = store
        self.lock = threading.RLock()
        self._forecast_lock = threading.Lock()
        self._retrain_lock = threading.Lock()
        self.rng = np.random.default_rng(cfg.seed + 1000)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: set[asyncio.Queue] = set()
        self.events: deque = deque(maxlen=80)  # recent sales / order events, newest first

        self.history_version = 0
        self._fc_cache: tuple | None = None  # (key, horizon, forecasts, last_date)
        self.days_since_retrain = 0
        self.retrain_state = {"running": False, "last_started": None, "last_finished": None,
                              "last_error": None, "last_reason": None}

        self._load_reference_data()
        self._open_db()
        self._init_state()

    # ------------------------------------------------------------------ setup
    def _load_reference_data(self):
        self.products_df = pd.read_csv(DATA_DIR / "products.csv")
        self.suppliers_df = pd.read_csv(DATA_DIR / "suppliers.csv")
        self.inventory_df = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
        self.baseline_sales = pd.read_csv(DATA_DIR / "sales_history.csv", parse_dates=["date"])
        merged = self.products_df.merge(self.suppliers_df, on="supplier_id")
        self.products = {r["product_id"]: r for r in merged.to_dict("records")}
        self.pids = list(self.products_df["product_id"])
        self.model_meta = {pid: (p["category"], float(p["unit_cost"])) for pid, p in self.products.items()}
        self.max_lead_time = int(self.products_df["lead_time_days"].max())

    def _open_db(self):
        path = self.cfg.db_path
        try:
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, check_same_thread=False)
            if path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")
        except (sqlite3.Error, OSError) as e:
            log.warning("cannot open %s (%s); live state will not persist", path, e)
            self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS daily_sales(date TEXT, product_id TEXT, units_sold INTEGER,
                promo_flag INTEGER, PRIMARY KEY(date, product_id));
            CREATE TABLE IF NOT EXISTS inventory(product_id TEXT PRIMARY KEY, stock INTEGER);
            CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        """)
        self.db.commit()

    def _meta_get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _meta_set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value)))

    def _init_state(self):
        base_end = self.baseline_sales["date"].max().date()
        fingerprint = [str(base_end), int(len(self.baseline_sales)),
                       self.cfg.live_mode == "simulate", self.cfg.seed]
        if self._meta_get("fingerprint") == fingerprint and self._try_restore():
            log.info("resumed live state at %s", self.sim_now)
        else:
            self._fresh_start(base_end, fingerprint)
        self._rebuild_series()
        self._begin_day()
        if self._restored_intraday:
            self.today_units, self.sim_emitted = self._restored_intraday

    def _fresh_start(self, base_end, fingerprint):
        for t in ("daily_sales", "inventory", "orders", "meta"):
            self.db.execute(f"DELETE FROM {t}")
        start = self.cfg.sim_start_date or datetime.now(timezone.utc).date()
        start = max(start, base_end + timedelta(days=1))

        # Backfill the gap between the end of the bundled history and "today"
        # with the same demand process, so history is contiguous up to now.
        rows, d = [], base_end + timedelta(days=1)
        while d < start:
            for pid in self.pids:
                units, promo = simulate_day(PROFILE_BY_ID[pid], d, self.cfg.seed)
                rows.append((d.isoformat(), pid, units, promo))
            d += timedelta(days=1)
        self.db.executemany("INSERT INTO daily_sales VALUES(?,?,?,?)", rows)
        extra = pd.DataFrame(rows, columns=["date", "product_id", "units_sold", "promo_flag"])
        extra["date"] = pd.to_datetime(extra["date"])
        self.sales_df = pd.concat([self.baseline_sales, extra], ignore_index=True)

        self.stock = {r.product_id: int(r.current_stock) for r in self.inventory_df.itertuples()}
        self.sim_now = datetime.combine(start, time.min)
        self.orders: list[dict] = []
        self.order_seq = 1
        self._restored_intraday = None
        self._meta_set("fingerprint", fingerprint)
        self._persist(commit=True)

    def _try_restore(self) -> bool:
        try:
            rows = self.db.execute("SELECT date, product_id, units_sold, promo_flag FROM daily_sales").fetchall()
            extra = pd.DataFrame(rows, columns=["date", "product_id", "units_sold", "promo_flag"])
            extra["date"] = pd.to_datetime(extra["date"])
            sales = pd.concat([self.baseline_sales, extra], ignore_index=True)
            sim_now = datetime.fromisoformat(self._meta_get("sim_now"))
            if sales["date"].max().date() != sim_now.date() - timedelta(days=1):
                return False  # history and clock disagree: start clean
            stock = {pid: s for pid, s in self.db.execute("SELECT product_id, stock FROM inventory")}
            if set(stock) != set(self.pids):
                return False
            self.sales_df, self.sim_now, self.stock = sales, sim_now, {k: int(v) for k, v in stock.items()}
            self.orders = [json.loads(d) for (d,) in self.db.execute("SELECT data FROM orders ORDER BY id")]
            self.order_seq = int(self._meta_get("order_seq", len(self.orders) + 1))
            intraday = self._meta_get("intraday")
            self._restored_intraday = (
                ({k: int(v) for k, v in intraday["today_units"].items()},
                 {k: int(v) for k, v in intraday["sim_emitted"].items()}) if intraday else None)
            return True
        except Exception as e:  # corrupt or incompatible DB
            log.warning("could not restore live state (%s); starting fresh", e)
            return False

    def _rebuild_series(self):
        df = self.sales_df.sort_values(["product_id", "date"])
        self.series = {pid: list(map(float, g["units_sold"])) for pid, g in df.groupby("product_id")}
        self.last_closed = df["date"].max().date()

    def _begin_day(self):
        """Plan today's demand (simulate mode) and reset the day's counters."""
        d = self.sim_now.date()
        self.day_total, self.day_promo = {}, {}
        for pid in self.pids:
            if self.cfg.live_mode == "simulate":
                self.day_total[pid], self.day_promo[pid] = simulate_day(PROFILE_BY_ID[pid], d, self.cfg.seed)
            else:
                self.day_total[pid], self.day_promo[pid] = 0, 0
        self.today_units = {pid: 0 for pid in self.pids}
        self.sim_emitted = {pid: 0 for pid in self.pids}

    def _persist(self, commit=False):
        self.db.executemany("INSERT OR REPLACE INTO inventory VALUES(?,?)", list(self.stock.items()))
        self._meta_set("sim_now", self.sim_now.isoformat())
        self._meta_set("order_seq", self.order_seq)
        if hasattr(self, "today_units"):
            self._meta_set("intraday", {"today_units": self.today_units, "sim_emitted": self.sim_emitted})
        self.db.commit()

    # ---------------------------------------------------------- model bootstrap
    def ensure_model(self):
        """Load the saved model; if it's missing or was built with different
        library versions, train a fresh one from the current history."""
        try:
            self.store.load()
            return
        except ModelIncompatible as e:
            log.warning("%s -> training a new model", e)
        from train_model import save, train
        with self.lock:
            sales = self.sales_df.copy()
        result = train(sales, self.products_df, fast=True, verbose=False)
        try:
            save(result, self.cfg.model_dir)
        except OSError as e:
            log.warning("could not save model: %s", e)
        self.store.swap(result)

    # ------------------------------------------------------------------ events
    def attach_loop(self, loop):
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subs.add(q)
        return q

    def unsubscribe(self, q):
        self._subs.discard(q)

    @staticmethod
    def _offer(q: asyncio.Queue, item):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(item)
            except Exception:
                pass

    def _publish(self, etype: str, payload: dict):
        if not self._loop or not self._subs:
            return
        msg = (etype, json.dumps({"type": etype, **payload}, default=str))
        for q in list(self._subs):
            self._loop.call_soon_threadsafe(self._offer, q, msg)

    # ------------------------------------------------------------------- clock
    def tick(self):
        """Advance the live system by one step."""
        if self.cfg.live_mode == "off":
            return
        new_events: list[dict] = []
        with self.lock:
            closed_before = self.last_closed
            if self.cfg.live_mode == "simulate":
                self._advance(self.cfg.sim_minutes_per_tick, new_events)
            else:
                self._advance_to_wall_clock()
            self._receive_orders()
            self._persist(commit=True)
            payload = self._tick_payload(new_events)
            day_closed = self.last_closed != closed_before
        self._publish("tick", payload)
        if day_closed:
            self.get_forecasts(WARM_FORECAST_DAYS)  # recompute once, off the lock
        if self.cfg.auto_reorder:
            self._auto_reorder()

    def _advance(self, minutes: float, events: list):
        remaining = float(minutes)
        while remaining > 1e-9:
            now = self.sim_now
            into_day = now.hour * 60 + now.minute + now.second / 60 + now.microsecond / 6e7
            to_midnight = 1440 - into_day
            adv = min(remaining, to_midnight)
            crossed = adv >= to_midnight - 1e-9
            end = datetime.combine(now.date() + timedelta(days=1), time.min) if crossed \
                else now + timedelta(minutes=adv)
            end_hour = 24.0 if crossed else (into_day + adv) / 60
            for pid in self.pids:
                target = cumulative_units(self.day_total[pid], end_hour)
                delta = target - self.sim_emitted[pid]
                if delta > 0:
                    self._sell(pid, delta, end, "sim", events)
                    self.sim_emitted[pid] = target
            self.sim_now = end
            remaining -= adv
            if crossed:
                self._close_day(end.date() - timedelta(days=1))
                self._begin_day()

    def _advance_to_wall_clock(self):
        """Ingest mode: the clock is real time; days roll over at UTC midnight."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        while now.date() > self.sim_now.date():
            self._close_day(self.sim_now.date())
            self.sim_now = datetime.combine(self.sim_now.date() + timedelta(days=1), time.min)
            self._begin_day()
        self.sim_now = max(self.sim_now, now)

    def _sell(self, pid, units, t, source, events):
        before = self.stock[pid]
        sold = min(before, units)
        self.stock[pid] = before - sold
        self.today_units[pid] += units  # demand is recorded even when stock ran out
        ev = {"t": t.isoformat(timespec="seconds"), "kind": "sale", "product_id": pid,
              "product_name": self.products[pid]["product_name"], "units": int(units),
              "stock_after": int(self.stock[pid]), "source": source, "lost": int(units - sold)}
        self.events.appendleft(ev)
        events.append(ev)

    def _close_day(self, day):
        units = {pid: self.today_units[pid] for pid in self.pids}
        if self.cfg.live_mode != "simulate" and sum(units.values()) == 0:
            # nothing reported all day (feed gap): carry forward the recent average
            units = {pid: int(round(np.mean(self.series[pid][-7:]))) for pid in self.pids}
        rows = [(day.isoformat(), pid, int(units[pid]), int(self.day_promo[pid])) for pid in self.pids]
        self.db.executemany("INSERT OR REPLACE INTO daily_sales VALUES(?,?,?,?)", rows)
        new = pd.DataFrame(rows, columns=["date", "product_id", "units_sold", "promo_flag"])
        new["date"] = pd.to_datetime(new["date"])
        self.sales_df = pd.concat([self.sales_df, new], ignore_index=True)
        for pid in self.pids:
            self.series[pid].append(float(units[pid]))
        self.last_closed = day
        self.history_version += 1
        self.days_since_retrain += 1
        self._publish("day_closed", {"date": day.isoformat(), "history_version": self.history_version})
        if self.cfg.retrain_every_days and self.days_since_retrain >= self.cfg.retrain_every_days:
            self.retrain(reason="scheduled")

    # ------------------------------------------------------------------ orders
    def _open_orders(self, pid=None):
        return [o for o in self.orders if o["status"] == "in_transit" and (pid is None or o["product_id"] == pid)]

    def place_order(self, product_id: str, qty: int, source: str = "manual") -> dict:
        qty = int(qty)
        with self.lock:
            if product_id not in self.products:
                raise KeyError(product_id)
            p = self.products[product_id]
            if qty <= 0 or qty > p["max_stock"] * MAX_ORDER_QTY_FACTOR:
                raise ValueError("quantity out of range")
            delay = 0 if self.rng.random() < p["on_time_rate"] else int(self.rng.integers(1, 4))
            eta = self.sim_now + timedelta(days=int(p["lead_time_days"]) + delay)
            order = {
                "id": self.order_seq, "product_id": product_id, "product_name": p["product_name"],
                "supplier_id": p["supplier_id"], "supplier_name": p["supplier_name"], "qty": qty,
                "unit_cost": float(p["unit_cost"]),
                "total_cost": round(qty * float(p["unit_cost"]) * float(p["price_index"]), 2),
                "placed_at": self.sim_now.isoformat(timespec="seconds"),
                "eta": eta.isoformat(timespec="seconds"), "received_at": None,
                "delay_days": delay, "status": "in_transit", "source": source,
            }
            self.order_seq += 1
            self.orders.append(order)
            self._save_order(order)
            self.db.commit()
            ev = {"t": order["placed_at"], "kind": "order_placed", "product_id": product_id,
                  "product_name": p["product_name"], "units": qty, "source": source, "order_id": order["id"]}
            self.events.appendleft(ev)
        self._publish("order_placed", {"order": order})
        return order

    def _save_order(self, order):
        self.db.execute("INSERT OR REPLACE INTO orders VALUES(?,?)", (order["id"], json.dumps(order)))

    def _receive_orders(self):
        for o in self._open_orders():
            if datetime.fromisoformat(o["eta"]) <= self.sim_now:
                self.stock[o["product_id"]] += o["qty"]
                o["status"], o["received_at"] = "received", self.sim_now.isoformat(timespec="seconds")
                self._save_order(o)
                self.events.appendleft({"t": o["received_at"], "kind": "order_received",
                                        "product_id": o["product_id"], "product_name": o["product_name"],
                                        "units": o["qty"], "source": o["source"], "order_id": o["id"]})
                self._publish("order_received", {"order": o})
        # keep memory bounded: all open orders plus the 60 most recent received
        received = [o for o in self.orders if o["status"] == "received"]
        if len(received) > 60:
            drop = {o["id"] for o in received[:-60]}
            self.orders = [o for o in self.orders if o["id"] not in drop]
            self.db.executemany("DELETE FROM orders WHERE id=?", [(i,) for i in drop])

    def _auto_reorder(self):
        try:
            for r in self.recommendations(7):
                max_stock = self.products[r["product_id"]]["max_stock"]
                # recommended_order_qty is already 0 unless stock + on-order <= reorder point
                if r["recommended_order_qty"] >= max(1, 0.05 * max_stock):
                    self.place_order(r["product_id"], r["recommended_order_qty"], source="auto")
        except Exception:
            log.exception("auto reorder failed")

    # --------------------------------------------------------------- ingestion
    def ingest_sale(self, product_id: str, units: int, source: str = "ingest") -> dict:
        if product_id not in self.products:
            raise KeyError(product_id)
        units = int(units)
        if units <= 0 or units > 100000:
            raise ValueError("units out of range")
        events: list[dict] = []
        with self.lock:
            self._sell(product_id, units, self.sim_now, source, events)
            self._persist(commit=True)
            payload = self._tick_payload(events)
        self._publish("tick", payload)
        return events[0]

    def ingest_stock(self, product_id: str, stock: int) -> dict:
        if product_id not in self.products:
            raise KeyError(product_id)
        stock = int(stock)
        if stock < 0:
            raise ValueError("stock must be >= 0")
        with self.lock:
            self.stock[product_id] = stock
            self._persist(commit=True)
            payload = self._tick_payload([])
        self._publish("tick", payload)
        return {"product_id": product_id, "stock": stock}

    # --------------------------------------------------------------- forecasts
    def get_forecasts(self, horizon: int):
        """Cached recursive forecasts for every product. One computation serves
        all horizons (a 7-day forecast is the prefix of a 30-day one) until a
        day closes or the model changes."""
        horizon = max(1, min(int(horizon), MAX_COMPUTE_DAYS))
        with self.lock:
            key = (self.history_version, self.store.version)
            c = self._fc_cache
            if c and c[0] == key and c[1] >= horizon:
                return c[2], c[3]
        with self._forecast_lock:
            with self.lock:
                key = (self.history_version, self.store.version)
                c = self._fc_cache
                if c and c[0] == key and c[1] >= horizon:
                    return c[2], c[3]
                series = {pid: np.array(v[-60:]) for pid, v in self.series.items()}
                last_date = self.last_closed
            h = max(horizon, WARM_FORECAST_DAYS)
            fc = self.store.forecast(series, last_date, self.model_meta, h)
            with self.lock:
                self._fc_cache = (key, h, fc, last_date)
            return fc, last_date

    def forecast_product(self, product_id: str, horizon: int) -> dict:
        fc, last_date = self.get_forecasts(horizon)
        with self.lock:
            recent = np.array(self.series[product_id][-14:])
            p = self.products[product_id]
        preds = fc[product_id][:horizon]
        daily = [{"date": (last_date + timedelta(days=i + 1)).isoformat(), "predicted_units": round(float(v), 1)}
                 for i, v in enumerate(preds)]
        return {
            "product_id": product_id, "product_name": p["product_name"], "horizon_days": horizon,
            "daily_forecast": daily, "total_predicted_demand": round(float(preds.sum()), 1),
            "avg_daily_demand_recent": round(float(recent.mean()), 2),
            "demand_std_recent": round(float(recent.std(ddof=1)), 2),
            "model_used": self.store.metadata.get("best_model"),
            "model_version": self.store.version,
        }

    def recommendations(self, horizon: int) -> list[dict]:
        fc, _ = self.get_forecasts(horizon + self.max_lead_time)
        out = []
        with self.lock:
            for pid in self.pids:
                p = self.products[pid]
                lead = int(p["lead_time_days"])
                recent = np.array(self.series[pid][-14:])
                open_orders = self._open_orders(pid)
                arrival = min((datetime.fromisoformat(o["eta"]) for o in open_orders), default=None)
                days_to_arrival = max(0.0, (arrival - self.sim_now).total_seconds() / 86400) if arrival else None
                rec = optimization.build_recommendation(
                    product_id=pid, product_name=p["product_name"], current_stock=self.stock[pid],
                    avg_daily_demand=float(recent.mean()), demand_std=float(recent.std(ddof=1)),
                    lead_time_days=lead,
                    predicted_demand_horizon=float(fc[pid][:horizon].sum()), horizon_days=horizon,
                    predicted_demand_coverage=float(fc[pid][:horizon + lead].sum()),
                    max_stock=float(p["max_stock"]), preferred_supplier_id=p["supplier_id"],
                    on_order_qty=sum(o["qty"] for o in open_orders),
                    days_until_arrival=days_to_arrival)
                d = rec.to_dict()
                d["next_eta"] = min((o["eta"] for o in open_orders), default=None)
                out.append(d)
        return out

    # ------------------------------------------------------------ read models
    def _tick_payload(self, new_events):
        return {
            "sim_time": self.sim_now.isoformat(timespec="seconds"),
            "last_closed": self.last_closed.isoformat(),
            "history_version": self.history_version,
            "stock": {k: int(v) for k, v in self.stock.items()},
            "on_order": {pid: sum(o["qty"] for o in self._open_orders(pid)) for pid in self.pids},
            "today_units": {k: int(v) for k, v in self.today_units.items()},
            "events": new_events[-30:],
        }

    def status(self) -> dict:
        with self.lock:
            return {
                "live_mode": self.cfg.live_mode,
                "running": self.cfg.autostart and self.cfg.live_mode != "off",
                "sim_time": self.sim_now.isoformat(timespec="seconds"),
                "last_closed": self.last_closed.isoformat(),
                "history_version": self.history_version,
                "tick_seconds": self.cfg.tick_seconds,
                "sim_minutes_per_tick": self.cfg.sim_minutes_per_tick,
                "auto_reorder": self.cfg.auto_reorder,
                "retrain_every_days": self.cfg.retrain_every_days,
                "days_since_retrain": self.days_since_retrain,
                "model_version": self.store.version,
                "retrain": dict(self.retrain_state),
                "subscribers": len(self._subs),
                "open_orders": len(self._open_orders()),
            }

    def products_with_stock(self) -> list[dict]:
        with self.lock:
            return [{
                "product_id": pid, "product_name": p["product_name"], "category": p["category"],
                "current_stock": int(self.stock[pid]), "min_stock": int(p["min_stock"]),
                "max_stock": int(p["max_stock"]), "unit_cost": float(p["unit_cost"]),
                "warehouse": p["warehouse"], "supplier_id": p["supplier_id"],
                "supplier_name": p["supplier_name"], "lead_time_days": int(p["lead_time_days"]),
            } for pid, p in self.products.items()]

    def sales_history(self, product_id: str, last_n_days: int) -> list[dict]:
        with self.lock:
            s = self.sales_df[self.sales_df["product_id"] == product_id].sort_values("date").tail(last_n_days)
        return [{"date": d.strftime("%Y-%m-%d"), "units_sold": int(u), "promo_flag": int(p)}
                for d, u, p in zip(s["date"], s["units_sold"], s["promo_flag"])]

    def orders_view(self) -> list[dict]:
        with self.lock:
            return sorted((dict(o) for o in self.orders), key=lambda o: -o["id"])

    def recent_events(self, n: int = 40) -> list[dict]:
        with self.lock:
            return list(self.events)[:n]

    def alerts(self, horizon: int) -> list[dict]:
        recs = self.recommendations(horizon)
        alerts = []
        for r in recs:
            eta = f" Order in transit, due {r['next_eta'][:10]}." if r["next_eta"] else ""
            base = {"product_id": r["product_id"], "product_name": r["product_name"]}
            if r["current_stock"] <= 0:
                alerts.append({**base, "severity": "critical", "icon": "🚨",
                               "message": "Out of stock — sales are being lost." + eta})
            elif r["risk_level"] == "CRITICAL":
                msg = (f"Predicted stockout in {r['predicted_stockout_in_days']} days"
                       if r["predicted_stockout_in_days"] is not None else "Inventory critically low")
                alerts.append({**base, "severity": "critical", "icon": "🚨", "message": msg + "." + eta})
            elif r["risk_level"] == "HIGH":
                msg = "Inventory below reorder point" + ("." if r["on_order_qty"] else " — order recommended.")
                alerts.append({**base, "severity": "high", "icon": "⚠", "message": msg + eta})
            elif r["risk_level"] == "OVERSTOCK":
                alerts.append({**base, "severity": "info", "icon": "📦",
                               "message": "Inventory significantly exceeds expected demand."})
        with self.lock:
            for pid in self.pids:
                s = self.series[pid]
                last7, prev7 = sum(s[-7:]), sum(s[-14:-7])
                if prev7 > 0 and (last7 - prev7) / prev7 * 100 >= 30:
                    alerts.append({"product_id": pid, "product_name": self.products[pid]["product_name"],
                                   "severity": "info", "icon": "📈",
                                   "message": f"Demand increased {(last7 - prev7) / prev7 * 100:.0f}% vs prior week"})
        order = {"critical": 0, "high": 1, "info": 2}
        return sorted(alerts, key=lambda a: order[a["severity"]])

    @staticmethod
    def summarize(recs: list[dict], horizon: int, total_units: int, open_orders: list[dict]) -> dict:
        breakdown: dict[str, int] = {}
        for r in recs:
            breakdown[r["risk_level"]] = breakdown.get(r["risk_level"], 0) + 1
        return {
            "total_inventory_units": total_units,
            "low_stock_products": sum(1 for r in recs if r["risk_level"] in ("HIGH", "CRITICAL")),
            "critical_products": breakdown.get("CRITICAL", 0),
            "overstock_products": breakdown.get("OVERSTOCK", 0),
            "total_predicted_demand_next_period": round(sum(r["predicted_demand_horizon"] for r in recs), 1),
            "forecast_horizon_days": horizon,
            "status_breakdown": breakdown,
            "open_orders": len(open_orders),
            "units_on_order": sum(o["qty"] for o in open_orders),
            "product_statuses": [{"product_id": r["product_id"], "product_name": r["product_name"],
                                  "risk_level": r["risk_level"]} for r in recs],
        }

    def dashboard_summary(self, horizon: int) -> dict:
        recs = self.recommendations(horizon)
        with self.lock:
            total = int(sum(self.stock.values()))
            open_orders = self._open_orders()
        return self.summarize(recs, horizon, total, open_orders)

    def snapshot(self, horizon: int) -> dict:
        """Everything the dashboard needs in one consistent read."""
        recs = self.recommendations(horizon)
        with self.lock:
            total = int(sum(self.stock.values()))
            open_orders = self._open_orders()
        return {
            "status": self.status(),
            "summary": self.summarize(recs, horizon, total, open_orders),
            "alerts": self.alerts(horizon),
            "reorders": recs,
            "orders": self.orders_view()[:25],
            "events": self.recent_events(40),
        }

    # ---------------------------------------------------------------- retrain
    def retrain(self, fast: bool | None = None, reason: str = "manual") -> dict:
        """Retrain on all history collected so far, in the background, and
        hot-swap the new model in when it's done."""
        with self._retrain_lock:
            st = self.retrain_state
            if st["running"]:
                return {"started": False, "reason": "a retrain is already running"}
            last = st["last_started"]
            if last and (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() < RETRAIN_COOLDOWN_S:
                return {"started": False, "reason": f"retrained less than {RETRAIN_COOLDOWN_S}s ago"}
            st.update(running=True, last_started=datetime.now(timezone.utc).isoformat(), last_error=None,
                      last_reason=reason)
            self.days_since_retrain = 0
        fast = self.cfg.retrain_fast if fast is None else fast
        threading.Thread(target=self._do_retrain, args=(fast,), daemon=True, name="retrain").start()
        self._publish("retrain_started", {"reason": reason})
        return {"started": True, "reason": reason}

    def _do_retrain(self, fast: bool):
        try:
            from train_model import save, train
            with self.lock:
                sales = self.sales_df.copy()
            result = train(sales, self.products_df, fast=fast, verbose=False)
            try:
                save(result, self.cfg.model_dir)
            except OSError as e:
                log.warning("retrained model not saved to disk: %s", e)
            self.store.swap(result)
            md = result.metadata
            self.retrain_state.update(last_error=None)
            self._publish("model_retrained", {
                "model_version": md["model_version"], "best_model": md["best_model"],
                "data_through": md["data_through"], "training_rows": md["training_rows"],
                "comparison": md["model_comparison"], "baseline": md.get("baseline")})
        except Exception as e:
            log.exception("retrain failed")
            self.retrain_state.update(last_error=str(e))
            self._publish("retrain_failed", {"error": str(e)})
        finally:
            self.retrain_state.update(running=False, last_finished=datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------- loop
    async def run(self):
        """Background loop: tick every `tick_seconds` until cancelled."""
        while True:
            await asyncio.sleep(self.cfg.tick_seconds)
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                log.exception("tick failed")

    def close(self):
        with self.lock:
            try:
                self._persist(commit=True)
                self.db.close()
            except sqlite3.Error:
                pass
