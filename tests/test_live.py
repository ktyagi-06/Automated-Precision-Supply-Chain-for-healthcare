import pytest

from config import Settings
from forecasting import ModelStore
from live import LiveEngine


def make_engine(**env):
    import dataclasses
    cfg = dataclasses.replace(Settings.from_env(), autostart=False, **env)
    store = ModelStore(cfg.model_dir)
    eng = LiveEngine(cfg, store)
    eng.ensure_model()
    return eng


@pytest.fixture(scope="module")
def engine():
    return make_engine(sim_minutes_per_tick=180, retrain_every_days=0)


def total_units(eng):
    return sum(eng.stock.values())


def test_history_is_contiguous_up_to_today(engine):
    assert engine.last_closed == engine.sim_now.date() - __import__("datetime").timedelta(days=1)
    dates = engine.sales_df[engine.sales_df.product_id == "P001"]["date"].sort_values()
    assert (dates.diff().dropna() == __import__("pandas").Timedelta(days=1)).all()


def test_stock_is_conserved_across_sales_and_deliveries():
    eng = make_engine(sim_minutes_per_tick=180, retrain_every_days=0, auto_reorder=False)
    start = total_units(eng)
    sold = {"n": 0}
    orig = eng._sell

    def spy(pid, units, t, src, events):
        before = eng.stock[pid]
        orig(pid, units, t, src, events)
        sold["n"] += before - eng.stock[pid]

    eng._sell = spy
    eng.place_order("P001", 100)
    for _ in range(8 * 14):  # two simulated weeks
        eng.tick()
    received = sum(o["qty"] for o in eng.orders if o["status"] == "received")
    assert received == 100
    assert total_units(eng) == start - sold["n"] + received
    assert all(v >= 0 for v in eng.stock.values())


def test_day_close_appends_history_and_invalidates_forecast_cache():
    eng = make_engine(sim_minutes_per_tick=720, retrain_every_days=0)
    rows, ver = len(eng.sales_df), eng.history_version
    fc1, last1 = eng.get_forecasts(7)
    for _ in range(2):
        eng.tick()  # 2 x 12h = one full day
    assert len(eng.sales_df) == rows + len(eng.pids)
    assert eng.history_version == ver + 1
    fc2, last2 = eng.get_forecasts(7)
    assert last2 == last1 + __import__("datetime").timedelta(days=1)


def test_ingested_sale_reduces_stock_and_records_lost_sales():
    eng = make_engine(retrain_every_days=0, auto_reorder=False)
    eng.ingest_stock("P002", 3)
    ev = eng.ingest_sale("P002", 5)
    assert eng.stock["P002"] == 0 and ev["lost"] == 2
    with pytest.raises(KeyError):
        eng.ingest_sale("NOPE", 1)
    with pytest.raises(ValueError):
        eng.ingest_sale("P002", 0)


def test_forecast_horizons_are_prefixes_of_one_another(engine):
    a = engine.forecast_product("P003", 7)["daily_forecast"]
    b = engine.forecast_product("P003", 30)["daily_forecast"]
    assert a == b[:7]


def test_order_validation(engine):
    with pytest.raises(ValueError):
        engine.place_order("P001", 0)
    with pytest.raises(KeyError):
        engine.place_order("NOPE", 5)


def test_state_survives_restart(tmp_path):
    import dataclasses
    cfg = dataclasses.replace(Settings.from_env(), autostart=False, db_path=str(tmp_path / "live.db"),
                              sim_minutes_per_tick=360, retrain_every_days=0, auto_reorder=False)
    a = LiveEngine(cfg, ModelStore(cfg.model_dir))
    a.ensure_model()
    a.place_order("P005", 40)
    for _ in range(6):
        a.tick()  # 1.5 simulated days
    snap = (a.sim_now, dict(a.stock), len(a.orders), a.last_closed, dict(a.today_units))
    a.close()
    b = LiveEngine(cfg, ModelStore(cfg.model_dir))
    b.ensure_model()
    assert (b.sim_now, dict(b.stock), len(b.orders), b.last_closed, dict(b.today_units)) == snap
    # and it keeps running from there without double-counting today's demand
    b.tick()
    assert b.sim_now > snap[0]


def test_ingest_mode_only_moves_on_pushed_data():
    from datetime import datetime, timezone
    eng = make_engine(live_mode="ingest", retrain_every_days=0, auto_reorder=False)
    before = dict(eng.stock)
    eng.tick()
    eng.tick()
    assert eng.stock == before                      # no simulated sales in ingest mode
    assert all(e["source"] != "sim" for e in eng.recent_events(80))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((eng.sim_now - now).total_seconds()) < 5   # clock follows real time
    eng.ingest_sale("P001", 4)
    assert eng.stock["P001"] == before["P001"] - 4 and eng.today_units["P001"] == 4


def test_auto_reorder_triggers_on_inventory_position_not_just_empty_pipeline():
    """A long-lead product with an order already in transit must still be
    topped up if stock + on-order is below the reorder point."""
    eng = make_engine(retrain_every_days=0, auto_reorder=False)
    eng.ingest_stock("P009", 0)
    eng.place_order("P009", 50, source="manual")          # far too small to cover demand
    rec = next(r for r in eng.recommendations(7) if r["product_id"] == "P009")
    assert rec["on_order_qty"] == 50 and rec["recommended_order_qty"] > 0
    eng.cfg = __import__("dataclasses").replace(eng.cfg, auto_reorder=True)
    eng._auto_reorder()
    auto = [o for o in eng.orders if o["product_id"] == "P009" and o["source"] == "auto"]
    assert len(auto) == 1
    eng._auto_reorder()                                    # now covered: no duplicate order
    assert len([o for o in eng.orders if o["product_id"] == "P009" and o["source"] == "auto"]) == 1
