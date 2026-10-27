import math

import optimization as opt


def _rec(**kw):
    base = dict(product_id="X", product_name="X", current_stock=100, avg_daily_demand=10, demand_std=3,
                lead_time_days=4, predicted_demand_horizon=70, horizon_days=7, max_stock=1000,
                preferred_supplier_id="S", predicted_demand_coverage=110)
    base.update(kw)
    return opt.build_recommendation(**base)


def test_safety_stock_and_reorder_point_formulas():
    ss = opt.compute_safety_stock(3, 4)
    assert math.isclose(ss, 1.6449 * 3 * 2)
    assert math.isclose(opt.compute_reorder_point(10, 4, ss), 40 + ss)


def test_no_order_while_position_above_reorder_point():
    assert _rec(current_stock=500).recommended_order_qty == 0


def test_order_covers_lead_time_plus_horizon():
    r = _rec(current_stock=20)
    assert r.recommended_order_qty == round(110 + r.safety_stock - 20)


def test_on_order_stock_counts_toward_position():
    base = _rec(current_stock=20)
    covered = _rec(current_stock=20, on_order_qty=base.recommended_order_qty)
    assert covered.recommended_order_qty == 0


def test_order_never_exceeds_warehouse_capacity():
    r = _rec(current_stock=0, max_stock=60)
    assert r.recommended_order_qty <= 60


def test_risk_levels():
    assert _rec(current_stock=0).risk_level == "CRITICAL"
    assert _rec(current_stock=15).risk_level == "CRITICAL"   # 1.5 days of stock
    assert _rec(current_stock=45).risk_level == "HIGH"       # below ROP
    assert _rec(current_stock=980).risk_level == "OVERSTOCK"


def test_incoming_order_downgrades_critical_to_high():
    # 1.5 days of stock left, but a delivery lands in half a day
    assert _rec(current_stock=15, days_until_arrival=0.5).risk_level == "HIGH"
    # delivery arrives after the stock would run out -> still an emergency
    assert _rec(current_stock=15, days_until_arrival=3).risk_level == "CRITICAL"
    # nothing left on the shelf is critical regardless of what is coming
    assert _rec(current_stock=0, days_until_arrival=0.1).risk_level == "CRITICAL"
