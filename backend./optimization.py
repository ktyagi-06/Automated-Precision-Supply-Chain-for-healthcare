"""
Inventory optimization logic for the Automated Precision Supply Chain.

Implements the formulas described in the project spec:
  Safety Stock          = z * sigma_demand * sqrt(lead_time)   (statistical version)
  Reorder Point (ROP)   = (avg daily demand * lead time) + safety stock
  Expected Requirement  = predicted demand over horizon + safety stock
  Recommended Order     = max(0, Expected Requirement - Current Inventory)

A statistical safety-stock formula is used (rather than a flat buffer) so it
scales with each product's real demand volatility and lead time, which is
what makes the recommendation "precise" rather than a rule of thumb.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import math

# z-score for a target service (cycle) level, e.g. 0.95 -> 1.645
SERVICE_LEVEL_Z = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
}
DEFAULT_SERVICE_LEVEL = 0.95


@dataclass
class ReorderRecommendation:
    product_id: str
    product_name: str
    current_stock: int
    avg_daily_demand: float
    demand_std: float
    lead_time_days: int
    predicted_demand_horizon: float
    safety_stock: float
    reorder_point: float
    expected_requirement: float
    recommended_order_qty: int
    risk_level: str
    days_of_stock_remaining: float
    predicted_stockout_in_days: Optional[float]
    preferred_supplier_id: str
    service_level: float

    def to_dict(self):
        return asdict(self)


def compute_safety_stock(demand_std: float, lead_time_days: float,
                          service_level: float = DEFAULT_SERVICE_LEVEL) -> float:
    """Statistical safety stock: z * sigma_d * sqrt(L).
    Falls back to a small buffer if std is ~0 (very stable demand)."""
    z = SERVICE_LEVEL_Z.get(service_level, 1.6449)
    if demand_std <= 0:
        return 0.0
    return z * demand_std * math.sqrt(max(lead_time_days, 1))


def compute_reorder_point(avg_daily_demand: float, lead_time_days: float,
                           safety_stock: float) -> float:
    return (avg_daily_demand * lead_time_days) + safety_stock


def classify_risk(current_stock: float, reorder_point: float,
                   days_of_stock_remaining: float, max_stock: float) -> str:
    if current_stock <= 0:
        return "CRITICAL"
    if days_of_stock_remaining <= 2:
        return "CRITICAL"
    if current_stock <= reorder_point:
        return "HIGH"
    if current_stock >= max_stock * 0.95:
        return "OVERSTOCK"
    if current_stock <= reorder_point * 1.25:
        return "WATCH"
    return "HEALTHY"


def build_recommendation(
    product_id: str,
    product_name: str,
    current_stock: float,
    avg_daily_demand: float,
    demand_std: float,
    lead_time_days: int,
    predicted_demand_horizon: float,
    horizon_days: int,
    max_stock: float,
    preferred_supplier_id: str,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> ReorderRecommendation:
    safety_stock = compute_safety_stock(demand_std, lead_time_days, service_level)
    reorder_point = compute_reorder_point(avg_daily_demand, lead_time_days, safety_stock)
    expected_requirement = predicted_demand_horizon + safety_stock
    recommended_order = max(0, round(expected_requirement - current_stock))

    days_of_stock_remaining = (
        current_stock / avg_daily_demand if avg_daily_demand > 0 else float("inf")
    )
    predicted_stockout_in_days = (
        round(days_of_stock_remaining, 1)
        if avg_daily_demand > 0 and days_of_stock_remaining <= horizon_days
        else None
    )

    risk = classify_risk(current_stock, reorder_point, days_of_stock_remaining, max_stock)

    return ReorderRecommendation(
        product_id=product_id,
        product_name=product_name,
        current_stock=int(current_stock),
        avg_daily_demand=round(avg_daily_demand, 2),
        demand_std=round(demand_std, 2),
        lead_time_days=int(lead_time_days),
        predicted_demand_horizon=round(predicted_demand_horizon, 1),
        safety_stock=round(safety_stock, 1),
        reorder_point=round(reorder_point, 1),
        expected_requirement=round(expected_requirement, 1),
        recommended_order_qty=int(recommended_order),
        risk_level=risk,
        days_of_stock_remaining=round(days_of_stock_remaining, 1) if math.isfinite(days_of_stock_remaining) else -1,
        predicted_stockout_in_days=predicted_stockout_in_days,
        preferred_supplier_id=preferred_supplier_id,
        service_level=service_level,
    )
