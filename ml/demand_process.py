"""
Demand process shared by the data generator and the live simulator.

Every product has a profile (base demand, trend, weekly/yearly seasonality,
noise). `expected_units` is the deterministic part of daily demand;
`simulate_day` adds noise and promotions *statelessly* (seeded by product and
date), so the same date always produces the same demand. That makes the live
backfill reproducible and lets the simulator reuse exactly the process the
model was trained on.

Replace the simulator with real sales by POSTing to /api/ingest/sale.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

# First day of the generated history; trend is measured from here.
START_DATE = datetime(2024, 9, 14)

# Fraction of days on which a 3-5 day promotion window starts (~6 per 2 years).
PROMO_START_PROB = 0.008
PROMO_MULT = 1.8


@dataclass(frozen=True)
class ProductProfile:
    product_id: str
    product_name: str
    category: str
    base_demand: float
    yearly_trend: float   # fractional growth per year
    weekly_amp: float
    yearly_amp: float
    noise_level: float
    unit_cost: float
    min_stock: int
    max_stock: int
    supplier_id: str


# max_stock is the warehouse limit and the order-up-to ceiling. It is sized to
# hold the supplier lead time plus ~2 weeks of cover at today's demand rate,
# plus safety stock, with 20% headroom. Demand trends compound (up to +40%/yr),
# so limits sized for the start of the history sat *below* the reorder point
# for several products and capped every order too low to ever recover.
PROFILES: list[ProductProfile] = [
    ProductProfile("P001", "Wireless Mouse",       "Electronics", 45,  0.15, 0.25, 0.10, 0.18,  8.50,  60,  1200, "SUP-A"),
    ProductProfile("P002", "Mechanical Keyboard",  "Electronics", 28,  0.20, 0.20, 0.15, 0.20, 34.00,  40,  800, "SUP-A"),
    ProductProfile("P003", "USB-C Cable 1m",       "Accessories", 90,  0.05, 0.15, 0.05, 0.15,  2.10, 120, 2400, "SUP-B"),
    ProductProfile("P004", "Bluetooth Headphones", "Electronics", 35, -0.10, 0.30, 0.35, 0.25, 22.00,  50,  450, "SUP-A"),
    ProductProfile("P005", "Laptop Stand",         "Accessories", 18,  0.30, 0.10, 0.05, 0.20, 15.00,  25,  800, "SUP-B"),
    ProductProfile("P006", "Webcam 1080p",         "Electronics", 22,  0.40, 0.15, 0.20, 0.22, 18.50,  30,  850, "SUP-A"),
    ProductProfile("P007", "Desk Mat",             "Accessories", 15,  0.08, 0.05, 0.05, 0.18,  9.00,  20,  550, "SUP-C"),
    ProductProfile("P008", "Portable SSD 1TB",     "Storage",     12,  0.35, 0.10, 0.10, 0.25, 62.00,  15,  550, "SUP-B"),
    ProductProfile("P009", "Phone Charger 20W",    "Accessories", 60, -0.05, 0.20, 0.15, 0.20,  6.20,  80, 1500, "SUP-C"),
    ProductProfile("P010", "Monitor Arm",          "Accessories", 10,  0.25, 0.05, 0.10, 0.20, 28.00,  12,  400, "SUP-B"),
]
PROFILE_BY_ID = {p.product_id: p for p in PROFILES}
PRODUCT_INDEX = {p.product_id: i for i, p in enumerate(PROFILES)}

# Share of a day's units sold in each hour (retail-like: quiet nights, busy
# late morning and early evening). Used only by the live simulator.
HOURLY_WEIGHTS = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0,
     5.5, 5.0, 4.5, 4.5, 4.5, 4.5, 5.0, 5.0, 4.0, 2.5, 1.2, 0.6]
)
HOURLY_SHARE = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if hasattr(d, "to_pydatetime"):
        return d.to_pydatetime().date()
    return d


def expected_units(profile: ProductProfile, day_index: int, d, promo: bool) -> float:
    """Deterministic expected demand for day `day_index` since START_DATE."""
    d = _to_date(d)
    t_years = day_index / 365.0
    trend_mult = (1 + profile.yearly_trend) ** t_years

    dow = d.weekday()  # 0 = Monday
    weekly_mult = 1 + profile.weekly_amp * np.sin(2 * np.pi * (dow / 7) + 1.0) * (1 if dow < 5 else 1.6)

    day_of_year = d.timetuple().tm_yday
    yearly_mult = 1 + profile.yearly_amp * np.sin(2 * np.pi * (day_of_year / 365.0) - 1.4)

    promo_mult = PROMO_MULT if promo else 1.0
    return float(profile.base_demand * trend_mult * weekly_mult * yearly_mult * promo_mult)


def promo_active(profile: ProductProfile, d, seed: int = 7) -> bool:
    """Stateless promo calendar: a window starting on any of the last 5 days
    that is still running today."""
    d = _to_date(d)
    idx = PRODUCT_INDEX[profile.product_id]
    for k in range(5):
        day = d - timedelta(days=k)
        rng = np.random.default_rng([seed, idx, 1, day.toordinal()])
        if rng.random() < PROMO_START_PROB:
            length = int(rng.integers(3, 6))
            if k < length:
                return True
    return False


def simulate_day(profile: ProductProfile, d, seed: int = 7) -> tuple[int, int]:
    """Realised (units_sold, promo_flag) for one product on one date."""
    d = _to_date(d)
    day_index = (d - START_DATE.date()).days
    promo = promo_active(profile, d, seed)
    expected = expected_units(profile, day_index, d, promo)
    rng = np.random.default_rng([seed, PRODUCT_INDEX[profile.product_id], 2, d.toordinal()])
    units = max(0, int(round(expected + rng.normal(0, profile.noise_level * expected))))
    return units, int(promo)


_CUM_SHARE = np.concatenate([[0.0], np.cumsum(HOURLY_SHARE)])


def cumulative_units(day_units: int, hour_of_day: float) -> int:
    """Units sold from midnight up to `hour_of_day` (0-24, fractional ok).
    Cumulative rounding means increments between any two times sum exactly,
    and cumulative_units(n, 24) == n: no units are created or lost."""
    return int(round(float(np.interp(hour_of_day, np.arange(25), _CUM_SHARE)) * day_units))
