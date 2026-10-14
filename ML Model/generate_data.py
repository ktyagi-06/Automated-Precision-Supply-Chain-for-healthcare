"""
Generates realistic synthetic data for the Automated Precision Supply Chain project:
  - products.csv     : product master data
  - suppliers.csv     : supplier master data + performance
  - sales_history.csv : ~2 years of daily sales per product with trend, weekly
                         seasonality, yearly seasonality, promo spikes and noise
  - inventory_snapshot.csv : current stock levels per product (as of "today")

Run: python3 generate_data.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

N_DAYS = 730  # 2 years of history
START_DATE = datetime(2024, 9, 14)
TODAY = START_DATE + timedelta(days=N_DAYS - 1)

# ---------------------------------------------------------------------------
# 1. Suppliers
# ---------------------------------------------------------------------------
suppliers = pd.DataFrame([
    {"supplier_id": "SUP-A", "supplier_name": "Supplier A", "avg_lead_time_days": 4,
     "on_time_rate": 0.96, "price_index": 1.05, "reliability_rating": "Excellent"},
    {"supplier_id": "SUP-B", "supplier_name": "Supplier B", "avg_lead_time_days": 7,
     "on_time_rate": 0.88, "price_index": 0.95, "reliability_rating": "Good"},
    {"supplier_id": "SUP-C", "supplier_name": "Supplier C", "avg_lead_time_days": 11,
     "on_time_rate": 0.71, "price_index": 0.85, "reliability_rating": "Poor"},
])
suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Products
# ---------------------------------------------------------------------------
PRODUCTS = [
    # id,          name,                  category,        base_demand, trend/yr, weekly_amp, yearly_amp, noise, unit_cost, min, max, supplier
    ("P001", "Wireless Mouse",        "Electronics",   45,  0.15, 0.25, 0.10, 0.18,  8.50,  60,  400, "SUP-A"),
    ("P002", "Mechanical Keyboard",   "Electronics",   28,  0.20, 0.20, 0.15, 0.20, 34.00,  40,  260, "SUP-A"),
    ("P003", "USB-C Cable 1m",        "Accessories",   90,  0.05, 0.15, 0.05, 0.15,  2.10, 120,  800, "SUP-B"),
    ("P004", "Bluetooth Headphones",  "Electronics",   35, -0.10, 0.30, 0.35, 0.25, 22.00,  50,  320, "SUP-A"),
    ("P005", "Laptop Stand",          "Accessories",   18,  0.30, 0.10, 0.05, 0.20, 15.00,  25,  180, "SUP-B"),
    ("P006", "Webcam 1080p",          "Electronics",   22,  0.40, 0.15, 0.20, 0.22, 18.50,  30,  200, "SUP-A"),
    ("P007", "Desk Mat",              "Accessories",   15,  0.08, 0.05, 0.05, 0.18,  9.00,  20,  150, "SUP-C"),
    ("P008", "Portable SSD 1TB",      "Storage",       12,  0.35, 0.10, 0.10, 0.25, 62.00,  15,  120, "SUP-B"),
    ("P009", "Phone Charger 20W",     "Accessories",   60, -0.05, 0.20, 0.15, 0.20,  6.20,  80,  500, "SUP-C"),
    ("P010", "Monitor Arm",           "Accessories",   10,  0.25, 0.05, 0.10, 0.20, 28.00,  12,  100, "SUP-B"),
]

cols = ["product_id", "product_name", "category", "base_demand", "yearly_trend",
        "weekly_amp", "yearly_amp", "noise_level", "unit_cost", "min_stock",
        "max_stock", "supplier_id"]
products_full = pd.DataFrame(PRODUCTS, columns=cols)

sup_lead = suppliers.set_index("supplier_id")["avg_lead_time_days"].to_dict()
products_full["lead_time_days"] = products_full["supplier_id"].map(sup_lead)
products_full["warehouse"] = RNG.choice(["WH-North", "WH-South", "WH-Central"], size=len(products_full))

products_public = products_full[["product_id", "product_name", "category", "unit_cost",
                                  "min_stock", "max_stock", "supplier_id", "lead_time_days",
                                  "warehouse"]]
products_public.to_csv(DATA_DIR / "products.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Daily sales history with trend + weekly seasonality + yearly seasonality
#    + occasional promo spikes + noise + occasional stockout-truncated days
# ---------------------------------------------------------------------------
dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
rows = []

for _, p in products_full.iterrows():
    base = p.base_demand
    yearly_trend = p.yearly_trend  # fractional growth per year
    weekly_amp = p.weekly_amp
    yearly_amp = p.yearly_amp
    noise_level = p.noise_level

    # promo calendar: ~6 random promo windows of 3-5 days over 2 years
    promo_days = set()
    for _ in range(6):
        start = RNG.integers(0, N_DAYS - 5)
        length = RNG.integers(3, 6)
        for d in range(start, start + length):
            promo_days.add(d)

    for i, date in enumerate(dates):
        t_years = i / 365.0
        trend_mult = (1 + yearly_trend) ** t_years

        dow = date.dayofweek  # 0=Mon
        weekly_mult = 1 + weekly_amp * np.sin(2 * np.pi * (dow / 7) + 1.0) * (1 if dow < 5 else 1.6)

        day_of_year = date.timetuple().tm_yday
        yearly_mult = 1 + yearly_amp * np.sin(2 * np.pi * (day_of_year / 365.0) - 1.4)

        promo_mult = 1.8 if i in promo_days else 1.0

        expected = base * trend_mult * weekly_mult * yearly_mult * promo_mult
        noise = RNG.normal(0, noise_level * expected)
        units = max(0, round(expected + noise))

        rows.append((date.strftime("%Y-%m-%d"), p.product_id, units, 1 if i in promo_days else 0))

sales = pd.DataFrame(rows, columns=["date", "product_id", "units_sold", "promo_flag"])
sales.to_csv(DATA_DIR / "sales_history.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Current inventory snapshot (as of "today" = last date in history)
#    Deliberately engineered so some products are healthy, some low, some
#    overstocked, so the demo alerts have real variety.
# ---------------------------------------------------------------------------
recent = sales[sales["date"] >= (TODAY - timedelta(days=14)).strftime("%Y-%m-%d")]
recent_avg = recent.groupby("product_id")["units_sold"].mean().to_dict()

snapshot_rows = []
scenario_cycle = ["healthy", "low", "critical", "overstock"]
for idx, p in products_full.reset_index(drop=True).iterrows():
    avg_daily = recent_avg.get(p.product_id, p.base_demand)
    scenario = scenario_cycle[idx % len(scenario_cycle)]
    if scenario == "healthy":
        stock = avg_daily * p.lead_time_days * RNG.uniform(2.5, 3.5)
    elif scenario == "low":
        stock = avg_daily * p.lead_time_days * RNG.uniform(0.9, 1.2)
    elif scenario == "critical":
        stock = avg_daily * RNG.uniform(1.0, 2.5)
    else:  # overstock
        stock = p.max_stock * RNG.uniform(0.85, 0.98)
    stock = int(max(0, min(stock, p.max_stock * 1.1)))
    snapshot_rows.append((p.product_id, stock, TODAY.strftime("%Y-%m-%d")))

inventory = pd.DataFrame(snapshot_rows, columns=["product_id", "current_stock", "as_of_date"])
inventory.to_csv(DATA_DIR / "inventory_snapshot.csv", index=False)

print(f"Generated {len(sales)} sales rows for {len(products_full)} products over {N_DAYS} days.")
print(f"Date range: {dates[0].date()} -> {dates[-1].date()}  (TODAY = {TODAY.date()})")
print("Files written: products.csv, suppliers.csv, sales_history.csv, inventory_snapshot.csv")
