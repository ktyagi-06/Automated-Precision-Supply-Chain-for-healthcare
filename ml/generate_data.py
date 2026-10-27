"""
Generates the synthetic baseline data for the Automated Precision Supply Chain:
  - products.csv           : product master data
  - suppliers.csv          : supplier master data + performance
  - sales_history.csv      : 2 years of daily sales per product (trend, weekly and
                             yearly seasonality, promo spikes, noise)
  - inventory_snapshot.csv : starting stock levels per product

Product profiles live in demand_process.py so the live simulator uses the
same demand process the model is trained on.

Run (from the repo root): python ml/generate_data.py
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from demand_process import PROFILES, START_DATE, expected_units

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

N_DAYS = 730  # 2 years of history
TODAY = START_DATE + timedelta(days=N_DAYS - 1)


def main():
    # -----------------------------------------------------------------------
    # 1. Suppliers
    # -----------------------------------------------------------------------
    suppliers = pd.DataFrame([
        {"supplier_id": "SUP-A", "supplier_name": "Supplier A", "avg_lead_time_days": 4,
         "on_time_rate": 0.96, "price_index": 1.05, "reliability_rating": "Excellent"},
        {"supplier_id": "SUP-B", "supplier_name": "Supplier B", "avg_lead_time_days": 7,
         "on_time_rate": 0.88, "price_index": 0.95, "reliability_rating": "Good"},
        {"supplier_id": "SUP-C", "supplier_name": "Supplier C", "avg_lead_time_days": 11,
         "on_time_rate": 0.71, "price_index": 0.85, "reliability_rating": "Poor"},
    ])
    suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)

    # -----------------------------------------------------------------------
    # 2. Products
    # -----------------------------------------------------------------------
    products = pd.DataFrame([{
        "product_id": p.product_id, "product_name": p.product_name, "category": p.category,
        "unit_cost": p.unit_cost, "min_stock": p.min_stock, "max_stock": p.max_stock,
        "supplier_id": p.supplier_id,
    } for p in PROFILES])
    sup_lead = suppliers.set_index("supplier_id")["avg_lead_time_days"].to_dict()
    products["lead_time_days"] = products["supplier_id"].map(sup_lead)
    products["warehouse"] = RNG.choice(["WH-North", "WH-South", "WH-Central"], size=len(products))
    products.to_csv(DATA_DIR / "products.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Daily sales history
    # -----------------------------------------------------------------------
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    rows = []
    for p in PROFILES:
        # promo calendar: ~6 random promo windows of 3-5 days over 2 years
        promo_days = set()
        for _ in range(6):
            start = RNG.integers(0, N_DAYS - 5)
            length = RNG.integers(3, 6)
            for d in range(start, start + length):
                promo_days.add(d)

        for i, date in enumerate(dates):
            promo = i in promo_days
            expected = expected_units(p, i, date, promo)
            noise = RNG.normal(0, p.noise_level * expected)
            units = max(0, round(expected + noise))
            rows.append((date.strftime("%Y-%m-%d"), p.product_id, units, 1 if promo else 0))

    sales = pd.DataFrame(rows, columns=["date", "product_id", "units_sold", "promo_flag"])
    sales.to_csv(DATA_DIR / "sales_history.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Starting inventory snapshot, deliberately varied (healthy / low /
    #    critical / overstock) so the dashboard shows real variety.
    # -----------------------------------------------------------------------
    recent = sales[sales["date"] >= (TODAY - timedelta(days=14)).strftime("%Y-%m-%d")]
    recent_avg = recent.groupby("product_id")["units_sold"].mean().to_dict()

    snapshot_rows = []
    scenario_cycle = ["healthy", "low", "critical", "overstock"]
    for idx, p in enumerate(PROFILES):
        avg_daily = recent_avg.get(p.product_id, p.base_demand)
        lead = sup_lead[p.supplier_id]
        scenario = scenario_cycle[idx % len(scenario_cycle)]
        if scenario == "healthy":
            stock = min(avg_daily * lead * RNG.uniform(1.5, 2.2), p.max_stock * 0.8)
        elif scenario == "low":
            stock = avg_daily * lead * RNG.uniform(0.9, 1.2)
        elif scenario == "critical":
            stock = avg_daily * RNG.uniform(1.0, 2.5)
        else:  # overstock
            stock = p.max_stock * RNG.uniform(0.85, 0.98)
        stock = int(max(0, min(stock, p.max_stock * 1.1)))
        snapshot_rows.append((p.product_id, stock, TODAY.strftime("%Y-%m-%d")))

    inventory = pd.DataFrame(snapshot_rows, columns=["product_id", "current_stock", "as_of_date"])
    inventory.to_csv(DATA_DIR / "inventory_snapshot.csv", index=False)

    print(f"Generated {len(sales)} sales rows for {len(PROFILES)} products over {N_DAYS} days.")
    print(f"Date range: {dates[0].date()} -> {dates[-1].date()}")
    print("Files written: products.csv, suppliers.csv, sales_history.csv, inventory_snapshot.csv")


if __name__ == "__main__":
    main()
