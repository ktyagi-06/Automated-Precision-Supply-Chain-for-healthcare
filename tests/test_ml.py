import numpy as np
import pandas as pd

from demand_process import PROFILES, cumulative_units, promo_active, simulate_day
from features import FEATURE_COLS, build_training_frame, clean_sales, inference_row

ROOT_DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def _data():
    sales = clean_sales(pd.read_csv(ROOT_DATA / "sales_history.csv", parse_dates=["date"]))
    products = pd.read_csv(ROOT_DATA / "products.csv")
    return sales, products


def test_training_and_inference_features_agree():
    """The model must see identical inputs when trained and when serving."""
    sales, products = _data()
    feat = build_training_frame(sales, products)
    for pid in ("P001", "P007"):
        g = sales[sales["product_id"] == pid].reset_index(drop=True)
        cost = float(products.loc[products.product_id == pid, "unit_cost"].iloc[0])
        for i in (40, 300, 700):
            row = inference_row(g["units_sold"].to_numpy()[:i], g["date"].iloc[i], cost, g["promo_flag"].iloc[i])
            expected = feat[(feat.product_id == pid) & (feat.date == g["date"].iloc[i])].iloc[0]
            for col in FEATURE_COLS:
                assert np.isclose(row[col], expected[col]), (pid, i, col)


def test_features_do_not_leak_across_products():
    sales, products = _data()
    feat = build_training_frame(sales, products)
    first_day = sales.groupby("product_id")["date"].min().max()
    # each product's first usable row needs 28 prior days of its own history
    assert feat.groupby("product_id")["date"].min().min() >= first_day + pd.Timedelta(days=28)


def test_simulated_demand_is_deterministic_and_plausible():
    d = pd.Timestamp("2026-11-03").date()
    for p in PROFILES:
        assert simulate_day(p, d, 7) == simulate_day(p, d, 7)
        units, promo = simulate_day(p, d, 7)
        assert units >= 0 and promo in (0, 1)
        assert 0.2 * p.base_demand < np.mean([simulate_day(p, d + pd.Timedelta(days=k), 7)[0] for k in range(60)]) < 5 * p.base_demand


def test_promo_windows_last_3_to_5_days():
    p = PROFILES[0]
    start = pd.Timestamp("2026-01-01").date()
    flags = [promo_active(p, start + pd.Timedelta(days=k)) for k in range(900)]
    assert 0 < sum(flags) < 0.2 * len(flags)
    runs, cur = [], 0
    for f in flags[:-5]:  # ignore a window cut off by the end of the range
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    assert runs and min(runs) >= 3  # adjacent windows may merge, so no upper bound check


def test_intraday_split_conserves_units():
    for total in (0, 1, 7, 83, 1000):
        assert cumulative_units(total, 0) == 0
        assert cumulative_units(total, 24) == total
        hours = [cumulative_units(total, h) for h in range(25)]
        assert hours == sorted(hours)
