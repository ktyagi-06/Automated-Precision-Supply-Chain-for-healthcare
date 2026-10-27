"""
Feature engineering shared by training and serving.

`build_training_frame` (vectorised, whole table) and `inference_row` (one
product, one future day) implement the same features; tests/test_features.py
checks they agree, so the model sees identical inputs when it is trained and
when it forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14)
ROLL_WINDOWS = (7, 14, 28)
MIN_HISTORY = 28  # days of history needed to build one feature row

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "day_of_year_sin", "day_of_year_cos", "promo_flag", "unit_cost",
]
CAT_COLS = ["product_id", "category"]
TARGET = "units_sold"


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, clip negatives and fill missing calendar days
    (linear interpolation) so every product has a contiguous daily series."""
    sales = sales.sort_values(["product_id", "date"]).reset_index(drop=True)
    sales["units_sold"] = sales["units_sold"].clip(lower=0)
    sales = sales.drop_duplicates(subset=["product_id", "date"])
    frames = []
    for pid, g in sales.groupby("product_id"):
        g = g.set_index("date").asfreq("D")
        g["product_id"] = pid
        g["units_sold"] = g["units_sold"].interpolate().round()
        g["promo_flag"] = g["promo_flag"].fillna(0)
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def calendar_features(dates) -> dict:
    """Calendar features for a DatetimeIndex / datetime Series."""
    dt = pd.DatetimeIndex(dates)
    doy = dt.dayofyear.to_numpy()
    return {
        "day_of_week": dt.dayofweek.to_numpy(),
        "day_of_month": dt.day.to_numpy(),
        "month": dt.month.to_numpy(),
        "is_weekend": (dt.dayofweek.to_numpy() >= 5).astype(int),
        "day_of_year_sin": np.sin(2 * np.pi * doy / 365.0),
        "day_of_year_cos": np.cos(2 * np.pi * doy / 365.0),
    }


def build_training_frame(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling / calendar features for every (product, day), computed
    strictly *per product* so no values leak between products."""
    df = sales.sort_values(["product_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("product_id")["units_sold"]

    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    shifted = g.shift(1)  # window ends the day before the target day
    sg = shifted.groupby(df["product_id"])
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = sg.transform(lambda s, w=w: s.rolling(w).mean())
    df["roll_std_7"] = sg.transform(lambda s: s.rolling(7).std())

    for name, values in calendar_features(df["date"]).items():
        df[name] = values

    df = df.merge(products[["product_id", "category", "unit_cost"]], on="product_id", how="left")
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def inference_row(values: np.ndarray, target_date, unit_cost: float, promo_flag: int = 0) -> dict:
    """Feature row for `target_date` from the product's daily history
    `values` (oldest -> newest, ending the day before `target_date`)."""
    if len(values) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} days of history, got {len(values)}")
    v = np.asarray(values, dtype=float)
    row = {f"lag_{lag}": v[-lag] for lag in LAGS}
    for w in ROLL_WINDOWS:
        row[f"roll_mean_{w}"] = v[-w:].mean()
    row["roll_std_7"] = v[-7:].std(ddof=1)
    for name, arr in calendar_features([pd.Timestamp(target_date)]).items():
        row[name] = arr[0]
    row["promo_flag"] = promo_flag
    row["unit_cost"] = unit_cost
    return row
