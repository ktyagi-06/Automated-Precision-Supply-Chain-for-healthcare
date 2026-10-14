"""
Loads the trained demand model and produces N-day-ahead forecasts per
product using iterative (recursive) multi-step forecasting: each day's
prediction becomes an input lag feature for the next day's prediction.
"""
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "Models"

_model = joblib.load(MODEL_DIR / "demand_model.joblib")
_ohe = joblib.load(MODEL_DIR / "category_encoder.joblib")
with open(MODEL_DIR / "model_metadata.json") as f:
    _metadata = json.load(f)

_products = pd.read_csv(DATA_DIR / "products.csv")
_latest_state = pd.read_csv(MODEL_DIR / "latest_sales_state.csv", parse_dates=["date"])

FEATURE_COLS = _metadata["feature_columns"]
CAT_COLS = _metadata["categorical_columns"]
ALL_FEATURES = _metadata["all_feature_names"]


def _row_features(history: pd.DataFrame, target_date: pd.Timestamp,
                   product_id: str, category: str, unit_cost: float,
                   promo_flag: int = 0) -> dict:
    """Build one feature row given a rolling `history` DataFrame (date, units_sold)
    sorted ascending, ending the day before target_date."""
    series = history.set_index("date")["units_sold"]

    def lag(n):
        idx = target_date - timedelta(days=n)
        return series.get(idx, series.iloc[-n] if len(series) >= n else series.mean())

    last_vals = series.iloc[-28:]
    roll7 = series.iloc[-7:].mean()
    roll14 = series.iloc[-14:].mean()
    roll28 = last_vals.mean()
    std7 = series.iloc[-7:].std()
    if pd.isna(std7):
        std7 = 0.0

    row = {
        "lag_1": lag(1), "lag_2": lag(2), "lag_3": lag(3), "lag_7": lag(7), "lag_14": lag(14),
        "roll_mean_7": roll7, "roll_mean_14": roll14, "roll_mean_28": roll28, "roll_std_7": std7,
        "day_of_week": target_date.dayofweek,
        "day_of_month": target_date.day,
        "month": target_date.month,
        "is_weekend": int(target_date.dayofweek >= 5),
        "day_of_year_sin": np.sin(2 * np.pi * target_date.dayofyear / 365.0),
        "day_of_year_cos": np.cos(2 * np.pi * target_date.dayofyear / 365.0),
        "promo_flag": promo_flag,
        "unit_cost": unit_cost,
    }
    return row


def forecast_product(product_id: str, horizon_days: int = 7) -> dict:
    """Returns iterative daily forecast for one product over `horizon_days`."""
    prod = _products[_products["product_id"] == product_id]
    if prod.empty:
        raise ValueError(f"Unknown product_id: {product_id}")
    prod = prod.iloc[0]

    hist = _latest_state[_latest_state["product_id"] == product_id][["date", "units_sold"]].copy()
    hist = hist.sort_values("date").reset_index(drop=True)
    if hist.empty:
        raise ValueError(f"No historical state for product_id: {product_id}")

    # category lookup needed for one-hot encoding at inference time
    category = _category_for(product_id)

    last_date = hist["date"].max()
    predictions = []
    rolling_hist = hist.copy()

    for step in range(1, horizon_days + 1):
        target_date = last_date + timedelta(days=step)
        feat_row = _row_features(rolling_hist, target_date, product_id, category, prod.unit_cost)

        cat_df = pd.DataFrame([{"product_id": product_id, "category": category}])
        cat_encoded = _ohe.transform(cat_df[CAT_COLS])
        cat_names = _ohe.get_feature_names_out(CAT_COLS)
        cat_feat = dict(zip(cat_names, cat_encoded[0]))

        full_row = {**feat_row, **cat_feat}
        X = pd.DataFrame([full_row])[ALL_FEATURES]

        pred = float(_model.predict(X)[0])
        pred = max(0.0, pred)
        predictions.append({"date": target_date.strftime("%Y-%m-%d"), "predicted_units": round(pred, 1)})

        rolling_hist = pd.concat(
            [rolling_hist, pd.DataFrame([{"date": target_date, "units_sold": pred}])],
            ignore_index=True,
        )

    total = sum(p["predicted_units"] for p in predictions)
    recent_actuals = hist.tail(14)["units_sold"]
    avg_daily_demand = float(recent_actuals.mean())
    demand_std = float(recent_actuals.std()) if len(recent_actuals) > 1 else 0.0
    if pd.isna(demand_std):
        demand_std = 0.0

    return {
        "product_id": product_id,
        "product_name": prod.product_name,
        "horizon_days": horizon_days,
        "daily_forecast": predictions,
        "total_predicted_demand": round(total, 1),
        "avg_daily_demand_recent": round(avg_daily_demand, 2),
        "demand_std_recent": round(demand_std, 2),
        "model_used": _metadata["best_model"],
    }


def _category_for(product_id: str) -> str:
    row = _products[_products["product_id"] == product_id]
    return row.iloc[0]["category"] if not row.empty else "Unknown"


def get_model_metadata() -> dict:
    return _metadata
