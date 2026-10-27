"""
Automated Precision Supply Chain — demand forecasting pipeline
================================================================
RAW DATA -> CLEANING -> FEATURE ENGINEERING -> MODEL TRAINING
-> EVALUATION (one-step and 7-day recursive backtest) -> SAVE BEST MODEL

One global model forecasts next-day units sold per product from lagged sales,
rolling statistics, calendar features and a promo flag. Linear Regression,
Random Forest and XGBoost are compared on a time-based split (last 60 days
held out, no leakage). Because the API forecasts several days ahead by
feeding predictions back in, models are ranked by their *7-day recursive*
backtest error, not just one-step error.

Run (from the repo root):   python ml/train_model.py
The backend also imports `train` / `save` to retrain on live data.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import (CAT_COLS, FEATURE_COLS, TARGET, build_training_frame,
                      clean_sales)
from forecast_core import recursive_forecast

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODEL_DIR = ROOT_DIR / "Models"

VAL_DAYS = 60
BACKTEST_HORIZON = 7
BACKTEST_STEP = 5


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _candidates(fast: bool) -> dict:
    trees = 120 if fast else 300
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=trees, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=trees + 100, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, objective="reg:squarederror"),
    }


def _backtest(model, encoder, feature_names, series_by_pid, dates_by_pid, meta,
              cutoff: pd.Timestamp, last_date: pd.Timestamp) -> dict:
    """Recursive 7-day forecasts from several origins inside the held-out
    window, compared with what actually happened."""
    actual, predicted = [], []
    origin = cutoff
    n_origins = 0
    while origin + pd.Timedelta(days=BACKTEST_HORIZON) <= last_date:
        series = {pid: series_by_pid[pid][dates_by_pid[pid] <= origin] for pid in series_by_pid}
        fc = recursive_forecast(model, encoder, feature_names, series, origin, meta, BACKTEST_HORIZON)
        for pid in series_by_pid:
            future = series_by_pid[pid][(dates_by_pid[pid] > origin) &
                                        (dates_by_pid[pid] <= origin + pd.Timedelta(days=BACKTEST_HORIZON))]
            actual.extend(future)
            predicted.extend(fc[pid][: len(future)])
        origin += pd.Timedelta(days=BACKTEST_STEP)
        n_origins += 1
    actual, predicted = np.array(actual), np.array(predicted)
    return {
        "MAE_7d": round(float(mean_absolute_error(actual, predicted)), 2),
        "RMSE_7d": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "MAPE_7d": round(_mape(actual, predicted), 2),
        "origins": n_origins,
    }


def _naive_backtest(series_by_pid, dates_by_pid, cutoff, last_date) -> dict:
    """Baseline for context: forecast every future day as the trailing 28-day
    average. The model must beat this to be worth deploying."""
    actual, predicted = [], []
    origin = cutoff
    while origin + pd.Timedelta(days=BACKTEST_HORIZON) <= last_date:
        for pid, v in series_by_pid.items():
            d = dates_by_pid[pid]
            future = v[(d > origin) & (d <= origin + pd.Timedelta(days=BACKTEST_HORIZON))]
            actual.extend(future)
            predicted.extend([v[d <= origin][-28:].mean()] * len(future))
        origin += pd.Timedelta(days=BACKTEST_STEP)
    actual, predicted = np.array(actual), np.array(predicted)
    return {
        "model": "Baseline: 28-day average",
        "MAE_7d": round(float(mean_absolute_error(actual, predicted)), 2),
        "RMSE_7d": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "MAPE_7d": round(_mape(actual, predicted), 2),
    }


@dataclass
class TrainResult:
    model: object
    encoder: OneHotEncoder
    metadata: dict
    latest_state: pd.DataFrame
    comparison: pd.DataFrame


def train(sales: pd.DataFrame, products: pd.DataFrame, fast: bool = False,
          verbose: bool = True) -> TrainResult:
    """Train, compare and select a model. `sales` needs date, product_id,
    units_sold, promo_flag. `fast=True` uses fewer trees (used by live retraining)."""
    log = print if verbose else (lambda *a, **k: None)

    sales = clean_sales(sales)
    feat = build_training_frame(sales, products)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = encoder.fit_transform(feat[CAT_COLS])
    cat_names = encoder.get_feature_names_out(CAT_COLS).tolist()
    X = pd.concat([feat[FEATURE_COLS], pd.DataFrame(cat_encoded, columns=cat_names, index=feat.index)], axis=1)
    y = feat[TARGET]
    feature_names = FEATURE_COLS + cat_names

    last_date = feat["date"].max()
    cutoff = last_date - pd.Timedelta(days=VAL_DAYS)
    train_mask, val_mask = feat["date"] <= cutoff, feat["date"] > cutoff
    X_train, X_val, y_train, y_val = X[train_mask], X[val_mask], y[train_mask], y[val_mask]
    log(f"Train rows: {len(X_train)}  |  Validation rows: {len(X_val)}")
    log(f"Validation period: {feat.loc[val_mask, 'date'].min().date()} -> {last_date.date()}")

    meta = products.set_index("product_id")[["category", "unit_cost"]].apply(tuple, axis=1).to_dict()
    series_by_pid = {pid: g[TARGET].to_numpy(float) for pid, g in sales.groupby("product_id")}
    dates_by_pid = {pid: pd.DatetimeIndex(g["date"]) for pid, g in sales.groupby("product_id")}

    results, fitted = [], {}
    for name, model in _candidates(fast).items():
        model.fit(X_train, y_train)
        if hasattr(model, "n_jobs"):
            model.set_params(n_jobs=1)  # single-thread predict: faster for small batches
        preds = np.clip(model.predict(X_val), 0, None)
        row = {
            "model": name,
            "MAE": round(float(mean_absolute_error(y_val, preds)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_val, preds))), 2),
            "MAPE": round(_mape(y_val, preds), 2),
        }
        bt = _backtest(model, encoder, feature_names, series_by_pid, dates_by_pid, meta, cutoff, last_date)
        row.update({k: v for k, v in bt.items() if k != "origins"})
        results.append(row)
        fitted[name] = model
        log(f"{name:18s} 1-step MAE={row['MAE']:6.2f} MAPE={row['MAPE']:5.2f}% | "
            f"7-day recursive MAE={row['MAE_7d']:6.2f} MAPE={row['MAPE_7d']:5.2f}%")

    baseline = _naive_backtest(series_by_pid, dates_by_pid, cutoff, last_date)
    log(f"{'28-day average':18s} (baseline)  7-day recursive MAE={baseline['MAE_7d']:6.2f} MAPE={baseline['MAPE_7d']:5.2f}%")
    comparison = pd.DataFrame(results).sort_values("MAE_7d").reset_index(drop=True)
    best_name = comparison.iloc[0]["model"]
    best = fitted[best_name]
    log(f"\nSelected model: {best_name} (lowest 7-day recursive MAE)")

    importance = None
    if hasattr(best, "feature_importances_"):
        importance = pd.Series(best.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)

    # Refit the winner on all data for production use.
    final = type(best)(**best.get_params())
    final.fit(X, y)
    if hasattr(final, "n_jobs"):
        final.set_params(n_jobs=1)

    trained_at = pd.Timestamp.now(tz="UTC")
    metadata = {
        "model_version": trained_at.strftime("%Y%m%dT%H%M%SZ"),
        "best_model": best_name,
        "selection_metric": "MAE_7d",
        "feature_columns": FEATURE_COLS,
        "categorical_columns": CAT_COLS,
        "all_feature_names": feature_names,
        "model_comparison": results,
        "trained_at": trained_at.isoformat(),
        "training_rows": int(len(X)),
        "data_through": str(last_date.date()),
        "validation_period": [str(feat.loc[val_mask, "date"].min().date()), str(last_date.date())],
        "backtest": {"horizon_days": BACKTEST_HORIZON, "origins": bt["origins"], "step_days": BACKTEST_STEP},
        "baseline": baseline,
        "top_features": importance.to_dict() if importance is not None else None,
        "library_versions": library_versions(),
    }
    latest_state = sales.sort_values("date").groupby("product_id").tail(60)[
        ["date", "product_id", "units_sold", "promo_flag"]]
    return TrainResult(final, encoder, metadata, latest_state, comparison)


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def save(result: TrainResult, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(model_dir / "demand_model.joblib", lambda p: joblib.dump(result.model, p))
    _atomic_write(model_dir / "category_encoder.joblib", lambda p: joblib.dump(result.encoder, p))
    _atomic_write(model_dir / "model_metadata.json",
                  lambda p: p.write_text(json.dumps(result.metadata, indent=2, default=str)))
    _atomic_write(model_dir / "latest_sales_state.csv", lambda p: result.latest_state.to_csv(p, index=False))
    _atomic_write(model_dir / "model_comparison.csv", lambda p: result.comparison.to_csv(p, index=False))


def main():
    ap = argparse.ArgumentParser(description="Train the demand-forecasting model.")
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR, help="directory for model artifacts")
    ap.add_argument("--fast", action="store_true", help="fewer trees (quicker, slightly less accurate)")
    args = ap.parse_args()

    sales = pd.read_csv(DATA_DIR / "sales_history.csv", parse_dates=["date"])
    products = pd.read_csv(DATA_DIR / "products.csv")
    result = train(sales, products, fast=args.fast)
    if result.metadata["top_features"]:
        print("\nTop features:")
        print(pd.Series(result.metadata["top_features"]).round(4).to_string())
    save(result, args.out)
    print(f"\nSaved model artifacts to {args.out}/ (version {result.metadata['model_version']})")


if __name__ == "__main__":
    main()
