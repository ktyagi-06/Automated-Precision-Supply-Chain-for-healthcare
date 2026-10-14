"""
Automated Precision Supply Chain — Demand Forecasting Pipeline
================================================================
RAW DATA -> CLEANING -> FEATURE ENGINEERING -> MODEL TRAINING
-> MODEL EVALUATION -> SAVE BEST MODEL

Trains a per-product-aware demand forecaster that predicts next-day units
sold from lagged sales, rolling averages, calendar features and promo flags.
Compares Linear Regression, Random Forest and XGBoost, then persists the
best model (by validation MAE) plus metadata to /models/.

Run: python3 train_model.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
sales = pd.read_csv(DATA_DIR / "sales_history.csv", parse_dates=["date"])
products = pd.read_csv(DATA_DIR / "products.csv")

# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------
sales = sales.sort_values(["product_id", "date"]).reset_index(drop=True)
sales["units_sold"] = sales["units_sold"].clip(lower=0)
sales = sales.drop_duplicates(subset=["product_id", "date"])
# fill any missing calendar days per product with linear interpolation
full_frames = []
for pid, g in sales.groupby("product_id"):
    g = g.set_index("date").asfreq("D")
    g["product_id"] = pid
    g["units_sold"] = g["units_sold"].interpolate().round()
    g["promo_flag"] = g["promo_flag"].fillna(0)
    full_frames.append(g.reset_index())
sales = pd.concat(full_frames, ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"]).copy()
    grp = df.groupby("product_id")["units_sold"]

    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = grp.shift(lag)

    df["roll_mean_7"] = grp.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["roll_mean_14"] = grp.shift(1).rolling(14).mean().reset_index(level=0, drop=True)
    df["roll_mean_28"] = grp.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["roll_std_7"] = grp.shift(1).rolling(7).std().reset_index(level=0, drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.0)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.0)

    return df

feat = build_features(sales)
feat = feat.merge(products[["product_id", "category", "unit_cost"]], on="product_id", how="left")
feat = feat.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "day_of_year_sin", "day_of_year_cos", "promo_flag", "unit_cost",
]
TARGET = "units_sold"

# One-hot encode product_id and category (kept small/interpretable) so a
# single global model can specialize per product without training 10 models.
cat_cols = ["product_id", "category"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = ohe.fit_transform(feat[cat_cols])
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
X_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=feat.index)

X = pd.concat([feat[FEATURE_COLS], X_cat], axis=1)
y = feat[TARGET]
all_feature_names = FEATURE_COLS + cat_feature_names

# ---------------------------------------------------------------------------
# 4. Time-based train/validation split (last 60 days held out -> no leakage)
# ---------------------------------------------------------------------------
cutoff_date = feat["date"].max() - pd.Timedelta(days=60)
train_mask = feat["date"] <= cutoff_date
val_mask = feat["date"] > cutoff_date

X_train, X_val = X[train_mask], X[val_mask]
y_train, y_val = y[train_mask], y[val_mask]

print(f"Train rows: {len(X_train)}  |  Validation rows: {len(X_val)}")
print(f"Validation period: {feat.loc[val_mask, 'date'].min().date()} -> {feat.loc[val_mask, 'date'].max().date()}")

# ---------------------------------------------------------------------------
# 5. Train & evaluate candidate models
# ---------------------------------------------------------------------------
def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, objective="reg:squarederror",
    ),
}

results = []
fitted_models = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_val), 0, None)
    mae = mean_absolute_error(y_val, preds)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mp = mape(y_val.values, preds)
    results.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mp, 2)})
    fitted_models[name] = model
    print(f"{name:20s}  MAE={mae:6.2f}  RMSE={rmse:6.2f}  MAPE={mp:6.2f}%")

results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
best_name = results_df.iloc[0]["model"]
best_model = fitted_models[best_name]
print(f"\nSelected model: {best_name} (lowest validation MAE)")

# ---------------------------------------------------------------------------
# 6. Feature importance (for tree models) for interpretability
# ---------------------------------------------------------------------------
importance = None
if hasattr(best_model, "feature_importances_"):
    importance = (
        pd.Series(best_model.feature_importances_, index=all_feature_names)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\nTop features:")
    print(importance)

# ---------------------------------------------------------------------------
# 7. Refit best model on ALL data (train+val) for production use
# ---------------------------------------------------------------------------
final_model = type(best_model)(**best_model.get_params())
final_model.fit(X, y)

# ---------------------------------------------------------------------------
# 8. Persist model + encoder + metadata
# ---------------------------------------------------------------------------
joblib.dump(final_model, MODEL_DIR / "demand_model.joblib")
joblib.dump(ohe, MODEL_DIR / "category_encoder.joblib")

metadata = {
    "best_model": best_name,
    "feature_columns": FEATURE_COLS,
    "categorical_columns": cat_cols,
    "all_feature_names": all_feature_names,
    "model_comparison": results,
    "trained_at": pd.Timestamp.now().isoformat(),
    "training_rows": int(len(X)),
    "validation_period": [
        str(feat.loc[val_mask, "date"].min().date()),
        str(feat.loc[val_mask, "date"].max().date()),
    ],
    "top_features": importance.to_dict() if importance is not None else None,
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

# Also persist cleaned feature table's most recent rows per product — the
# backend uses these as the "current state" to build forward features when
# forecasting without needing to hold pandas objects in memory.
latest_state = (
    feat.sort_values("date")
    .groupby("product_id")
    .tail(28)  # need enough history for lag_14 / roll_mean_28 windows
    [["date", "product_id", "units_sold", "promo_flag"]]
)
latest_state.to_csv(MODEL_DIR / "latest_sales_state.csv", index=False)

results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

print(f"\nSaved model artifacts to {MODEL_DIR}/")
print(" - demand_model.joblib")
print(" - category_encoder.joblib")
print(" - model_metadata.json")
print(" - latest_sales_state.csv")
print(" - model_comparison.csv")
