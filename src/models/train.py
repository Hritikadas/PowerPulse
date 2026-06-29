"""
train.py
--------
Trains Linear Regression, Random Forest, and optionally XGBoost.

Usage:
    python src/models/train.py
Input:  data/processed/master_features.csv
Output: trained model .pkl files in src/models/
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

INPUT      = Path("data/processed/master_features.csv")
MODELS_DIR = Path("src/models")

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_peak",
    "mcp_lag1", "mcp_roll7_mean", "mcp_roll7_std",
    "mcv", "buy_volume", "sell_volume",
]
TARGET = "mcp"


def evaluate(name, model, X_train, X_test, y_train, y_test):
    for split, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = model.predict(X)
        mae  = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2   = r2_score(y, preds)
        print(f"  [{name}] {split:5s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")


def main():
    df = pd.read_csv(INPUT, parse_dates=["date"])
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time-ordered split
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                          random_state=42, verbosity=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        evaluate(name, model, X_train, X_test, y_train, y_test)
        out = MODELS_DIR / f"{name.lower()}.pkl"
        joblib.dump(model, out)
        print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
