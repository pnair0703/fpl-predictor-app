# training/train_model.py

import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1) Load training dataset
df = pd.read_csv("data/training_ready.csv")

# 2) Feature list must match what we will use at prediction time
features = [
    "minutes",
    "xG",
    "xA",
    "xGI",
    "ict_index",
    "team_strength",
    "was_home",
    "rolling_form",
    "rolling_xgi",
    "position_encoded",
    "opponent_strength",
]

target = "total_points"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.04,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)

model.fit(X_train, y_train)

# 3) Compute MSE then take sqrt manually → RMSE
mse = mean_squared_error(y_test, model.predict(X_test))
rmse = np.sqrt(mse)

print("Validation RMSE:", rmse)

# 4) Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_fpl_model.pkl")
print("Model saved → models/xgb_fpl_model.pkl")
