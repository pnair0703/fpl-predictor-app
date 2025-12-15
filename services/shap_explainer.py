# services/shap_explainer.py

import joblib
import pandas as pd

MODEL_PATH = "models/xgb_fpl_model.pkl"

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_shap_values(features_df: pd.DataFrame):
    """
    Returns feature contributions using XGBoost native importance.
    SHAP-style explanation without requiring shap dependency.
    """

    model = load_model()

    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")

    # Align with input feature order
    shap_like = {
        feature: score.get(feature, 0.0)
        for feature in features_df.columns
    }

    return shap_like
