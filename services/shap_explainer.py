# services/shap_explainer.py


import joblib
import numpy as np
import pandas as pd
from services.ml_predictor import prepare_features
def get_shap_values(model, X):
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values
# Load trained model ONCE
_model = joblib.load("models/xgb_fpl_model.pkl")

# Create SHAP explainer (tree-based = fast)
_explainer = shap.TreeExplainer(_model)


def get_shap_values(player_row):
    """
    Returns:
    - feature dataframe
    - SHAP values array
    - base value
    """
    X = prepare_features(player_row)

    shap_values = _explainer.shap_values(X)

    base_value = _explainer.expected_value

    return X, shap_values, base_value
