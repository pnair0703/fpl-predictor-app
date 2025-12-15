# services/shap_explainer.py

import pandas as pd
import joblib

# Try importing shap safely
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Load trained model
_model = joblib.load("models/xgb_fpl_model.pkl") if SHAP_AVAILABLE else None
_explainer = shap.TreeExplainer(_model) if SHAP_AVAILABLE else None


# --------------------------------------------------
# Build features EXACTLY as training schema expects
# --------------------------------------------------
FEATURE_COLUMNS = [
    "minutes",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "is_home",
    "fixture_difficulty",
    "form_1",
    "form_3",
    "form_5",
    "xGI_3",
]


def build_shap_features(player_row):
    """
    Convert live app player row into the exact
    feature schema used during training.
    """

    return pd.DataFrame([{
        "minutes": float(player_row.get("minutes", 0)),

        # 🔑 CRITICAL FIX: map live → training names
        "expected_goals": float(player_row.get("xG", 0)),
        "expected_assists": float(player_row.get("xA", 0)),
        "expected_goal_involvements": float(player_row.get("xGI", 0)),

        "ict_index": float(player_row.get("ict_index", 0)),
        "is_home": int(player_row.get("next_is_home", 0)),
        "fixture_difficulty": float(player_row.get("fixture_difficulty", 3)),

        # Rolling form features
        "form_1": float(player_row.get("form", 0)),
        "form_3": float(player_row.get("form", 0)),
        "form_5": float(player_row.get("form", 0)),

        # Rolling xGI
        "xGI_3": float(
            sum(player_row.get("recent_xgi_trend", [player_row.get("xGI", 0)])[-3:]) / 
            max(1, len(player_row.get("recent_xgi_trend", [])))
        ),
    }])[FEATURE_COLUMNS]


# --------------------------------------------------
# Public SHAP interface used by the UI
# --------------------------------------------------
def get_shap_values(player_row):
    """
    Returns (X, shap_values, base_value)
    or None if SHAP is unavailable.
    """

    if not SHAP_AVAILABLE:
        return None

    X = build_shap_features(player_row)

    shap_values = _explainer.shap_values(X)

    base_value = (
        _explainer.expected_value
        if not isinstance(_explainer.expected_value, list)
        else _explainer.expected_value[0]
    )

    return X, shap_values, base_value
