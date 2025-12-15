# services/shap_explainer.py

import numpy as np
import pandas as pd
import joblib

# Load trained model
MODEL_PATH = "models/xgb_fpl_model.pkl"
_model = joblib.load(MODEL_PATH)

FEATURES = [
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


def get_feature_attributions(player_row):
    """
    Returns normalized local feature contributions.
    These are NOT SHAP values — they are model-aware,
    scaled importance scores for explanation purposes.
    """

    # Build single-row dataframe
    X = pd.DataFrame([{f: float(player_row.get(f, 0)) for f in FEATURES}])

    # Raw feature importances from the trained model
    raw_importance = _model.feature_importances_

    # Contribution = importance * feature value
    contributions = raw_importance * X.iloc[0].values

    # Normalize to percentages
    total = np.sum(np.abs(contributions)) + 1e-9
    normalized = contributions / total

    return X, normalized
