import pandas as pd
import numpy as np

from services.ml_predictor import predict_from_features

FEATURES = [
    "minutes",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "is_home",
    "fixture_difficulty",
    "form_1",
    "form_3",
    "form_5",
    "xGI_3"
]

def get_shap_values(player_row):
    """
    Cloud-safe local feature attribution via sensitivity analysis.
    """

    # Build baseline feature vector ONCE
    X = pd.DataFrame([{f: float(player_row.get(f, 0)) for f in FEATURES}])

    baseline_pred = predict_from_features(X)

    contributions = {}

    for feature in FEATURES:
        X_perturbed = X.copy()

        original = X_perturbed.at[0, feature]

        # meaningful perturbation
        delta = max(abs(original) * 0.1, 0.2)
        X_perturbed.at[0, feature] = original + delta

        new_pred = predict_from_features(X_perturbed)

        contributions[feature] = new_pred - baseline_pred

    return X, contributions
