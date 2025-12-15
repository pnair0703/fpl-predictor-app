# services/shap_explainer.py

import pandas as pd
import numpy as np

from services.ml_predictor import predict_player_score

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
    Cloud-safe feature attribution.
    Measures how much each feature moves the prediction
    when perturbed slightly.
    """

    # Build baseline feature vector
    X = pd.DataFrame([{f: float(player_row.get(f, 0)) for f in FEATURES}])

    baseline_pred = predict_player_score(player_row)

    contributions = {}

    for feature in FEATURES:
        perturbed = player_row.copy()

        original_value = float(player_row.get(feature, 0))

        # Small, safe perturbation
        delta = max(0.05 * abs(original_value), 0.1)

        perturbed[feature] = original_value + delta

        new_pred = predict_player_score(perturbed)

        contributions[feature] = float(new_pred - baseline_pred)

    return X, contributions
