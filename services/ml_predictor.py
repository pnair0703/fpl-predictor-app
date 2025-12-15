# services/ml_predictor.py

import numpy as np
import pandas as pd
import joblib

# Load trained model
_model = joblib.load("models/xgb_fpl_model.pkl")


def prepare_features(row):
    """
    Builds a single-row feature DataFrame for model prediction.
    Includes position-aware feature gating (GK ≠ attacker).
    """

    matches = row.get("recent_matches", [])

    # -----------------------
    # Rolling form (last 5)
    # -----------------------
    pts = []
    for m in matches[-5:]:
        try:
            pts.append(float(m.get("total_points", 0)))
        except Exception:
            pts.append(0.0)

    rolling_form = float(np.mean(pts)) if pts else float(row.get("form", 0))

    # -----------------------
    # Rolling xGI (last 3)
    # -----------------------
    xgi_vals = []
    for m in matches[-3:]:
        try:
            xgi_vals.append(float(m.get("expected_goal_involvements", 0)))
        except Exception:
            xgi_vals.append(0.0)

    rolling_xgi = float(np.mean(xgi_vals)) if xgi_vals else float(row.get("xGI", 0))

    # -----------------------
    # Position encoding
    # -----------------------
    pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    position = pos_map.get(row.get("pos"), 0)

    # -----------------------
    # Base features
    # -----------------------
    minutes = float(row.get("minutes", 0))
    xG = float(row.get("xG", 0))
    xA = float(row.get("xA", 0))
    xGI = float(row.get("xGI", 0))
    ict_index = float(row.get("ict_index", 0))
    team_strength = float(row.get("team_strength", 3))
    was_home = 1 if row.get("next_is_home") else 0
    opponent_strength = float(row.get("fixture_difficulty", 3))

    # ------------------------------------------------
    # 🔒 POSITION-AWARE FEATURE GATING (KEY FIX)
    # ------------------------------------------------
    # Goalkeepers should NEVER use attacking metrics
    if position == 1:  # GK
        xG = 0.0
        xA = 0.0
        xGI = 0.0
        rolling_xgi = 0.0

    # -----------------------
    # Final feature dict
    # -----------------------
    features = {
        "minutes": minutes,
        "xG": xG,
        "xA": xA,
        "xGI": xGI,
        "ict_index": ict_index,
        "team_strength": team_strength,
        "was_home": was_home,
        "rolling_form": rolling_form,
        "rolling_xgi": rolling_xgi,
        "position_encoded": position,
        "opponent_strength": opponent_strength,
    }

    return pd.DataFrame([features])


def predict_player_score(row):
    """
    Predicts FPL points for a single player row.
    """
    X = prepare_features(row)
    return float(_model.predict(X)[0])


def predict_from_features(X):
    """
    Predict directly from a prepared feature DataFrame.
    Used for explainability and uncertainty modules.
    """
    return float(_model.predict(X)[0])
