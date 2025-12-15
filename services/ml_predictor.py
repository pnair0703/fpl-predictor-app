# services/ml_predictor.py

import pandas as pd
import numpy as np
import joblib

_model = joblib.load("models/xgb_fpl_model.pkl")

def prepare_features(row):
    import numpy as np
    import pandas as pd

    matches = row.get("recent_matches", [])

    # ---- rolling form (points) ----
    pts = []
    for m in matches[-5:]:
        try:
            pts.append(float(m.get("total_points", 0)))
        except:
            pts.append(0.0)

    rolling_form = float(np.mean(pts)) if pts else float(row.get("form", 0))

    # ---- rolling xGI ----
    xgi_vals = []
    for m in matches[-3:]:
        try:
            xgi_vals.append(float(m.get("expected_goal_involvements", 0)))
        except:
            xgi_vals.append(0.0)

    rolling_xgi = float(np.mean(xgi_vals)) if xgi_vals else float(row.get("xGI", 0))

    # ---- position encoding ----
    pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    pos_encoded = pos_map.get(row.get("pos"), 0)

    data = {
        "minutes": float(row.get("minutes", 0)),
        "xG": float(row.get("xG", 0)),
        "xA": float(row.get("xA", 0)),
        "xGI": float(row.get("xGI", 0)),
        "ict_index": float(row.get("ict_index", 0)),
        "team_strength": float(row.get("team_strength", 3)),
        "was_home": 1 if row.get("next_is_home") else 0,
        "rolling_form": rolling_form,
        "rolling_xgi": rolling_xgi,
        "position_encoded": pos_encoded,
        "opponent_strength": float(row.get("fixture_difficulty", 3)),
    }

    return pd.DataFrame([data])


def predict_player_score(row):
    features = prepare_features(row)
    return float(_model.predict(features)[0])
