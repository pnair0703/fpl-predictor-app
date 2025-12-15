# services/uncertainty.py

import numpy as np
from services.ml_predictor import predict_player_score

def predict_with_uncertainty(player_row, n_simulations=1000):
    """
    Returns P10, P50, P90 estimates using Monte Carlo simulation
    around the model's predicted score.
    """

    mean_pred = predict_player_score(player_row)

    # Conservative, FPL-realistic std deviation
    # You can tune this later
    std_dev = max(1.5, 0.35 * mean_pred)

    simulations = np.random.normal(
        loc=mean_pred,
        scale=std_dev,
        size=n_simulations
    )

    simulations = np.clip(simulations, 0, None)

    return {
        "p10": float(np.percentile(simulations, 10)),
        "p50": float(np.percentile(simulations, 50)),
        "p90": float(np.percentile(simulations, 90)),
    }
