import numpy as np


def simulate_player_points(predicted, stdev, n_sims=10000):
    """
    Monte Carlo simulation for a single FPL player.
    - predicted: model's predicted points
    - stdev: estimated variance (higher for explosive players)
    """

    # Ensure numeric
    predicted = float(predicted)
    stdev = float(stdev)

    # Sample from a normal distribution
    sims = np.random.normal(loc=predicted, scale=stdev, size=n_sims)

    # Clamp negative scores to 0 (rare but possible)
    sims = np.clip(sims, 0, None)

    return sims


def captaincy_report(player, n_sims=10000):
    """
    Generates a captaincy profile for a single player:
    - expected points
    - captain expected points
    - haul probability (≥12 points)
    - blank probability (≤2 points)
    - volatility indicator
    """

    predicted = float(player["predicted_score"])
    form = float(player.get("form", 0))
    xgi = float(player.get("xGI", 0))

    # Estimate variance:
    # High xGI → higher variance (more explosive player)
    pos = player.get("pos")

    if pos == "GK":
        base_stdev = 1.2
    elif pos == "DEF":
        base_stdev = 1.6
    elif pos == "MID":
        base_stdev = 2.3
    else:  # FWD
        base_stdev = 3.0

    # Add explosiveness from attacking involvement
    explosiveness = xgi * 0.4

    stdev = max(base_stdev, base_stdev + explosiveness)

    raw_sims = simulate_player_points(predicted, stdev, n_sims)

    captain_sims = raw_sims * 2  # Doubled score

    report = {
        "player": player["web_name"],
        "predicted": predicted,
        "stdev": round(stdev, 2),
        "expected_points": round(raw_sims.mean(), 2),
        "expected_captain_points": round(captain_sims.mean(), 2),
        "haul_probability": (round((raw_sims >= 12).mean(), 3)),
        "blank_probability": round((raw_sims <= 2).mean(), 3),
        "distribution": raw_sims
    }

    return report


def compare_captains(players, n_sims=10000):
    """
    Compare multiple captaincy options head-to-head.

    players: list of player dictionaries
    returns a sorted list from best → worst captain choice.
    """

    outputs = []
    for p in players:
        outputs.append(captaincy_report(p, n_sims=n_sims))

    # Sort by expected captain return
    outputs.sort(key=lambda x: x["expected_captain_points"], reverse=True)
    return outputs
