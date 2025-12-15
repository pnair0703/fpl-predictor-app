import pandas as pd
import requests

def load_fpl_history():
    """
    Downloads all available FPL history for every player from the official API.
    Produces a clean training dataset ready for ML.
    """
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    players = bootstrap["elements"]

    rows = []

    for p in players:
        history = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{p['id']}/").json()["history"]

        for gw in history:
            rows.append({
                "player_id": p["id"],
                "web_name": p["web_name"],
                "position": p["element_type"],
                "team": p["team"],
                "round": gw["round"],
                "total_points": gw["total_points"],         # LABEL
                "minutes": gw["minutes"],
                "xG": gw.get("expected_goals", 0),
                "xA": gw.get("expected_assists", 0),
                "xGI": gw.get("expected_goal_involvements", 0),
                "ict_index": gw["ict_index"],
                "was_home": gw["was_home"],
                "opponent_team": gw["opponent_team"],
                "team_strength": p["team"],
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/fpl_raw_history.csv", index=False)
    print("Saved dataset: data/fpl_raw_history.csv")

load_fpl_history()
