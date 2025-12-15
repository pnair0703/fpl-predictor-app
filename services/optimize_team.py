# services/optimize_team.py

import pandas as pd

# FPL position codes
POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

def pick_best_xi(df: pd.DataFrame):
    """
    Select a Best XI squad using a simple greedy optimizer:
    - 1 GK
    - 3 DEF
    - 3 MID
    - 1 FWD
    - 3 FLEX (highest predicted scores regardless of position)
    - Max 3 players per team
    - Budget <= 100.0
    """

    df = df.copy()

    # Convert numeric safely
    df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce").fillna(5)
    df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce").fillna(0)

    # Add human-readable positions
    df["pos"] = df["position"].map(POSITION_MAP)

    budget = 100.0
    team_count = {}  # track max 3 per club
    selected = []

    def pick(position, count):
        """Pick top players in a certain position."""
        nonlocal budget, selected, team_count

        subset = df[df["pos"] == position].sort_values("predicted_score", ascending=False)

        for _, row in subset.iterrows():
            team = row["team"]

            # max 3 per club
            if team_count.get(team, 0) >= 3:
                continue

            # enough budget?
            if budget - row["now_cost"] < 0:
                continue

            # select player
            selected.append(row)
            budget -= row["now_cost"]
            team_count[team] = team_count.get(team, 0) + 1

            if len([s for s in selected if s["pos"] == position]) == count:
                break

    # Required positions
    pick("GK", 1)
    pick("DEF", 3)
    pick("MID", 3)
    pick("FWD", 1)

    # Now fill 3 FLEX spots by highest predicted score
    flex_needed = 3
    remaining = df.sort_values("predicted_score", ascending=False)

    for _, row in remaining.iterrows():

        if len(selected) >= 11:
            break

        team = row["team"]

        if row["id"] in [s["id"] for s in selected]:
            continue
        if team_count.get(team, 0) >= 3:
            continue
        if budget - row["now_cost"] < 0:
            continue

        selected.append(row)
        budget -= row["now_cost"]
        team_count[team] = team_count.get(team, 0) + 1

    # Convert final list to DataFrame
    xi_df = pd.DataFrame(selected)
    return xi_df
