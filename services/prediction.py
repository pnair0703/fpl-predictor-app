# services/prediction.py

import pandas as pd
from utils.fixture_difficulty import TEAM_FDR

def add_predicted_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a smart 'predicted_score' column using:
    - Player form
    - xGI (expected goal involvement)
    - Cost efficiency
    - Fixture difficulty of next opponent
    Ensures all fields are numeric + safe before computing.
    """

    df = df.copy()

    # ----------------------------
    # Ensure numeric values everywhere
    # ----------------------------
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce").fillna(5)
    df["xGI"] = pd.to_numeric(df["xGI"], errors="coerce").fillna(0)

    # Avoid divide-by-zero
    df["now_cost"] = df["now_cost"].replace(0, 0.1)

    # ----------------------------
    # Fixture Difficulty Mapping
    # ----------------------------

    # next_opponent holds team_id → map it to FDR value
    df["fixture_difficulty"] = df["next_opponent"].map(TEAM_FDR).fillna(3)

    # Normalize: difficulty 1 → easiest, 5 → hardest
    # Penalty: subtract some score for harder fixtures
    difficulty_penalty = 0.20 * (df["fixture_difficulty"] - 3)

    # ----------------------------
    # Weighted Prediction Model
    # ----------------------------
    df["predicted_score"] = (
        0.55 * df["form"] +
        0.30 * df["xGI"] +
        0.15 * (1 / df["now_cost"]) -

        # Harder fixtures reduce predicted score
        difficulty_penalty
    )

    return df
