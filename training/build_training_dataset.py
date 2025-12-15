# training/build_training_dataset.py

import pandas as pd

print("Loading raw FPL history...")

df = pd.read_csv("data/fpl_raw_history.csv")

# Sort for rolling computations
df = df.sort_values(["player_id", "round"])

# Rolling features
df["rolling_form"] = df.groupby("player_id")["total_points"].transform(lambda x: x.rolling(5, min_periods=1).mean())
df["rolling_xgi"] = df.groupby("player_id")["xGI"].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Position encoding
position_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
df["position_encoded"] = df["position"].map(position_map)

# Use team strength as opponent strength
df["opponent_strength"] = df["team_strength"]

# Final training dataset
final = df[[
    "minutes",
    "xG",
    "xA",
    "xGI",
    "ict_index",
    "opponent_strength",
    "was_home",
    "rolling_form",
    "rolling_xgi",
    "position_encoded",
    "team_strength",
    "total_points"
]]

print("Saving training dataset to data/training_ready.csv ...")
final.to_csv("data/training_ready.csv", index=False)
print("Done! Shape:", final.shape)
