import pandas as pd

def build_ml_dataset():
    df = pd.read_csv("data/fpl_raw_history.csv")

    # Rolling xGI windows
    df["xGI_3"] = df.groupby("player_id")["xGI"].rolling(3).mean().reset_index(0, drop=True)
    df["xGI_5"] = df.groupby("player_id")["xGI"].rolling(5).mean().reset_index(0, drop=True)

    # Fill missing
    df = df.fillna(0)

    # One-hot encode position
    df = pd.get_dummies(df, columns=["position"], prefix="pos")

    df.to_csv("data/training_ready.csv", index=False)
    print("Saved dataset: data/training_ready.csv")

build_ml_dataset()
