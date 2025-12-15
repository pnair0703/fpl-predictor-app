import streamlit as st
from services.fpl_client import FPLClient
from services.prediction import add_predicted_scores
from services.optimize_team import pick_best_xi
from ui.player_modal import show_player_modal
from utils.team_names import TEAM_NAMES
from utils.fixture_difficulty import TEAM_FDR
from services.captaincy_optimizer_ui import display_captaincy_optimizer
from ui.best_xi_pitch import display_best_xi_pitch
from services.ml_predictor import predict_player_score
from ui.shap_tab import display_shap_tab
from ui.uncertainty_tab import display_uncertainty_tab

# Position mapping
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

st.title("FPL Predictor")

# ----------------------------
# Load and prepare all data
# ----------------------------
client = FPLClient()
df = client.get_players_df()

df["pos"] = df["position"].map(POSITION_MAP)
##df = add_predicted_scores(df)
df["predicted_score"] = df.apply(lambda row: predict_player_score(row), axis=1)


df["next_opponent_name"] = df["next_opponent"].map(TEAM_NAMES)
df["fixture_difficulty"] = df["next_opponent"].map(TEAM_FDR).fillna(3)

df["fixture_label"] = df["fixture_difficulty"].apply(
    lambda x: "Easy" if x <= 2 else "Medium" if x == 3 else "Hard"
)

# ----------------------------
# Compute Best XI **before** navigation
# ----------------------------
best_xi = pick_best_xi(df)
player_row = None

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Player Overview", "Captaincy Optimizer", "Best XI Pitch", "SHAP Explainability", "Prediction Uncertainty"]
)



# ----------------------------
# PAGE 1: Player Overview
# ----------------------------
if page == "Player Overview":
    st.subheader("Top Players")

    player_names = df["web_name"].tolist()
    selected_name = st.selectbox("Select a player to view details", player_names)

    player_row = df[df["web_name"] == selected_name].iloc[0]

    show_player_modal(player_row)

   

# ----------------------------
# PAGE 2: Captaincy Optimizer
# ----------------------------
elif page == "Captaincy Optimizer":
    display_captaincy_optimizer(df)

# ----------------------------
# PAGE 3: Best XI Pitch
# ----------------------------
elif page == "Best XI Pitch":
    display_best_xi_pitch(best_xi)

elif page == "SHAP Explainability":
    st.subheader("SHAP Model Explanation")

    shap_player_name = st.selectbox(
        "Select a player to explain",
        df["web_name"].tolist(),
        key="shap_player_select"
    )

    shap_player_row = df[df["web_name"] == shap_player_name].iloc[0]

    display_shap_tab(shap_player_row)

elif page == "Prediction Uncertainty":
    display_uncertainty_tab(df)
