# ui/uncertainty_tab.py

import streamlit as st
from services.uncertainty import predict_with_uncertainty

FEATURE_EXPLANATIONS = {
    "ict_index": "FPL composite metric combining Influence, Creativity, and Threat.",
    "xGI": "Expected Goal Involvement (xG + xA).",
    "minutes": "Total minutes played this season.",
    "fixture_difficulty": "Difficulty rating of upcoming opponent (1–5).",
    "form": "Recent FPL form based on recent returns."
}
def interpret_variance(player_row, p10, p90):
    spread = p90 - p10
    pos = player_row["pos"]

    if pos == "GK":
        if spread < 5:
            return "Low variance goalkeeper: stable floor driven by clean sheet odds."
        else:
            return "Moderate variance goalkeeper: save volume and bonus create volatility."

    if pos == "DEF":
        if spread < 6:
            return "Stable defender: reliable minutes with limited attacking upside."
        else:
            return "Attacking defender: clean sheets plus returns increase variance."

    if pos == "MID":
        if spread < 7:
            return "Consistent midfielder: steady involvement and returns."
        else:
            return "High-upside midfielder: goal involvement drives volatility."

    if pos == "FWD":
        if spread < 8:
            return "Floor forward: minutes-driven reliability."
        else:
            return "Explosive forward: goal-dependent high variance."

    return "Variance profile unavailable."

def display_uncertainty_tab(df):
    st.subheader("Prediction Uncertainty")

    selected_player = st.selectbox(
        "Select a player",
        df["web_name"].tolist(),
        key="uncertainty_player"
    )

    player_row = df[df["web_name"] == selected_player].iloc[0]

    results = predict_with_uncertainty(player_row)

    st.markdown("### Predicted Points Distribution")

    col1, col2, col3 = st.columns(3)
    col1.metric("Floor (P10)", f"{results['p10']:.2f}")
    col2.metric("Expected (P50)", f"{results['p50']:.2f}")
    col3.metric("Ceiling (P90)", f"{results['p90']:.2f}")

    st.markdown("### Interpretation")

    interpretation = interpret_variance(
    player_row,
    results["p10"],
    results["p90"]
    )

    st.info(interpretation)


    st.markdown("### Key Feature Context")
    for feature, explanation in FEATURE_EXPLANATIONS.items():
        if feature in player_row:
            st.markdown(f"**{feature}**: {explanation}")
