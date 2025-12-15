# ui/shap_tab.py

import streamlit as st
import numpy as np
from services.shap_explainer import get_feature_attributions

FEATURE_EXPLANATIONS = {
    "minutes": "Minutes played recently. Higher minutes increase opportunity for FPL points.",
    "expected_goals": "Expected goals (xG). Measures scoring chance quality.",
    "expected_assists": "Expected assists (xA). Measures assist likelihood.",
    "expected_goal_involvements": "xG + xA. Overall attacking involvement.",
    "ict_index": "Influence, Creativity, Threat. FPL’s composite impact metric.",
    "is_home": "Home fixtures generally boost player performance.",
    "fixture_difficulty": "Opponent difficulty (1 = easy, 5 = hard).",
    "form_1": "Form from the most recent match.",
    "form_3": "Average form across last 3 matches.",
    "form_5": "Average form across last 5 matches.",
    "xGI_3": "Rolling average xGI over last 3 matches.",
}


def display_shap_tab(df):
    st.header("Model Explainability")

    selected_player = st.selectbox(
        "Select a player to explain",
        df["web_name"].tolist(),
        key="shap_player"
    )

    player_row = df[df["web_name"] == selected_player].iloc[0]

    X, contributions = get_feature_attributions(player_row)

    st.subheader("Feature Contributions (Relative Impact)")

    # Build display table
    impact_df = (
        X.T
        .rename(columns={0: "value"})
        .assign(impact=contributions)
        .assign(
            direction=lambda d: np.where(d["impact"] >= 0, "↑", "↓"),
            magnitude=lambda d: (np.abs(d["impact"]) * 100).round(1),
        )
        .sort_values("magnitude", ascending=False)
    )

    st.dataframe(
        impact_df[["value", "direction", "magnitude"]]
        .rename(columns={"magnitude": "impact (%)"}),
        use_container_width=True
    )

    st.markdown("### Interpretation")

    top = impact_df.head(3)

    bullets = []
    for idx, row in top.iterrows():
        sign = "positive" if row["impact"] > 0 else "negative"
        bullets.append(
            f"- **{idx}** was a **{sign} driver**, contributing **{row['magnitude']}%** of total model impact"
        )

    st.markdown(
        f"""
The model’s prediction for **{selected_player}** is primarily driven by:

{chr(10).join(bullets)}

This explains *why* the model favors or avoids this player.
"""
    )

    st.subheader("Feature Context")
    for feature in impact_df.index:
        explanation = FEATURE_EXPLANATIONS.get(feature)
        if explanation:
            st.markdown(f"**{feature}**")
            st.caption(explanation)

st.write(X)
