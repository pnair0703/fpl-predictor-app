# ui/shap_tab.py

import streamlit as st
import numpy as np
from services.shap_explainer import get_shap_values

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

    result = get_shap_values(player_row)

    if result is None:
        st.warning(
            "Explainability is unavailable in the cloud runtime "
            "due to dependency constraints."
        )
        return

    X, contributions = result

    st.subheader("Feature Contributions (Relative Impact)")

    # Normalize to percentages
    total = sum(abs(v) for v in contributions.values()) or 1
    rows = []

    for feature, value in contributions.items():
        rows.append({
            "Feature": feature,
            "Impact (%)": round(abs(value) / total * 100, 1),
            "Direction": "↑" if value > 0 else "↓"
        })

    rows.sort(key=lambda x: x["Impact (%)"], reverse=True)

    st.dataframe(rows, use_container_width=True)

    st.subheader("Interpretation")

    top = rows[:3]
    bullets = [
        f"- **{r['Feature']}** was a {'positive' if r['Direction']=='↑' else 'negative'} driver, "
        f"contributing **{r['Impact (%)']}%** of total model impact"
        for r in top
    ]

    st.markdown(
        "The model’s prediction for **{}** is primarily driven by:\n\n{}".format(
            selected_player,
            "\n".join(bullets)
        )
    )

    st.subheader("Feature Context")

    for feature, explanation in FEATURE_EXPLANATIONS.items():
        st.markdown(f"**{feature}**")
        st.caption(explanation)

