# ui/shap_tab.py

import streamlit as st
import pandas as pd

from services.shap_explainer import get_shap_values

FEATURE_EXPLANATIONS = {
    "minutes": "Minutes played recently. More minutes usually mean more scoring opportunities.",
    "expected_goals": "Expected goals (xG). Measures the quality of chances a player gets.",
    "expected_assists": "Expected assists (xA). Measures likelihood of assisting goals.",
    "expected_goal_involvements": "xG + xA. Overall attacking involvement.",
    "ict_index": "Influence, Creativity, Threat. FPL’s composite attacking impact metric.",
    "is_home": "Whether the upcoming match is at home. Home fixtures boost returns.",
    "fixture_difficulty": "Difficulty of the upcoming opponent (1 = easy, 5 = hard).",
    "form_1": "Form from the most recent match.",
    "form_3": "Average form over the last 3 matches.",
    "form_5": "Average form over the last 5 matches.",
    "xGI_3": "Average xGI over the last 3 matches."
}


def display_shap_tab(player_row):
    st.header("SHAP Model Explainability")

    # --- Build feature row (MUST match training columns) ---
    features = pd.DataFrame([{
        "minutes": player_row["minutes"],
        "expected_goals": player_row["xG"],
        "expected_assists": player_row["xA"],
        "expected_goal_involvements": player_row["xGI"],
        "ict_index": player_row["ict_index"],
        "is_home": int(player_row["next_is_home"]),
        "fixture_difficulty": player_row["fixture_difficulty"],
        "form_1": player_row["form"],
        "form_3": player_row["form"],
        "form_5": player_row["form"],
        "xGI_3": player_row["xGI"],
    }])

    shap_values = get_shap_values(features)

    if not shap_values:
        st.warning("Explainability unavailable for this model.")
        return

    st.subheader("Feature Contributions")

    # Sort by absolute impact
    sorted_items = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for feature, value in sorted_items:
        direction = "↑" if value >= 0 else "↓"
        st.metric(
            label=feature,
            value=f"{value:.3f}",
            delta=direction
        )

    # Interpretation
    st.subheader("Interpretation")

    top_features = sorted_items[:3]
    bullets = []

    for feat, val in top_features:
        change = "increased" if val > 0 else "decreased"
        bullets.append(
            f"- **{feat}** {change} the prediction by **{abs(val):.2f} points**"
        )

    st.markdown(
        f"""
The model’s prediction is driven primarily by the following factors:

{chr(10).join(bullets)}

This breakdown explains *why* the model favors or avoids this player.
"""
    )

    st.subheader("Feature Explanations")

    for feature, explanation in FEATURE_EXPLANATIONS.items():
        st.markdown(f"**{feature}**")
        st.caption(explanation)
