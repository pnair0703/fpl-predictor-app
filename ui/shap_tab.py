# ui/shap_tab.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from services.shap_explainer import get_shap_values

FEATURE_EXPLANATIONS = {
    "minutes": "Minutes played in recent matches. More minutes usually means more chances to score FPL points.",
    "expected_goals": "Expected goals (xG). Measures the quality of scoring chances the player gets.",
    "expected_assists": "Expected assists (xA). Measures how likely the player is to assist goals.",
    "expected_goal_involvements": "xG + xA combined. Overall attacking involvement.",
    "ict_index": "Influence, Creativity, Threat index. FPL’s composite metric for overall attacking impact.",
    "is_home": "Whether the next match is at home. Home fixtures generally boost performance.",
    "fixture_difficulty": "Difficulty rating of the upcoming opponent (1 = easy, 5 = hard).",
    "form_1": "Player’s form in the most recent match.",
    "form_3": "Average form across the last 3 matches.",
    "form_5": "Average form across the last 5 matches.",
    "xGI_3": "Average expected goal involvement over the last 3 matches."
}

def display_shap_tab(player_row):
    st.header("SHAP Model Explainability")
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        st.warning(
            "SHAP explainability is unavailable in the cloud environment.\n\n"
            "This feature works locally and is included for model transparency, "
            "but Streamlit Cloud currently runs Python 3.13 which SHAP does not yet support."
        )
        return
    
    
    X, shap_values, base_value = get_shap_values(player_row)

    predicted = float(base_value + shap_values[0].sum())

    st.subheader(f"Predicted Points: {predicted:.2f}")

    st.markdown(
        """
        This chart shows how each feature contributed to the model’s prediction.
        """
    )

    # SHAP bar plot
    fig, ax = plt.subplots()
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X.iloc[0],
            feature_names=X.columns,
        ),
        show=False,
        max_display=10
    )

    st.pyplot(fig)

    # Text explanation
    explain_text = build_shap_text(X, shap_values[0], predicted)
    st.markdown(explain_text)
    st.subheader("Feature explanations")

    for feature in FEATURE_EXPLANATIONS:
        explanation = FEATURE_EXPLANATIONS.get(
            feature,
            "No explanation available for this feature."
        )

        st.markdown(f"**{feature}**")
        st.caption(explanation)


    



def build_shap_text(X, shap_vals, predicted):
    contributions = list(zip(X.columns, shap_vals))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top = contributions[:3]

    bullets = []
    for feat, val in top:
        direction = "increased" if val > 0 else "decreased"
        bullets.append(
            f"- **{feat}** {direction} the prediction by **{abs(val):.2f}** points"
        )

    return f"""
### Interpretation

The model predicts **{predicted:.2f} points** for this player.

Key drivers:
{chr(10).join(bullets)}

This helps explain *why* the model prefers or avoids this player.
"""
