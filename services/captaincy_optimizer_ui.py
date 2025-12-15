import streamlit as st
import pandas as pd
from services.captaincy_optimizer import compare_captains


def display_captaincy_optimizer(df):
    """
    df: Pandas DataFrame of all FPL players, with predicted_score already added.
    """

    st.title("📊 Captaincy Optimizer")

    st.write("Select players and run a Monte Carlo simulation to estimate expected captain points.")

    # Player selection
    player_names = df["web_name"].tolist()

    selected_names = st.multiselect(
        "Choose 2–5 players to compare:",
        options=player_names,
        default=player_names[:3],
    )

    if len(selected_names) < 2:
        st.warning("Select at least two players.")
        return

    # Extract only selected players
    selected_df = df[df["web_name"].isin(selected_names)]

    # Convert to dict format for simulations
    selected_players = selected_df.to_dict("records")

    # Run the simulation
    results = compare_captains(selected_players, n_sims=8000)

    # Convert results to DataFrame for clean display
    results_df = pd.DataFrame(results)

    # Identify the best captain
    best_pick = results_df.loc[results_df["expected_captain_points"].idxmax()]

    st.subheader("💡 Recommended Captain")
    st.markdown(
        f"""
        ## 🏆 **{best_pick['player']}**

        - Expected captain points: **{best_pick['expected_captain_points']:.2f}**
        - Haul probability (>10 pts): **{best_pick['haul_probability']*100:.1f}%**
        - Blank probability (<2 pts): **{best_pick['blank_probability']*100:.1f}%**

        This pick offers the **best combination of ceiling and consistency**.
        """
    )

    st.markdown("---")

    # Clean results table
    st.subheader("📈 Simulation Summary")
    st.dataframe(
        results_df[
            ["player", "expected_points", "expected_captain_points", "haul_probability", "blank_probability"]
        ].rename(columns={
            "player": "Player",
            "expected_points": "Exp. Points ",
            "expected_captain_points": "Exp. Captain Points ",
            "haul_probability": "Haul Chance ",
            "blank_probability": "Blank Chance ",
        })
    )

    st.markdown("---")

    # Optional: Show distribution explanation text instead of raw array
    st.subheader("📊 Interpretation Guide")
    st.markdown(
        """
        - **Expected Points** → Average points the player earns in simulation  
        - **Expected Captain Points** → Doubled points (your captain score)  
        - **Haul Chance** → Probability of scoring 10+ points  
        - **Blank Chance** → Probability of scoring fewer than 2 points  
        """
    )
