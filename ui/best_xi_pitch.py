import streamlit as st
import pandas as pd

PITCH_COLOR = "#0B6623"   # Dark green

def display_player(player):
    """Render a single player headshot + name box."""
    photo_url = (
        f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{player['id']}.png"
    )

    st.markdown(
        f"""
        <div style='text-align:center; padding:6px;'>
            <img src="{photo_url}" style="width:70px; border-radius:6px;">
            <div style="color:white; font-weight:bold; margin-top:4px;">{player['web_name']}</div>
            <div style="color:#ddd; font-size:12px;">£{player['now_cost']}</div>
            <div style="color:#84ff84; font-size:13px;">Score: {player['predicted_score']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_best_xi_pitch(best_xi: pd.DataFrame):
    """Full visual pitch layout for Best XI."""
    st.markdown("##  Next Week Best XI Pitch ")
    st.markdown("Visual lineup based on predicted score")

    # Pitch container
    st.markdown(
        f"""
        <div style="
            background:{PITCH_COLOR};
            padding:20px;
            border-radius:12px;
            border:2px solid #004d1a;
        ">
        """,
        unsafe_allow_html=True
    )

    # Group players
    gk = best_xi[best_xi["pos"] == "GK"]
    defs = best_xi[best_xi["pos"] == "DEF"]
    mids = best_xi[best_xi["pos"] == "MID"]
    fwds = best_xi[best_xi["pos"] == "FWD"]

    # GK
    st.markdown("###  Goalkeeper")
    cols = st.columns(1)
    with cols[0]:
        display_player(gk.iloc[0])

    st.markdown("---")

    # DEF
    st.markdown("###  Defenders")
    cols = st.columns(len(defs))
    for col, (_, p) in zip(cols, defs.iterrows()):
        with col:
            display_player(p)

    st.markdown("---")

    # MID
    st.markdown("###  Midfielders")
    cols = st.columns(len(mids))
    for col, (_, p) in zip(cols, mids.iterrows()):
        with col:
            display_player(p)

    st.markdown("---")

    # FWD
    st.markdown("###  Forwards")
    cols = st.columns(len(fwds))
    for col, (_, p) in zip(cols, fwds.iterrows()):
        with col:
            display_player(p)

    # Close pitch div
    st.markdown("</div>", unsafe_allow_html=True)
