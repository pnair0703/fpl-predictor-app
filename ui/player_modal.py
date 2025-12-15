import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from streamlit.components.v1 import html as st_html

# Only use the NEW live RAG system
from rag.live_rag import build_live_rag_answer


def show_player_modal(player):
    # -------------------------------
    # Player Header Card
    # -------------------------------
    card_html = f"""
    <div style="
        background-color:#111;
        padding:20px;
        border-radius:12px;
        border:1px solid #333;
        margin-top:20px;
        margin-bottom:20px;
        font-family:Arial;
    ">
        <h2 style="color:white; margin-bottom:5px;">{player['web_name']}</h2>
        <h4 style="color:#bbb; margin-top:0;">{player['pos']}</h4>

        <p style="color:#bbb;">Team: {player['team']}</p>
        <p style="color:#bbb;">Price: £{player['now_cost']}</p>

        <p style="color:#bbb;">Next Opponent: {player['next_opponent_name']}</p>
        <p style="color:#bbb;">Fixture Difficulty: {player['fixture_difficulty']} ({player['fixture_label']})</p>
        <p style="color:#bbb;">Home Match: {"Yes" if player['next_is_home'] else "No"}</p>

        <h3 style="color:#84ff84; margin-top:10px;">Predicted Score: {round(player['predicted_score'], 2)}</h3>

        <hr style="border-color:#444;">
        <h3 style="color:white; margin-top:10px;">Advanced Stats</h3>

        <p style="color:#bbb; margin:3px 0;">xG: {player['xG']}</p>
        <p style="color:#bbb; margin:3px 0;">xA: {player['xA']}</p>
        <p style="color:#bbb; margin:3px 0;">xGI: {player['xGI']}</p>
        <p style="color:#bbb; margin:3px 0;">Threat: {player['threat']}</p>
        <p style="color:#bbb; margin:3px 0;">Creativity: {player['creativity']}</p>
        <p style="color:#bbb; margin:3px 0;">Influence: {player['influence']}</p>
        <p style="color:#bbb; margin:3px 0;">ICT Index: {player['ict_index']}</p>
    </div>
    """

    st_html(card_html, height=360)


    # -------------------------------
    # Chart 2: Rolling xGI Trend
    # -------------------------------
    if isinstance(player.get("recent_xgi_trend"), list) and len(player["recent_xgi_trend"]) > 0:
        xgi_values = player["recent_xgi_trend"]

        rolling = (
            pd.Series(xgi_values).rolling(window=3).mean().tolist()
            if len(xgi_values) >= 3 else xgi_values
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(1, len(rolling) + 1)),
            y=rolling,
            mode="lines+markers",
            line=dict(color="#ffaa00"),
            name="xGI Rolling Avg"
        ))

        fig2.update_layout(
            title="Rolling xGI Trend (Form Momentum)",
            template="plotly_dark",
            xaxis_title="Last Matches",
            yaxis_title="xGI",
            height=300,
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # -------------------------------
    # NEW LIVE RAG OUTPUT (ONLY ONCE)
    # -------------------------------
    st.markdown(build_live_rag_answer(player))



    # -------------------------------
    # Chart 1: Recent Match Points
    # -------------------------------
    if isinstance(player.get("recent_matches"), list) and len(player["recent_matches"]) > 0:
        points = [m.get("total_points", 0) for m in player["recent_matches"]]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=list(range(1, len(points) + 1)),
            y=points,
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            name="Points"
        ))

        fig1.update_layout(
            title="Recent Match Points",
            template="plotly_dark",
            xaxis_title="Last Matches",
            yaxis_title="Points",
            height=300,
        )

        st.plotly_chart(fig1, use_container_width=True)
