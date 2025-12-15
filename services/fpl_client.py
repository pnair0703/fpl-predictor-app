# services/fpl_client.py

import asyncio
import aiohttp
from fpl import FPL
import pandas as pd

async def fetch_players_async():
    """
    Fetch players using the FPL API.
    `include_summary=True` gives full metadata,
    `return_json=True` returns raw JSON (easier to extract fields).
    """
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        players = await fpl.get_players(include_summary=True, return_json=True)
        return players


class FPLClient:
    """Sync wrapper for async FPL fetching (Streamlit compatible)."""

    def get_players_df(self):
        # Fetch raw data
        data = asyncio.run(fetch_players_async())

        rows = []
        for p in data:

            # Recent match history for charts
            history = p.get("history", [])
            last5 = history[-5:] if len(history) >= 5 else history

            # Extract opponent for next fixture
            fixtures = p.get("fixtures", [])
            if fixtures and len(fixtures) > 0:
                next_fixture = fixtures[0]
                next_is_home = next_fixture.get("is_home", None)

                team_h = next_fixture.get("team_h")
                team_a = next_fixture.get("team_a")
                next_opponent = team_a if next_is_home else team_h
            else:
                next_opponent = None
                next_is_home = None

            # Build row
            rows.append({
                "id": p["id"],
                "web_name": p["web_name"],
                "team": p["team"],
                "position": p["element_type"],
                "now_cost": p["now_cost"] / 10,
                "total_points": p["total_points"],
                "minutes": p["minutes"],
                "form": float(p.get("form", 0)),

                # Advanced stats
                "xG": p.get("expected_goals", 0),
                "xA": p.get("expected_assists", 0),
                "xGI": p.get("expected_goal_involvements", 0),
                "ict_index": p.get("ict_index", 0),
                "threat": p.get("threat", 0),
                "creativity": p.get("creativity", 0),
                "influence": p.get("influence", 0),
                "recent_matches_values": [float(m.get("total_points", 0)) for m in last5],



                # For graphs
                "recent_matches": last5,    
                "recent_matches_values": [m.get("total_points", 0) for m in last5],

                "team_strength": 3,  # placeholder, improves ML stability

                # Fixture difficulty ingredients
                "next_opponent": next_opponent,
                "next_is_home": next_is_home,
            })

        return pd.DataFrame(rows)
