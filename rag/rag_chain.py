from rag.live_rag import search_player_news
import textwrap

def build_rag_answer(player):
    """
    Fetches live internet news using DuckDuckGo and generates a short
    recommendation summary based on what is found.
    """
    player_name = player["web_name"]

    news_snippets = search_player_news(player_name)

    if not news_snippets:
        return f"**No recent online news found for {player_name}.**\n\n" \
               f"Based on FPL stats alone, {player_name} looks like a solid pick."

    summary = " ".join(news_snippets[:3])

    wrapped_summary = textwrap.fill(summary, width=90)

    return f"""
### News-Based Recommendation (Live Web Search)
{wrapped_summary}
"""
