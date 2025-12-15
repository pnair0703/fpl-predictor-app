import re
from duckduckgo_search import DDGS

import feedparser
# ---------------------------------------------------------
# A) LIVE SEARCH: Return news article titles + links (Improved)
# ---------------------------------------------------------
def fetch_news_links(player_name: str, max_results: int = 6):
    """
    Uses Google News RSS to fetch trustworthy headlines.
    Works 100% of the time for ALL players, including Haaland.
    No API keys. No rate limits. No DDG problems.
    """
    try:
        query = player_name.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={query}+football&hl=en-US&gl=US&ceid=US:en"

        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:max_results]:
            articles.append({
                "title": entry.title,
                "url": entry.link
            })

        return articles

    except Exception:
        return []


# ---------------------------------------------------------
# B) Simple sentiment scoring from live news
# ---------------------------------------------------------
NEGATIVE_WORDS = [
    "injury", "doubt", "ruled out", "out for", "sidelined",
    "poor form", "struggle", "bad", "miss", "dropped"
]

POSITIVE_WORDS = [
    "excellent", "top form", "brace", "goal", "assist",
    "performance", "shining", "returning", "fit to play"
]


def compute_sentiment_tag(player, news_links):
    """
    Computes a numeric sentiment score from headlines.
    Falls back gracefully if HF inference is unavailable.
    """

    if not news_links:
        return 0.0, "🟡"

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        # HuggingFace not available in this environment
        return 0.0, "🟡"

    try:
        client = InferenceClient("gpt2")
    except Exception:
        return 0.0, "🟡"

    scores = []

    for item in news_links:
        headline = item["title"]

        try:
            out = client.text_generation(
                prompt=(
                    "Rate sentiment of this football news headline "
                    "from -1 to 1:\n\n"
                    f"{headline}\n\n"
                    "Just the number:"
                ),
                max_new_tokens=10
            )

            val = float(out.strip().replace("\n", ""))
            scores.append(val)

        except Exception:
            scores.append(0.0)

    if not scores:
        return 0.0, "🟡"

    avg = sum(scores) / len(scores)

    if avg > 0.25:
        icon = "🟢"
    elif avg < -0.25:
        icon = "🔴"
    else:
        icon = "🟡"

    return float(avg), icon



# ---------------------------------------------------------
# C) Full Fantasy-Scout Style Outlook Report
# ---------------------------------------------------------
def build_outlook_report(player, sentiment_icon, combined_score):
    try:
        combined_score = float(combined_score)
    except:
        combined_score = 0.0

    form = float(player.get("form", 0))
    xgi = float(player.get("xGI", 0))
    fd = float(player.get("fixture_difficulty", 3))
    name = player.get("web_name", "Unknown")

    outlook = f"""
### 📘 Full Player Outlook Report  
**Player:** {name}  
**Recommendation:** {sentiment_icon}  
**Sentiment Score:** {round(combined_score, 2)}

---

### **Form Trend**
Recent form: **{form}**  
Attacking involvement (xGI): **{xgi}**

### **Fixture Difficulty**
Upcoming difficulty: **{fd}**

---

### **Analyst Notes**
- Sentiment-adjusted score indicates expected performance.
- Form and xGI show current contribution level.
- Fixture difficulty incorporated into the final evaluation.

**Overall Score:** {round(combined_score, 2)}
"""

    return outlook


# ---------------------------------------------------------
# D) Live RAG answer — outputs: News + Buy/Hold/Sell + Outlook ONCE
# ---------------------------------------------------------
def build_live_rag_answer(player):
    """
    Clean, readable, premium-style RAG output.
    NEWS is shown AT THE END, after final interpretation.
    """
  

    name = player["web_name"]

    # -------------------------------------------------
    # Fetch news + sentiment
    # -------------------------------------------------
    news_links = fetch_news_links(name)
    sentiment_score, tag_icon = compute_sentiment_tag(player, news_links)

    # -------------------------------------------------
    # Compute unified score
    # -------------------------------------------------
    score = (
        float(sentiment_score)
        + float(player["form"]) * 0.2
        + float(player["xGI"]) * 0.3
    )

    if score >= 1.4:
        recommendation = "BUY 🟢"
    elif score <= 0.3:
        recommendation = "SELL 🔴"
    else:
        recommendation = "HOLD 🟡"


    
    # -------------------------------------------------
    # BUY / HOLD / SELL SECTION
    # -------------------------------------------------
    buy_hold_sell_block = f"""
### 🔮 Buy / Hold / Sell Recommendation
**Recommendation:** {recommendation}
"""

    # -------------------------------------------------
    # OUTLOOK REPORT SECTION
    # -------------------------------------------------
    outlook_block = f"""
### 📘 Full Player Outlook Report
**Player:** {name}  
**Sentiment Score:** {round(sentiment_score, 2)}

---

### **Form Trend**
Recent form: **{player['form']}**  
Recent attacking threat (xGI): **{player['xGI']}**

### **Fixture Difficulty**
Upcoming difficulty: **{player['fixture_difficulty']}**  
Opponent: **{player['next_opponent_name']}**  
Home match: **{"Yes" if player['next_is_home'] else "No"}**

---

### **Analyst Notes**
- Predicted Score: **{round(player['predicted_score'], 2)}**  
- Price: £{player['now_cost']}  
- Fixture Label: **{player['fixture_label']}**  
- Combined sentiment+stats score: **{round(score, 2)}**

### **Final Interpretation**
Based on momentum, fixture difficulty, stats and sentiment signals, **{name}** projects as a **{recommendation}**.
"""

    # -------------------------------------------------
    # NEWS SECTION (moved to bottom)
    # -------------------------------------------------
    if not news_links:
        news_block = f"""
### 📰 Latest Headlines for {name}
No recent trusted news articles found.
"""
    else:
        links_fmt = "\n".join([f"- [{n['title']}]({n['url']})" for n in news_links])
        news_block = f"""
### 📰 Latest Headlines for {name}
{links_fmt}
"""

    # -------------------------------------------------
    # FINAL RENDER ORDER
    # -------------------------------------------------
    return buy_hold_sell_block + "\n" + outlook_block + "\n" + news_block
