import textwrap

def generate_recommendation(player):
    """
    Lightweight RAG-style reasoning using player's own stats as context.
    """

    name = player["web_name"]
    score = player["predicted_score"]
    fdr = player["fixture_difficulty"]
    xgi = player["xGI"]
    form = player["form"]
    price = player["now_cost"]
    is_home = player["next_is_home"]

    # Basic heuristics for tone
    if score > 6:
        verdict = "Strong Buy"
    elif score > 4:
        verdict = "Consider"
    else:
        verdict = "Avoid"

    # Build reasoning summary
    reasoning = f"""
    {name} shows a predicted score of {round(score, 2)}, driven by form {form} 
    and xGI {round(xgi, 2)}. 
    The upcoming fixture difficulty is {fdr} and the match is {'at home' if is_home else 'away'}.
    At a price of £{price}, his statistical momentum suggests: **{verdict}**.
    """

    return textwrap.dedent(reasoning).strip()
