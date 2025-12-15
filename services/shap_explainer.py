# services/shap_explainer.py

def get_shap_values(model, X):
    """
    Lazily compute SHAP values.
    Returns None if SHAP is unavailable (e.g. Streamlit Cloud).
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        return shap_values
    except Exception:
        return None
