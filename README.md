
 FPL Predictor

End-to-End Machine Learning System for Forecasting Fantasy Football Performance

FPL Predictor is a production-style machine learning application that forecasts Fantasy Premier League (FPL) player returns, quantifies prediction uncertainty, and supports decision-making through explainable ML and probabilistic simulation.

This project demonstrates the full applied ML lifecycle: data ingestion, feature engineering, model training, inference, explainability, uncertainty modeling, and interactive deployment.

Live App: [https://fpl-predictor-app.streamlit.app/](https://fpl-predictor-app.streamlit.app/)
Tech Stack: Python, XGBoost, pandas, NumPy, Streamlit

---

 Key Capabilities

 1. Machine Learning-Based Player Score Prediction

* Trained an XGBoost regression model on historical FPL match-level data
* Predicts next-gameweek FPL points per player
* Features include:

  * Minutes played
  * Expected goals (xG), expected assists (xA), expected goal involvement (xGI)
  * ICT Index (Influence, Creativity, Threat)
  * Fixture difficulty and home or away indicator
  * Rolling form and rolling xGI windows
* Model inference is integrated directly into the live application

 2. Prediction Uncertainty and Risk Modeling

* Implemented Monte Carlo simulation around point predictions
* Generates a full distribution of outcomes instead of a single point estimate
* Outputs:

  * Floor (10th percentile)
  * Median expectation (50th percentile)
  * Ceiling (90th percentile)
* Enables risk-aware decision-making rather than deterministic ranking

 3. Captaincy Optimization via Simulation

* Simulates thousands of match outcomes per player
* Computes:

  * Expected captain points
  * Probability of high returns (hauls)
  * Probability of low returns (blanks)
* Ranks captain options by expected doubled return
* Demonstrates probabilistic decision modeling under uncertainty

 4. Model Explainability

* Integrated feature-level explainability to explain model predictions
* Shows relative contribution of each feature to predicted output
* Includes human-readable explanations of model behavior
* Designed to handle cloud runtime constraints gracefully


 5. Context Augmentation

* Supplements model predictions with recent football context
* Demonstrates hybrid ML plus information retrieval design
* Provides qualitative insight alongside quantitative forecasts

---

 Installation and Local Development

```bash
git clone https://github.com/pnair0703/fpl-predictor-app.git
cd fpl-predictor-app
pip install -r requirements.txt
streamlit run app.py
```

The application will run locally but is already in the streamlit cloud right now

 Example Use Cases

* Evaluating player upside versus consistency
* Optimizing captain choices using probabilistic modeling
* Understanding model predictions via feature attribution
* Demonstrating explainable ML in an interactive setting

 Future Improvements

* Bayesian uncertainty estimation
* Time-series modeling of player form
* Model ensembling
* Feature drift detection across gameweeks


## Author

Pranav Nair
Targeting Data Scientist and Machine Learning Engineer roles

