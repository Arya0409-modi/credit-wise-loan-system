# Credit-Wise Loan System

An explainable loan eligibility and credit risk assessment system built for a 5th-semester minor project. It combines rule-based credit scoring (300–900 scale), Fixed Obligation to Income Ratio (FOIR) analysis, a Random Forest approval probability model, interactive "What-If" simulation, SQLite persistence, and downloadable assessment reports.

## Key Features
- **Credit Score (300–900) & Risk Tiering:** Evaluates income, existing debt/EMI, loan amount, tenure, repayment history, and employment stability into Low, Moderate, or High Risk tiers.
- **FOIR & Amortization Breakdown:** Computes monthly EMI, total EMI burden, FOIR percentage, maximum eligible loan, and year-wise principal vs. interest repayment schedules.
- **Machine Learning Probability (`RandomForestClassifier`):** Estimates loan approval probability alongside global feature importance.
- **Interactive "What-If" Simulator:** Dynamically tests changes in loan amount and tenure against the active borrower profile.
- **SQLite History & Analytics:** Saves evaluations, supports filtering/search, includes one-click sample record seeding, and exports `.txt` assessment reports.

## Project Structure
- `app.py` — Streamlit user interface, visualizations (Plotly), presets, simulator, and history dashboard
- `scoring.py` — Rule-based credit score, FOIR calculation, risk flags, max eligible loan, and tips
- `ml_engine.py` — EMI calculation, Random Forest model training/loading (`loan_model.joblib`), and probability prediction
- `database.py` — SQLite initialization, insert, query, and delete operations (`loan_system.db`)
- `report_generator.py` — Formatted text report generator for download

## How to Run Locally
```bash
pip install streamlit pandas plotly scikit-learn joblib
streamlit run app.py