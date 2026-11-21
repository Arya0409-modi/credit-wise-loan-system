import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "loan_model.joblib"

FEATURE_COLUMNS = [
    "monthly_income",
    "existing_emi",
    "loan_amount",
    "tenure_months",
    "repayment_score",
    "employment_stability",
    "foir_ratio",
]

FEATURE_LABELS = {
    "monthly_income": "Monthly Income",
    "existing_emi": "Existing Monthly EMI",
    "loan_amount": "Requested Loan Amount",
    "tenure_months": "Loan Tenure (Months)",
    "repayment_score": "Repayment History Score",
    "employment_stability": "Employment Stability",
    "foir_ratio": "Debt Burden (FOIR Ratio)",
}


def calculate_monthly_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Standard reducing-balance EMI calculation."""
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    monthly_rate = annual_rate / (12 * 100)
    if monthly_rate == 0:
        return principal / tenure_months
    factor = (1 + monthly_rate) ** tenure_months
    emi = principal * monthly_rate * factor / (factor - 1)
    return round(float(emi), 2)


def generate_training_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic credit/loan dataset for ML training."""
    rng = np.random.default_rng(random_state)

    monthly_income = np.round(rng.integers(15000, 250001, size=n_samples), -3)
    existing_emi_ratio = rng.uniform(0.0, 0.45, size=n_samples)
    existing_emi = np.round(monthly_income * existing_emi_ratio, -2)

    tenure_months = rng.choice(
        [12, 24, 36, 48, 60, 84],
        size=n_samples,
        p=[0.12, 0.22, 0.25, 0.18, 0.15, 0.08],
    )
    loan_multiplier = rng.uniform(1.5, 18.0, size=n_samples)
    loan_amount = np.round(np.clip(monthly_income * loan_multiplier, 25000, 3500000), -3)

    repayment_score = rng.choice([1, 2, 3], size=n_samples, p=[0.20, 0.35, 0.45])
    employment_stability = rng.choice([1, 2, 3], size=n_samples, p=[0.22, 0.38, 0.40])

    assumed_rate = 10.5
    new_emi = np.array([
        calculate_monthly_emi(float(p), assumed_rate, int(t))
        for p, t in zip(loan_amount, tenure_months)
    ])
    foir_ratio = np.round((existing_emi + new_emi) / np.maximum(monthly_income, 1), 4)

    risk_index = (
        (monthly_income / 100000.0) * 1.4
        + (repayment_score - 2.0) * 1.35
        + (employment_stability - 2.0) * 0.85
        - (foir_ratio - 0.35) * 5.2
        - (loan_amount / np.maximum(monthly_income * 12, 1)) * 0.65
        + rng.normal(0, 0.35, size=n_samples)
    )
    approved = (risk_index > 0.15).astype(int)

    df = pd.DataFrame({
        "monthly_income": monthly_income,
        "existing_emi": existing_emi,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "repayment_score": repayment_score,
        "employment_stability": employment_stability,
        "foir_ratio": foir_ratio,
        "approved": approved,
    })
    return df


def train_and_save_model(model_path: str = MODEL_PATH) -> tuple[RandomForestClassifier, float]:
    """Train Random Forest model and persist it to disk."""
    df = generate_training_data()
    X = df[FEATURE_COLUMNS]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = round(float(accuracy_score(y_test, preds)) * 100, 2)

    joblib.dump({"model": model, "accuracy": acc}, model_path)
    return model, acc


def load_or_train_model(model_path: str = MODEL_PATH) -> tuple[RandomForestClassifier, float]:
    """Load saved model if available, otherwise train and save."""
    if os.path.exists(model_path):
        bundle = joblib.load(model_path)
        return bundle["model"], bundle.get("accuracy", 90.0)
    return train_and_save_model(model_path)


def predict_approval_probability(
    monthly_income: float,
    existing_emi: float,
    loan_amount: float,
    tenure_months: int,
    repayment_score: int,
    employment_stability: int,
    annual_rate: float = 10.5,
) -> dict:
    """Predict loan approval probability (%) and extract feature importances."""
    model, model_accuracy = load_or_train_model()

    new_emi = calculate_monthly_emi(loan_amount, annual_rate, tenure_months)
    foir_ratio = round((existing_emi + new_emi) / max(monthly_income, 1.0), 4)

    input_df = pd.DataFrame([{
        "monthly_income": float(monthly_income),
        "existing_emi": float(existing_emi),
        "loan_amount": float(loan_amount),
        "tenure_months": int(tenure_months),
        "repayment_score": int(repayment_score),
        "employment_stability": int(employment_stability),
        "foir_ratio": float(foir_ratio),
    }])[FEATURE_COLUMNS]

    proba = model.predict_proba(input_df)[0]
    approval_prob = round(float(proba[1]) * 100, 1)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[col] for col in FEATURE_COLUMNS],
        "Importance (%)": [round(float(imp) * 100, 1) for imp in importances],
    }).sort_values("Importance (%)", ascending=True)

    return {
        "approval_probability": approval_prob,
        "model_accuracy": model_accuracy,
        "foir_ratio": foir_ratio,
        "new_emi": new_emi,
        "importance_df": importance_df,
    }


if __name__ == "__main__":
    trained_model, test_acc = train_and_save_model()
    print(f"Model trained successfully! Test Accuracy: {test_acc}%")