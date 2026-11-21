from ml_engine import calculate_monthly_emi


def compute_max_eligible_loan(
    monthly_income: float,
    existing_emi: float,
    tenure_months: int,
    annual_rate: float = 10.5,
    max_foir_limit: float = 0.50,
) -> float:
    """Calculate maximum loan principal possible within the allowed FOIR ceiling."""
    if monthly_income <= 0 or tenure_months <= 0:
        return 0.0
    max_total_emi = monthly_income * max_foir_limit
    available_emi = max(0.0, max_total_emi - existing_emi)
    if available_emi <= 0:
        return 0.0

    monthly_rate = annual_rate / (12 * 100)
    if monthly_rate == 0:
        return round(available_emi * tenure_months, 2)

    factor = (1 + monthly_rate) ** tenure_months
    max_principal = available_emi * (factor - 1) / (monthly_rate * factor)
    return round(float(max_principal), 2)


def evaluate_credit_profile(
    monthly_income: float,
    existing_emi: float,
    loan_amount: float,
    tenure_months: int,
    repayment_score: int,
    employment_stability: int,
    annual_rate: float = 10.5,
) -> dict:
    """Compute 300-900 credit score, FOIR %, risk flags, factor breakdown, and tips."""
    safe_income = max(float(monthly_income), 1.0)
    new_emi = calculate_monthly_emi(loan_amount, annual_rate, tenure_months)
    total_emi = round(float(existing_emi) + new_emi, 2)
    foir_pct = round((total_emi / safe_income) * 100, 1)
    existing_foir_pct = round((float(existing_emi) / safe_income) * 100, 1)

    # Base score = 580, range clamped to [300, 900]
    score = 580.0

    # 1. Repayment History impact (-110 to +110)
    repayment_map = {1: -110.0, 2: 0.0, 3: 110.0}
    repayment_impact = repayment_map.get(int(repayment_score), 0.0)
    score += repayment_impact

    # 2. FOIR / Debt Burden impact (-130 to +85)
    if foir_pct <= 30.0:
        foir_impact = 85.0
    elif foir_pct <= 40.0:
        foir_impact = 40.0
    elif foir_pct <= 50.0:
        foir_impact = -25.0
    elif foir_pct <= 65.0:
        foir_impact = -80.0
    else:
        foir_impact = -130.0
    score += foir_impact

    # 3. Employment Stability impact (-65 to +65)
    stability_map = {1: -65.0, 2: 10.0, 3: 65.0}
    stability_impact = stability_map.get(int(employment_stability), 0.0)
    score += stability_impact

    # 4. Loan-to-Annual-Income ratio impact (-60 to +40)
    annual_income = safe_income * 12.0
    lti_ratio = float(loan_amount) / max(annual_income, 1.0)
    if lti_ratio <= 0.5:
        lti_impact = 40.0
    elif lti_ratio <= 1.0:
        lti_impact = 15.0
    elif lti_ratio <= 1.5:
        lti_impact = -20.0
    else:
        lti_impact = -60.0
    score += lti_impact

    credit_score = int(round(max(300.0, min(900.0, score))))
    max_eligible_loan = compute_max_eligible_loan(
        monthly_income=safe_income,
        existing_emi=float(existing_emi),
        tenure_months=int(tenure_months),
        annual_rate=annual_rate,
        max_foir_limit=0.50,
    )

    # Risk Flags
    risk_flags = []
    if int(repayment_score) == 1:
        risk_flags.append("Poor Repayment History — past delays or defaults strongly lower creditworthiness.")
    if existing_foir_pct >= 35.0:
        risk_flags.append(f"High Existing Debt Load — current EMIs already consume {existing_foir_pct}% of monthly income.")
    if foir_pct > 50.0:
        risk_flags.append(f"FOIR Above Safe Ceiling — total EMI burden is {foir_pct}% (recommended <= 50%).")
    if float(loan_amount) > max_eligible_loan:
        risk_flags.append(
            f"Requested Amount Exceeds Eligibility — max safe loan at current tenure & FOIR is ₹{max_eligible_loan:,.0f}."
        )
    if int(employment_stability) == 1:
        risk_flags.append("Short / Unstable Income Track — income stability under 1 year increases underwriting risk.")

    # Risk Tier & Decision
    if credit_score >= 740 and foir_pct <= 45.0 and int(repayment_score) >= 2:
        risk_tier = "Low Risk"
        decision = "Eligible"
    elif credit_score >= 640 and foir_pct <= 55.0 and int(repayment_score) > 1:
        risk_tier = "Moderate Risk"
        decision = "Conditional / Review"
    else:
        risk_tier = "High Risk"
        decision = "High Risk / Needs Adjustment"

    # Actionable Improvement Tips
    tips = []
    if float(loan_amount) > max_eligible_loan and max_eligible_loan > 0:
        tips.append(f"Consider reducing requested loan amount to around ₹{max_eligible_loan:,.0f}.")
    if foir_pct > 45.0 and int(tenure_months) < 60:
        longer_tenure = min(84, int(tenure_months) + 24)
        longer_emi = calculate_monthly_emi(loan_amount, annual_rate, longer_tenure)
        new_foir = round(((float(existing_emi) + longer_emi) / safe_income) * 100, 1)
        tips.append(f"Extending tenure to {longer_tenure} months lowers EMI to ₹{longer_emi:,.0f} (FOIR ~{new_foir}%).")
    if float(existing_emi) > 0 and existing_foir_pct >= 25.0:
        tips.append("Pay down smaller existing loans/credit card dues to free up monthly cash flow.")
    if int(repayment_score) < 3:
        tips.append("Maintain 100% on-time EMI/bill payments over the next 6–12 months to boost score.")
    if int(employment_stability) < 3:
        tips.append("Provide co-applicant support or proof of steady income continuity.")
    if not tips:
        tips.append("Profile is strong! Keep FOIR under 40% and maintain timely repayments.")

    factor_breakdown = [
        {"Factor": "Repayment History", "Score Impact": repayment_impact},
        {"Factor": "FOIR / Debt Ratio", "Score Impact": foir_impact},
        {"Factor": "Employment Stability", "Score Impact": stability_impact},
        {"Factor": "Loan-to-Income", "Score Impact": lti_impact},
    ]

    return {
        "credit_score": credit_score,
        "risk_tier": risk_tier,
        "decision": decision,
        "new_emi": new_emi,
        "total_emi": total_emi,
        "foir_pct": foir_pct,
        "existing_foir_pct": existing_foir_pct,
        "max_eligible_loan": max_eligible_loan,
        "risk_flags": risk_flags,
        "tips": tips,
        "factor_breakdown": factor_breakdown,
    }