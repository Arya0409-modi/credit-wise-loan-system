from datetime import datetime


def build_assessment_report(
    applicant_name: str,
    loan_purpose: str,
    monthly_income: float,
    existing_emi: float,
    loan_amount: float,
    tenure_months: int,
    annual_rate: float,
    repayment_label: str,
    stability_label: str,
    scoring_result: dict,
    ml_result: dict,
) -> str:
    """Create a formatted text report summarizing credit & loan evaluation."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name_clean = applicant_name.strip() if applicant_name.strip() else "Applicant"

    flags = scoring_result.get("risk_flags", [])
    tips = scoring_result.get("tips", [])

    flags_block = (
        "\n".join(f"  - {flag}" for flag in flags)
        if flags
        else "  - No major risk flags detected."
    )
    tips_block = "\n".join(f"  - {tip}" for tip in tips)

    report = f"""============================================================
CREDIT-WISE LOAN SYSTEM — ASSESSMENT SUMMARY
============================================================
Generated On          : {generated_at}
Applicant Name        : {name_clean}
Loan Purpose          : {loan_purpose}

--- APPLICANT FINANCIAL & CREDIT PROFILE ---
Monthly Income        : Rs. {monthly_income:,.2f}
Existing Monthly EMI  : Rs. {existing_emi:,.2f}
Requested Loan Amount : Rs. {loan_amount:,.2f}
Tenure                : {tenure_months} months
Assumed Interest Rate : {annual_rate:.2f}% p.a.
Repayment History     : {repayment_label}
Employment Stability  : {stability_label}

--- EVALUATION RESULTS ---
Rule-Based Score      : {scoring_result['credit_score']} / 900
Risk Category         : {scoring_result['risk_tier']}
System Decision       : {scoring_result['decision']}
Calculated Monthly EMI: Rs. {scoring_result['new_emi']:,.2f}
Total Monthly EMI     : Rs. {scoring_result['total_emi']:,.2f}
FOIR Ratio            : {scoring_result['foir_pct']:.1f}% (Existing: {scoring_result['existing_foir_pct']:.1f}%)
Max Eligible Loan     : Rs. {scoring_result['max_eligible_loan']:,.2f}
ML Approval Prob.     : {ml_result['approval_probability']:.1f}% (RF Model Acc: {ml_result['model_accuracy']:.1f}%)

--- RISK FLAGS ---
{flags_block}

--- ACTIONABLE IMPROVEMENT TIPS ---
{tips_block}
============================================================
"""
    return report.strip()