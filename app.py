import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import delete_evaluation, fetch_evaluations, init_db, save_evaluation
from ml_engine import calculate_monthly_emi, load_or_train_model, predict_approval_probability
from report_generator import build_assessment_report
from scoring import evaluate_credit_profile

st.set_page_config(
    page_title="Credit-Wise Loan System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
_, model_acc = load_or_train_model()

st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 60%, #0d9488 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            color: #ffffff;
            margin-bottom: 1.1rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
        }
        .main-header p {
            margin: 0.35rem 0 0 0;
            color: #e2e8f0;
            font-size: 0.95rem;
        }
        .decision-banner {
            padding: 0.85rem 1.1rem;
            border-radius: 10px;
            font-weight: 600;
            margin: 0.6rem 0 1rem 0;
            border-left: 5px solid;
        }
        .banner-low {
            background-color: #ecfdf5;
            color: #065f46;
            border-color: #10b981;
        }
        .banner-mod {
            background-color: #fffbeb;
            color: #92400e;
            border-color: #f59e0b;
        }
        .banner-high {
            background-color: #fef2f2;
            color: #991b1b;
            border-color: #ef4444;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "applicant_name": "Riya Sharma",
    "loan_purpose": "Education / Skill Upgrade",
    "monthly_income": 65000.0,
    "existing_emi": 8000.0,
    "loan_amount": 350000.0,
    "tenure_months": 36,
    "annual_rate": 10.5,
    "repayment_label": "Good (On-time history)",
    "stability_label": "Stable (3+ yrs / Steady income)",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


SAMPLE_PROFILES = [
    {
        "applicant_name": "Ananya Verma",
        "loan_purpose": "Higher Studies / Laptop & Kit",
        "monthly_income": 85000.0,
        "existing_emi": 6000.0,
        "loan_amount": 300000.0,
        "tenure_months": 36,
        "annual_rate": 10.0,
        "repayment_label": "Good (On-time history)",
        "stability_label": "Stable (3+ yrs / Steady income)",
        "rep_val": 3,
        "stab_val": 3,
    },
    {
        "applicant_name": "Kabir Nair",
        "loan_purpose": "Personal / Medical Buffer",
        "monthly_income": 50000.0,
        "existing_emi": 12000.0,
        "loan_amount": 450000.0,
        "tenure_months": 24,
        "annual_rate": 11.0,
        "repayment_label": "Average (1-2 minor delays)",
        "stability_label": "Moderate (1-3 yrs experience)",
        "rep_val": 2,
        "stab_val": 2,
    },
    {
        "applicant_name": "Devansh Patel",
        "loan_purpose": "Debt Consolidation",
        "monthly_income": 32000.0,
        "existing_emi": 14000.0,
        "loan_amount": 500000.0,
        "tenure_months": 24,
        "annual_rate": 12.5,
        "repayment_label": "Poor (Past defaults / late EMI)",
        "stability_label": "Early / Unstable (<1 yr)",
        "rep_val": 1,
        "stab_val": 1,
    },
]


def seed_sample_records() -> int:
    """Insert three representative demo evaluations into SQLite."""
    inserted = 0
    for prof in SAMPLE_PROFILES:
        sc = evaluate_credit_profile(
            monthly_income=prof["monthly_income"],
            existing_emi=prof["existing_emi"],
            loan_amount=prof["loan_amount"],
            tenure_months=prof["tenure_months"],
            repayment_score=prof["rep_val"],
            employment_stability=prof["stab_val"],
            annual_rate=prof["annual_rate"],
        )
        ml = predict_approval_probability(
            monthly_income=prof["monthly_income"],
            existing_emi=prof["existing_emi"],
            loan_amount=prof["loan_amount"],
            tenure_months=prof["tenure_months"],
            repayment_score=prof["rep_val"],
            employment_stability=prof["stab_val"],
            annual_rate=prof["annual_rate"],
        )
        save_evaluation({
            "applicant_name": prof["applicant_name"],
            "loan_purpose": prof["loan_purpose"],
            "monthly_income": prof["monthly_income"],
            "existing_emi": prof["existing_emi"],
            "loan_amount": prof["loan_amount"],
            "tenure_months": prof["tenure_months"],
            "annual_rate": prof["annual_rate"],
            "repayment_label": prof["repayment_label"],
            "stability_label": prof["stability_label"],
            "credit_score": sc["credit_score"],
            "risk_tier": sc["risk_tier"],
            "foir_pct": sc["foir_pct"],
            "monthly_emi": sc["new_emi"],
            "max_eligible_loan": sc["max_eligible_loan"],
            "ml_approval_prob": ml["approval_probability"],
            "decision": sc["decision"],
            "risk_flags_text": " | ".join(sc["risk_flags"]) if sc["risk_flags"] else "None",
            "tips_text": " | ".join(sc["tips"]),
        })
        inserted += 1
    return inserted


def apply_preset(preset_name: str) -> None:
    if preset_name == "strong":
        st.session_state.update({
            "applicant_name": "Ananya Verma",
            "loan_purpose": "Higher Studies / Laptop & Kit",
            "monthly_income": 85000.0,
            "existing_emi": 6000.0,
            "loan_amount": 300000.0,
            "tenure_months": 36,
            "annual_rate": 10.0,
            "repayment_label": "Good (On-time history)",
            "stability_label": "Stable (3+ yrs / Steady income)",
        })
    elif preset_name == "borderline":
        st.session_state.update({
            "applicant_name": "Kabir Nair",
            "loan_purpose": "Personal / Medical Buffer",
            "monthly_income": 50000.0,
            "existing_emi": 12000.0,
            "loan_amount": 450000.0,
            "tenure_months": 24,
            "annual_rate": 11.0,
            "repayment_label": "Average (1-2 minor delays)",
            "stability_label": "Moderate (1-3 yrs experience)",
        })
    elif preset_name == "high_risk":
        st.session_state.update({
            "applicant_name": "Devansh Patel",
            "loan_purpose": "Debt Consolidation",
            "monthly_income": 32000.0,
            "existing_emi": 14000.0,
            "loan_amount": 500000.0,
            "tenure_months": 24,
            "annual_rate": 12.5,
            "repayment_label": "Poor (Past defaults / late EMI)",
            "stability_label": "Early / Unstable (<1 yr)",
        })


def build_yearly_amortization(principal: float, annual_rate: float, tenure_months: int) -> pd.DataFrame:
    """Build year-wise principal, interest, and ending balance schedule."""
    if principal <= 0 or tenure_months <= 0:
        return pd.DataFrame(columns=["Year", "Principal Paid (₹)", "Interest Paid (₹)", "Remaining Balance (₹)"])

    emi = calculate_monthly_emi(principal, annual_rate, tenure_months)
    monthly_rate = annual_rate / (12 * 100)
    balance = float(principal)

    rows = []
    year_num = 1
    yr_principal = 0.0
    yr_interest = 0.0

    for month in range(1, int(tenure_months) + 1):
        interest_part = balance * monthly_rate if monthly_rate > 0 else 0.0
        principal_part = min(balance, max(0.0, emi - interest_part))
        balance = max(0.0, balance - principal_part)

        yr_principal += principal_part
        yr_interest += interest_part

        if month % 12 == 0 or month == int(tenure_months):
            rows.append({
                "Year": f"Year {year_num}",
                "Principal Paid (₹)": round(yr_principal, 0),
                "Interest Paid (₹)": round(yr_interest, 0),
                "Remaining Balance (₹)": round(balance, 0),
            })
            year_num += 1
            yr_principal = 0.0
            yr_interest = 0.0

    return pd.DataFrame(rows)


def build_model_rule_comparison(credit_score: int, ml_prob: float) -> str:
    """Compare rule-based credit score tier with ML approval probability."""
    rule_positive = credit_score >= 640
    ml_positive = ml_prob >= 50.0

    if rule_positive and ml_positive:
        return (
            f"**Aligned Positive Signal:** Rule Score ({credit_score}/900) and ML Approval Probability ({ml_prob:.1f}%) "
            "both support creditworthiness."
        )
    if (not rule_positive) and (not ml_positive):
        return (
            f"**Aligned Caution Signal:** Both Rule Score ({credit_score}/900) and ML Approval Probability ({ml_prob:.1f}%) "
            "flag elevated risk — adjusting loan amount, tenure, or debt load is recommended."
        )
    return (
        f"**Mixed / Borderline Signal:** Rule Score ({credit_score}/900) and ML Approval Probability ({ml_prob:.1f}%) "
        "show a borderline split — check FOIR, repayment history, and loan-to-income balance closely."
    )


REPAYMENT_OPTIONS = {
    "Good (On-time history)": 3,
    "Average (1-2 minor delays)": 2,
    "Poor (Past defaults / late EMI)": 1,
}

STABILITY_OPTIONS = {
    "Stable (3+ yrs / Steady income)": 3,
    "Moderate (1-3 yrs experience)": 2,
    "Early / Unstable (<1 yr)": 1,
}

st.markdown(
    f"""
    <div class="main-header">
        <h1>Credit-Wise Loan System</h1>
        <p>Explainable Credit Scoring (300–900) • FOIR & Risk Flag Engine • Random Forest Approval Probability ({model_acc:.1f}% Test Acc) • SQLite History</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Quick Demo Profiles")
    st.caption("Load realistic scenarios in one click during demo:")
    if st.button("1. Strong Borrower", width="stretch"):
        apply_preset("strong")
        st.rerun()
    if st.button("2. Borderline / High FOIR", width="stretch"):
        apply_preset("borderline")
        st.rerun()
    if st.button("3. High Risk Profile", width="stretch"):
        apply_preset("high_risk")
        st.rerun()

    st.divider()
    st.markdown("**Architecture Modules**")
    st.markdown(
        "- `ml_engine.py` — Random Forest model\n"
        "- `scoring.py` — Score, FOIR & tips\n"
        "- `database.py` — SQLite persistence\n"
        "- `report_generator.py` — Summary export"
    )

tab1, tab2, tab3 = st.tabs([
    "📊 Credit & Loan Evaluator",
    "🎛️ Live What-If Simulator",
    "🗂️ Saved Applications & Analytics",
])

with tab1:
    left_col, right_col = st.columns([1.05, 1.45], gap="large")

    with left_col:
        st.subheader("Applicant & Loan Details")
        applicant_name = st.text_input("Applicant Name", key="applicant_name")
        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Education / Skill Upgrade",
                "Higher Studies / Laptop & Kit",
                "Personal / Medical Buffer",
                "Small Business / Freelance Setup",
                "Vehicle / Home Improvement",
                "Debt Consolidation",
            ],
            key="loan_purpose",
        )

        c1, c2 = st.columns(2)
        with c1:
            monthly_income = st.number_input(
                "Monthly Net Income (₹)",
                min_value=5000.0,
                max_value=2000000.0,
                step=2500.0,
                key="monthly_income",
            )
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=10000.0,
                max_value=5000000.0,
                step=10000.0,
                key="loan_amount",
            )
            repayment_label = st.selectbox(
                "Repayment History",
                list(REPAYMENT_OPTIONS.keys()),
                key="repayment_label",
            )
        with c2:
            existing_emi = st.number_input(
                "Existing Monthly EMI / Debt (₹)",
                min_value=0.0,
                max_value=1500000.0,
                step=1000.0,
                key="existing_emi",
            )
            tenure_months = st.selectbox(
                "Tenure (Months)",
                [12, 24, 36, 48, 60, 84],
                key="tenure_months",
            )
            stability_label = st.selectbox(
                "Employment / Income Stability",
                list(STABILITY_OPTIONS.keys()),
                key="stability_label",
            )

        annual_rate = st.slider(
            "Assumed Annual Interest Rate (% p.a.)",
            min_value=7.5,
            max_value=18.0,
            step=0.25,
            key="annual_rate",
        )

    rep_val = REPAYMENT_OPTIONS[repayment_label]
    stab_val = STABILITY_OPTIONS[stability_label]

    scoring_res = evaluate_credit_profile(
        monthly_income=monthly_income,
        existing_emi=existing_emi,
        loan_amount=loan_amount,
        tenure_months=int(tenure_months),
        repayment_score=rep_val,
        employment_stability=stab_val,
        annual_rate=float(annual_rate),
    )

    ml_res = predict_approval_probability(
        monthly_income=monthly_income,
        existing_emi=existing_emi,
        loan_amount=loan_amount,
        tenure_months=int(tenure_months),
        repayment_score=rep_val,
        employment_stability=stab_val,
        annual_rate=float(annual_rate),
    )

    with right_col:
        st.subheader("Credit & Eligibility Assessment")

        m1, m2, m3 = st.columns(3)
        m1.metric("Credit Score", f"{scoring_res['credit_score']} / 900")
        m2.metric("ML Approval Prob.", f"{ml_res['approval_probability']:.1f}%")
        m3.metric("Total FOIR", f"{scoring_res['foir_pct']:.1f}%")

        m4, m5, m6 = st.columns(3)
        m4.metric("New Monthly EMI", f"₹{scoring_res['new_emi']:,.0f}")
        m5.metric("Total Monthly EMI", f"₹{scoring_res['total_emi']:,.0f}")
        m6.metric("Max Eligible Loan", f"₹{scoring_res['max_eligible_loan']:,.0f}")

        tier = scoring_res["risk_tier"]
        if tier == "Low Risk":
            banner_cls = "banner-low"
        elif tier == "Moderate Risk":
            banner_cls = "banner-mod"
        else:
            banner_cls = "banner-high"

        st.markdown(
            f"""
            <div class="decision-banner {banner_cls}">
                Risk Category: {tier} &nbsp;|&nbsp; System Status: {scoring_res['decision']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(build_model_rule_comparison(scoring_res["credit_score"], ml_res["approval_probability"]))

        ch1, ch2 = st.columns(2)
        with ch1:
            fb_df = pd.DataFrame(scoring_res["factor_breakdown"])
            colors = ["#10b981" if val >= 0 else "#ef4444" for val in fb_df["Score Impact"]]
            fig_impact = go.Figure(
                go.Bar(
                    x=fb_df["Score Impact"],
                    y=fb_df["Factor"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.0f}" for v in fb_df["Score Impact"]],
                    textposition="outside",
                )
            )
            fig_impact.update_layout(
                title="Score Factor Impact (pts)",
                height=250,
                margin=dict(l=10, r=35, t=40, b=10),
                xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#64748b"),
            )
            st.plotly_chart(fig_impact, width="stretch")

        with ch2:
            remaining_income = max(0.0, float(monthly_income) - float(scoring_res["total_emi"]))
            fig_emi = go.Figure(
                go.Bar(
                    x=["Existing EMI", "New Loan EMI", "Remaining Income"],
                    y=[float(existing_emi), float(scoring_res["new_emi"]), remaining_income],
                    marker_color=["#f59e0b", "#3b82f6", "#14b8a6"],
                    text=[f"₹{v:,.0f}" for v in [existing_emi, scoring_res["new_emi"], remaining_income]],
                    textposition="outside",
                )
            )
            fig_emi.update_layout(
                title="Monthly Income Allocation (₹)",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_emi, width="stretch")

        rf_col, tip_col = st.columns(2)
        with rf_col:
            st.markdown("**Risk Flags**")
            if scoring_res["risk_flags"]:
                for flag in scoring_res["risk_flags"]:
                    st.warning(flag)
            else:
                st.success("No major risk flags detected for this profile.")

        with tip_col:
            st.markdown("**Actionable Improvement Tips**")
            for tip in scoring_res["tips"]:
                st.info(tip)

        report_text = build_assessment_report(
            applicant_name=applicant_name,
            loan_purpose=loan_purpose,
            monthly_income=float(monthly_income),
            existing_emi=float(existing_emi),
            loan_amount=float(loan_amount),
            tenure_months=int(tenure_months),
            annual_rate=float(annual_rate),
            repayment_label=repayment_label,
            stability_label=stability_label,
            scoring_result=scoring_res,
            ml_result=ml_res,
        )

        act1, act2 = st.columns(2)
        with act1:
            if st.button("💾 Save Evaluation to Database", type="primary", width="stretch"):
                row_id = save_evaluation({
                    "applicant_name": applicant_name.strip() or "Unnamed Applicant",
                    "loan_purpose": loan_purpose,
                    "monthly_income": monthly_income,
                    "existing_emi": existing_emi,
                    "loan_amount": loan_amount,
                    "tenure_months": tenure_months,
                    "annual_rate": annual_rate,
                    "repayment_label": repayment_label,
                    "stability_label": stability_label,
                    "credit_score": scoring_res["credit_score"],
                    "risk_tier": scoring_res["risk_tier"],
                    "foir_pct": scoring_res["foir_pct"],
                    "monthly_emi": scoring_res["new_emi"],
                    "max_eligible_loan": scoring_res["max_eligible_loan"],
                    "ml_approval_prob": ml_res["approval_probability"],
                    "decision": scoring_res["decision"],
                    "risk_flags_text": " | ".join(scoring_res["risk_flags"]) if scoring_res["risk_flags"] else "None",
                    "tips_text": " | ".join(scoring_res["tips"]),
                })
                st.session_state["last_saved_msg"] = f"Saved evaluation #{row_id} for {applicant_name.strip() or 'Applicant'}!"
                st.toast(st.session_state["last_saved_msg"], icon="💾")

            if st.session_state.get("last_saved_msg"):
                st.success(st.session_state["last_saved_msg"])

        with act2:
            safe_filename = (applicant_name.strip() or "applicant").lower().replace(" ", "_")
            st.download_button(
                label="📄 Download Assessment Report (.txt)",
                data=report_text,
                file_name=f"credit_report_{safe_filename}.txt",
                mime="text/plain",
                width="stretch",
            )

    st.divider()
    st.subheader("Loan Repayment Schedule (Principal vs. Interest)")
    amort_df = build_yearly_amortization(float(loan_amount), float(annual_rate), int(tenure_months))
    if not amort_df.empty:
        am_c1, am_c2 = st.columns([1.35, 1])
        with am_c1:
            fig_amort = go.Figure()
            fig_amort.add_trace(
                go.Bar(
                    name="Principal Paid",
                    x=amort_df["Year"],
                    y=amort_df["Principal Paid (₹)"],
                    marker_color="#0d9488",
                )
            )
            fig_amort.add_trace(
                go.Bar(
                    name="Interest Paid",
                    x=amort_df["Year"],
                    y=amort_df["Interest Paid (₹)"],
                    marker_color="#f59e0b",
                )
            )
            fig_amort.update_layout(
                barmode="stack",
                title="Year-Wise Principal vs. Interest Breakdown (₹)",
                height=270,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_amort, width="stretch")

        with am_c2:
            st.markdown("**Yearly Amortization Summary**")
            st.dataframe(amort_df, width="stretch", hide_index=True)

with tab2:
    st.subheader("Interactive 'What-If' Loan Adjustment Simulator")
    st.caption(
        "Adjust loan amount and tenure dynamically using current applicant income, existing EMI, and credit profile."
    )

    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        sim_amount = st.slider(
            "Simulated Loan Amount (₹)",
            min_value=25000,
            max_value=2500000,
            value=int(min(2500000, max(25000, loan_amount))),
            step=25000,
        )
    with sim_c2:
        sim_tenure = st.slider(
            "Simulated Tenure (Months)",
            min_value=12,
            max_value=84,
            value=int(tenure_months),
            step=6,
        )

    sim_scoring = evaluate_credit_profile(
        monthly_income=monthly_income,
        existing_emi=existing_emi,
        loan_amount=float(sim_amount),
        tenure_months=int(sim_tenure),
        repayment_score=rep_val,
        employment_stability=stab_val,
        annual_rate=float(annual_rate),
    )
    sim_ml = predict_approval_probability(
        monthly_income=monthly_income,
        existing_emi=existing_emi,
        loan_amount=float(sim_amount),
        tenure_months=int(sim_tenure),
        repayment_score=rep_val,
        employment_stability=stab_val,
        annual_rate=float(annual_rate),
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Simulated EMI",
        f"₹{sim_scoring['new_emi']:,.0f}",
        delta=f"₹{sim_scoring['new_emi'] - scoring_res['new_emi']:+,.0f}",
        delta_color="inverse",
    )
    s2.metric(
        "Simulated FOIR",
        f"{sim_scoring['foir_pct']:.1f}%",
        delta=f"{sim_scoring['foir_pct'] - scoring_res['foir_pct']:+.1f}%",
        delta_color="inverse",
    )
    s3.metric(
        "Simulated Score",
        f"{sim_scoring['credit_score']}",
        delta=f"{sim_scoring['credit_score'] - scoring_res['credit_score']:+d} pts",
    )
    s4.metric(
        "Simulated ML Prob.",
        f"{sim_ml['approval_probability']:.1f}%",
        delta=f"{sim_ml['approval_probability'] - ml_res['approval_probability']:+.1f}%",
    )

    st.caption("Tip: Move the sliders above to compare changes against the baseline profile in Tab 1.")
    st.info(f"**Simulated Risk Tier & Decision:** {sim_scoring['risk_tier']} — *{sim_scoring['decision']}*")

    st.markdown("#### Random Forest Global Feature Importance")
    imp_df = ml_res["importance_df"]
    fig_imp = go.Figure(
        go.Bar(
            x=imp_df["Importance (%)"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color="#1e3a8a",
            text=[f"{v:.1f}%" for v in imp_df["Importance (%)"]],
            textposition="outside",
        )
    )
    fig_imp.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=20, b=10),
    )
    st.plotly_chart(fig_imp, width="stretch")

with tab3:
    st.subheader("Saved Evaluations History & Analytics (SQLite)")

    f1, f2, f3 = st.columns([1.2, 2, 1.1])
    with f1:
        risk_filter = st.selectbox("Filter by Risk Category", ["All", "Low Risk", "Moderate Risk", "High Risk"])
    with f2:
        search_q = st.text_input("Search by Applicant Name or Purpose", placeholder="e.g., Riya or Education")
    with f3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Load Sample Records", width="stretch"):
            count = seed_sample_records()
            st.success(f"Added {count} sample records to SQLite.")
            st.rerun()

    hist_df = fetch_evaluations(risk_filter=risk_filter, search_query=search_q)

    if hist_df.empty:
        st.info("No saved evaluations yet. Click 'Load Sample Records' or evaluate a profile in Tab 1 and save it.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Saved Records", len(hist_df))
        k2.metric("Average Credit Score", f"{hist_df['credit_score'].mean():.0f} / 900")
        k3.metric("Average ML Approval Prob.", f"{hist_df['ml_approval_prob'].mean():.1f}%")

        tier_counts = hist_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Category", "Count"]
        fig_hist = go.Figure(
            go.Bar(
                x=tier_counts["Risk Category"],
                y=tier_counts["Count"],
                marker_color=["#0d9488", "#f59e0b", "#ef4444"][: len(tier_counts)],
                text=tier_counts["Count"],
                textposition="outside",
            )
        )
        fig_hist.update_layout(
            title="Saved Applications by Risk Category",
            height=240,
            margin=dict(l=10, r=10, t=38, b=10),
        )
        st.plotly_chart(fig_hist, width="stretch")

        display_cols = [
            "id",
            "created_at",
            "applicant_name",
            "loan_purpose",
            "monthly_income",
            "loan_amount",
            "tenure_months",
            "credit_score",
            "foir_pct",
            "ml_approval_prob",
            "risk_tier",
            "decision",
        ]
        st.dataframe(hist_df[display_cols], width="stretch", hide_index=True)

        del_col1, del_col2 = st.columns([1, 3])
        with del_col1:
            record_to_delete = st.selectbox("Select Record ID to Delete", hist_df["id"].tolist())
        with del_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete Selected Record"):
                delete_evaluation(int(record_to_delete))
                st.success(f"Deleted record #{record_to_delete}.")
                st.rerun()