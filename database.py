import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "loan_system.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create the loan_evaluations table if it does not already exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS loan_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                applicant_name TEXT NOT NULL,
                loan_purpose TEXT NOT NULL,
                monthly_income REAL NOT NULL,
                existing_emi REAL NOT NULL,
                loan_amount REAL NOT NULL,
                tenure_months INTEGER NOT NULL,
                annual_rate REAL NOT NULL,
                repayment_label TEXT NOT NULL,
                stability_label TEXT NOT NULL,
                credit_score INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                foir_pct REAL NOT NULL,
                monthly_emi REAL NOT NULL,
                max_eligible_loan REAL NOT NULL,
                ml_approval_prob REAL NOT NULL,
                decision TEXT NOT NULL,
                risk_flags_text TEXT NOT NULL,
                tips_text TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_evaluation(record: dict, db_path: str = DB_PATH) -> int:
    """Insert a single loan evaluation into SQLite and return its row ID."""
    init_db(db_path)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO loan_evaluations (
                created_at, applicant_name, loan_purpose, monthly_income,
                existing_emi, loan_amount, tenure_months, annual_rate,
                repayment_label, stability_label, credit_score, risk_tier,
                foir_pct, monthly_emi, max_eligible_loan, ml_approval_prob,
                decision, risk_flags_text, tips_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                record["applicant_name"],
                record["loan_purpose"],
                float(record["monthly_income"]),
                float(record["existing_emi"]),
                float(record["loan_amount"]),
                int(record["tenure_months"]),
                float(record["annual_rate"]),
                record["repayment_label"],
                record["stability_label"],
                int(record["credit_score"]),
                record["risk_tier"],
                float(record["foir_pct"]),
                float(record["monthly_emi"]),
                float(record["max_eligible_loan"]),
                float(record["ml_approval_prob"]),
                record["decision"],
                record["risk_flags_text"],
                record["tips_text"],
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_evaluations(risk_filter: str = "All", search_query: str = "", db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch saved evaluations as a DataFrame with optional risk tier & name search."""
    init_db(db_path)
    query = "SELECT * FROM loan_evaluations WHERE 1=1"
    params: list = []

    if risk_filter and risk_filter != "All":
        query += " AND risk_tier = ?"
        params.append(risk_filter)

    if search_query.strip():
        query += " AND (LOWER(applicant_name) LIKE ? OR LOWER(loan_purpose) LIKE ?)"
        like_term = f"%{search_query.strip().lower()}%"
        params.extend([like_term, like_term])

    query += " ORDER BY id DESC"

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def delete_evaluation(record_id: int, db_path: str = DB_PATH) -> None:
    """Delete a specific evaluation row by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM loan_evaluations WHERE id = ?", (int(record_id),))
        conn.commit()